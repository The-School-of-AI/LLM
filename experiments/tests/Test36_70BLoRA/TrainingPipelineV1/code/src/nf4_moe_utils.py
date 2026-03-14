"""
NF4 MoE Integration Utilities.

Provides functions to:
1. Quantize existing MoE expert weights (W_gate, W_up, W_down) to NF4
2. Immediately dequantize back to bf16 compute buffers for fast forward pass
3. Patch MoEFFN modules to use the bf16 buffers + LoRA adapters

Strategy:
    - Expert weights are quantized to NF4 for storage compression
    - Dequantized bf16 copies are cached as non-parameter buffers for compute
    - Forward pass uses cached bf16 weights → identical speed to non-NF4 baseline
    - NF4 packed data is retained for checkpoint saving
    - No optimizer state for frozen base weights (memory savings)

    For 70B (E=260): uses fused NF4 Triton kernel (tile-by-tile dequant in SRAM)
    to avoid materializing the full [260, K, N] bf16 tensor (~10GB/projection).

Usage:
    from src.nf4_moe_utils import quantize_moe_experts, patch_moe_nf4_forward

    quantize_moe_experts(model, config)
    patch_moe_nf4_forward(model)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from .kernels.nf4_quantize import (
    NF4QuantConfig,
    NF4Parameter,
    quantize_tensor_nf4,
)

logger = logging.getLogger(__name__)

# Expert count threshold: below this, cache dequantized bf16 weights for speed.
# Above this (70B with 260 experts), use on-the-fly dequant to save memory.
_NF4_CACHE_THRESHOLD = 64


# ============================================================================
# V5: Custom autograd for NF4 frozen base weights + LoRA
# ============================================================================
# Key insight: In QLoRA, base weights are FROZEN. The standard
# FusedLoRAGroupedGEMMFn wastes compute on:
#   1. grad_W_base = _grouped_gemm_dweight(x, grad_output) — full dweight for
#      frozen weights that will never be updated
#   2. Saves W_base [E_active, K, N] bf16 in autograd graph for backward —
#      ~2 GB per projection × 3 projections × 20 layers = ~120 GB
#
# NF4FrozenLoRAGroupedGEMMFn eliminates both:
#   - Forward: dequant NF4→bf16, compute base+LoRA GEMM, save only NF4 packed
#     refs (NOT the bf16 weights) for backward
#   - Backward: re-dequant from NF4 for dx = grad_output @ W^T, compute LoRA
#     grads, SKIP grad_W_base entirely
#   - Saves: 60 fewer _grouped_gemm_dweight calls/step + ~120 GB less autograd
#     tensor storage


class NF4FrozenLoRAGroupedGEMMFn(torch.autograd.Function):
    """
    NF4-aware fused LoRA grouped GEMM with frozen base weights.

    Forward: dequant NF4→bf16 → base GEMM + LoRA GEMM → result
    Backward: re-dequant NF4→bf16 → dx via W^T, LoRA grads only (no grad_W_base)

    Saves NF4 packed data (compact) instead of bf16 weights (large) in autograd.
    """

    @staticmethod
    def forward(ctx, x, w_nf4_packed, w_nf4_absmax, lora_A, lora_B,
                counts, offsets, max_M, E, K, N, block_size, scaling):
        """
        Args:
            x: [M_total, K] sorted tokens
            w_nf4_packed: [E, K*N//2] uint8 NF4 packed weights
            w_nf4_absmax: [E, blocks] float absmax per block
            lora_A: [E, rank, K] LoRA down-projection
            lora_B: [E, N, rank] LoRA up-projection
            counts: [E] int64 tokens per expert
            offsets: [E+1] int64 cumulative offsets
            max_M: int
            E, K, N: int dimensions
            block_size: int NF4 block size
            scaling: float LoRA alpha/rank
        """
        from .kernels.triton_moe_grouped_gemm import _grouped_gemm_forward

        # Dequant NF4 → bf16
        W_bf16 = _dequant_nf4_batched(
            w_nf4_packed, w_nf4_absmax, (E, K, N), block_size, x.dtype
        )

        # Base GEMM: x @ W_base[e].T → [M_total, N]
        base_out = _grouped_gemm_forward(x, W_bf16, offsets, E, max_M)

        # LoRA: x @ A[e].T @ B[e].T * scaling
        A_t = lora_A.transpose(-2, -1).contiguous()  # [E, K, rank]
        lora_mid = _grouped_gemm_forward(x, A_t, offsets, E, max_M)  # [M_total, rank]
        B_t = lora_B.transpose(-2, -1).contiguous()  # [E, rank, N]
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)  # [M_total, N]

        result = base_out + lora_out * scaling

        # Free bf16 weights — NOT saved for backward (key memory saving)
        del W_bf16, base_out, lora_out, A_t, B_t

        # Save compact NF4 refs + LoRA params for backward
        ctx.save_for_backward(x, w_nf4_packed, w_nf4_absmax,
                              lora_A, lora_B, lora_mid, counts, offsets)
        ctx.max_M = max_M
        ctx.E = E
        ctx.K = K
        ctx.N = N
        ctx.block_size = block_size
        ctx.scaling = scaling
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x, w_nf4_packed, w_nf4_absmax,
         lora_A, lora_B, lora_mid, counts, offsets) = ctx.saved_tensors
        max_M = ctx.max_M
        E = ctx.E
        K = ctx.K
        N = ctx.N
        block_size = ctx.block_size
        scaling = ctx.scaling

        from .kernels.triton_moe_grouped_gemm import (
            _grouped_gemm_forward, _grouped_gemm_dweight,
        )

        grad_output = grad_output.contiguous()
        rank = lora_A.shape[1]

        # ── LoRA B gradient: dB[e] = (go * scaling)^T @ lora_mid ──────────
        go_scaled = (grad_output * scaling).contiguous()
        grad_lora_B = _grouped_gemm_dweight(
            go_scaled, lora_mid, offsets, E, N, rank, max_M, lora_B.dtype
        )  # [E, N, rank]

        # ── LoRA A gradient: dA[e] = ((go * scaling) @ B[e])^T @ x ────────
        # grad_lora_mid = go_scaled @ B: [M_e, N] @ [N, rank] = [M_e, rank]
        grad_lora_mid = _grouped_gemm_forward(
            go_scaled, lora_B, offsets, E, max_M
        )  # [M_total, rank]
        del go_scaled

        grad_lora_A = _grouped_gemm_dweight(
            grad_lora_mid, x, offsets, E, rank, K, max_M, lora_A.dtype
        )  # [E, rank, K]

        # ── Input gradient: dx = go @ W_base^T + grad_lora_mid @ A ────────
        # Re-dequant NF4 → bf16 for W^T (the key trade: recompute vs store)
        W_bf16 = _dequant_nf4_batched(
            w_nf4_packed, w_nf4_absmax, (E, K, N), block_size, x.dtype
        )
        W_bf16_t = W_bf16.transpose(-2, -1).contiguous()  # [E, N, K]
        del W_bf16

        grad_x = _grouped_gemm_forward(
            grad_output, W_bf16_t, offsets, E, max_M
        )  # [M_total, K]
        del W_bf16_t

        # LoRA contribution to dx
        grad_x_lora = _grouped_gemm_forward(
            grad_lora_mid, lora_A, offsets, E, max_M
        )  # [M_total, K]
        grad_x = grad_x + grad_x_lora
        del grad_x_lora, grad_lora_mid

        # NO grad_W_base — base weights are frozen in QLoRA

        # Returns: grad for each forward arg
        # (x, w_nf4_packed, w_nf4_absmax, lora_A, lora_B,
        #  counts, offsets, max_M, E, K, N, block_size, scaling)
        return (grad_x, None, None, grad_lora_A, grad_lora_B,
                None, None, None, None, None, None, None, None)


def nf4_frozen_lora_grouped_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    absmax: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_counts,
    K: int,
    N: int,
    scaling: float,
    block_size: int,
) -> torch.Tensor:
    """
    NF4 frozen-base LoRA grouped GEMM wrapper.

    Dequant happens INSIDE the autograd function so bf16 weights are never
    saved in the autograd graph. Re-dequant in backward trades ~2ms compute
    for ~2 GB memory savings per projection.

    Args:
        x: [M_total, in_features] sorted tokens
        w_packed: [E, K*N//2] uint8 NF4 packed weights
        absmax: [E, blocks] float absmax
        lora_A: [E, rank, K] LoRA down
        lora_B: [E, N, rank] LoRA up
        expert_counts: [E] tensor or list
        K, N: int weight dimensions
        scaling: float alpha/rank
        block_size: int NF4 block size

    Returns: [M_total, N]
    """
    x = x.contiguous()
    E = w_packed.shape[0]

    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.to(device=x.device, dtype=torch.int64).contiguous()
    else:
        counts = torch.tensor(expert_counts, device=x.device, dtype=torch.int64)

    offsets = torch.zeros(E + 1, device=x.device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    max_M = int(counts.max().item()) if counts.numel() > 0 else 0

    return NF4FrozenLoRAGroupedGEMMFn.apply(
        x, w_packed, absmax, lora_A, lora_B,
        counts, offsets, max_M, E, K, N, block_size, scaling,
    )


def _dequant_nf4_batched(packed, absmax, shape, block_size, compute_dtype):
    """Batched vectorized NF4 dequantization — optimized with byte lookup table.

    Uses a precomputed 256-entry lookup table that maps each packed byte directly
    to two dequantized values, eliminating nibble extraction, int16/int64 index
    tensors, and the 16-element NF4 lookup. This reduces peak memory by ~4× and
    compute by ~2× vs the naive approach.

    Processes experts in chunks to limit peak memory on large-E models.

    Args:
        packed: [E, K*N//2] uint8
        absmax: [E, blocks_per_expert] float
        shape: (E, K, N)
        block_size: int
        compute_dtype: torch.dtype

    Returns: [E, K, N] tensor in compute_dtype
    """
    E, K, N = shape
    numel = K * N
    half_numel = numel // 2  # packed bytes per expert

    # Build byte→(val_high, val_low) lookup table: 256 entries × 2 float32 values
    # Each byte encodes two 4-bit NF4 indices. Precompute both dequantized values.
    lut = _get_nf4_byte_lut(packed.device)  # [256, 2] float32

    CHUNK_E = 32
    out = torch.empty(E, K, N, device=packed.device, dtype=compute_dtype)

    for c_start in range(0, E, CHUNK_E):
        c_end = min(c_start + CHUNK_E, E)
        c_size = c_end - c_start

        p = packed[c_start:c_end]       # [c_size, half_numel] uint8
        a = absmax[c_start:c_end]       # [c_size, blocks]

        # Direct byte lookup: [c_size, half_numel] → [c_size, half_numel, 2] float32
        pair_vals = lut[p.long()]       # [c_size, half_numel, 2]

        # Reshape to interleaved: [c_size, numel]
        values = pair_vals.reshape(c_size, -1)  # high0,low0,high1,low1,...
        del pair_vals
        values = values[:, :numel]

        # Apply per-block absmax scaling
        pad = (block_size - numel % block_size) % block_size
        padded_total = numel + pad
        if pad > 0:
            values = torch.nn.functional.pad(values, (0, pad))
        num_blocks = padded_total // block_size
        values = values.view(c_size, num_blocks, block_size)
        values = values * a[:, :num_blocks].unsqueeze(2)
        values = values.reshape(c_size, -1)[:, :numel]

        out[c_start:c_end] = values.view(c_size, K, N).to(compute_dtype)
        del values

    return out


# Cache the byte lookup table per device
_NF4_BYTE_LUT_CACHE = {}

def _get_nf4_byte_lut(device):
    """Get or create the 256-entry NF4 byte lookup table for a device.

    Returns: [256, 2] float32 tensor where entry[b] = (nf4_val[b>>4], nf4_val[b&0xF])
    """
    dev_key = str(device)
    if dev_key not in _NF4_BYTE_LUT_CACHE:
        from .kernels.nf4_quantize import NF4_LEVELS
        nf4 = NF4_LEVELS.float()  # [16] float32
        lut = torch.empty(256, 2, dtype=torch.float32)
        for b in range(256):
            lut[b, 0] = nf4[b >> 4]    # high nibble
            lut[b, 1] = nf4[b & 0x0F]  # low nibble
        _NF4_BYTE_LUT_CACHE[dev_key] = lut.to(device)
    return _NF4_BYTE_LUT_CACHE[dev_key]


def quantize_moe_experts(
    model: nn.Module,
    config: Optional[NF4QuantConfig] = None,
) -> int:
    """
    Quantize MoE expert base weights to NF4 in-place.

    For each MoEFFN module with W_gate, W_up, W_down [E, K, N] parameters:
    1. Quantize to NF4 packed format (stored as buffers for checkpointing)
    2. If E <= _NF4_CACHE_THRESHOLD: dequantize back to bf16 and cache as
       non-parameter buffer `W_gate` (replaces the original nn.Parameter).
       Forward path uses these directly — zero overhead vs baseline.
    3. If E > threshold: only NF4 packed data is stored. Forward uses
       fused Triton kernel for tile-by-tile dequant.

    Returns: number of expert weight tensors quantized
    """
    if config is None:
        config = NF4QuantConfig()

    count = 0
    for mod_name, module in model.named_modules():
        expert_params = ['W_gate', 'W_up', 'W_down']
        has_experts = all(hasattr(module, p) for p in expert_params)
        if not has_experts:
            continue

        W_gate = getattr(module, 'W_gate')
        if not isinstance(W_gate, nn.Parameter):
            continue

        shapes = {}
        for pname in expert_params:
            param = getattr(module, pname)
            shape = getattr(param, 'ds_shape', param.shape)
            if len(shape) != 3:
                break
            shapes[pname] = shape
        else:
            E = shapes['W_gate'][0]
            use_cache = (E <= _NF4_CACHE_THRESHOLD)

            for pname in expert_params:
                param = getattr(module, pname)
                original_shape = shapes[pname]
                E, K, N = original_shape

                if hasattr(param, 'ds_id'):
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([param]):
                        full_data = param.data.clone()
                else:
                    full_data = param.data

                # Quantize each expert separately → [E, ...] packed layout
                packed_list = []
                absmax_list = []
                for e in range(E):
                    nf4_e = quantize_tensor_nf4(full_data[e], config)
                    packed_list.append(nf4_e.packed)
                    absmax_list.append(nf4_e.absmax)

                packed_stacked = torch.stack(packed_list, dim=0)
                absmax_stacked = torch.stack(absmax_list, dim=0)

                # Store NF4 packed data (for checkpointing and large-E compute)
                module.register_buffer(f'{pname}_nf4_packed', packed_stacked)
                module.register_buffer(f'{pname}_nf4_absmax', absmax_stacked)
                setattr(module, f'{pname}_nf4_shape', original_shape)
                setattr(module, f'{pname}_nf4_numel', E * K * N)
                setattr(module, f'{pname}_nf4_block_size', config.block_size)

                # Delete original nn.Parameter
                delattr(module, pname)

                if use_cache:
                    # Dequantize back to bf16 and cache as a plain buffer.
                    # This is the key optimization: forward reads bf16 directly,
                    # no per-step dequant overhead.
                    bf16_cached = _dequant_nf4_batched(
                        packed_stacked, absmax_stacked, original_shape,
                        config.block_size, config.compute_dtype
                    )
                    module.register_buffer(pname, bf16_cached)

                count += 1
                bf16_mb = E * K * N * 2 / 1e6
                nf4_mb = (packed_stacked.nbytes + absmax_stacked.nbytes) / 1e6
                logger.info(
                    f"  NF4: {mod_name}.{pname} [{E}×{K}×{N}] "
                    f"bf16={bf16_mb:.1f}MB → nf4={nf4_mb:.1f}MB "
                    f"({(1 - nf4_mb/bf16_mb)*100:.0f}% savings)"
                    f"{' [cached bf16]' if use_cache else ' [on-the-fly]'}"
                )

            module._nf4_quantized = True
            module._nf4_block_size = config.block_size
            module._nf4_compute_dtype = config.compute_dtype
            module._nf4_use_cache = use_cache

    return count


def _moe_grouped_nf4(module, sorted_x, expert_counts):
    """
    NF4-aware MoE grouped forward path.

    Strategy by model size:
    - Cached bf16 (E <= threshold, e.g. 8B with E=20): Uses the SAME code path
      as the original _moe_grouped — fused_moe_gate_up_silu or triton_grouped_gemm
      with the cached bf16 weight buffers. LoRA is handled identically to baseline
      (the 8B model's original _moe_grouped doesn't apply LoRA inline; the 70B does).
    - Large E (> threshold, e.g. 70B with E=260): Uses fused NF4 Triton kernel
      for tile-by-tile dequant + LoRA to avoid materializing full weight tensor.
    """
    try:
        from .models.liger_ops import liger_silu_mul
    except ImportError:
        liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

    # ── Fast path: cached bf16 weights ──────────────────────────────────────
    # Call the ORIGINAL _moe_grouped (saved before monkey-patching).
    # This preserves the exact baseline behavior: 8B model doesn't apply LoRA
    # inline, 70B model does. Zero overhead vs baseline.
    if getattr(module, '_nf4_use_cache', False):
        original_fn = getattr(module, '_nf4_original_moe_grouped', None)
        if original_fn is not None:
            if not getattr(module, '_nf4_cache_logged', False):
                logger.info("[NF4-V3] Using CACHED bf16 + original _moe_grouped (zero-overhead path)")
                module._nf4_cache_logged = True
            return original_fn(sorted_x, expert_counts)
        # Fallback if original wasn't saved (shouldn't happen)
        if not getattr(module, '_nf4_cache_logged', False):
            logger.info("[NF4-V3] Using CACHED bf16 + _run_baseline_grouped fallback")
            module._nf4_cache_logged = True
        return _run_baseline_grouped(
            module, sorted_x, expert_counts, liger_silu_mul
        )

    # ── Large-E path: active-expert-only dequant + baseline kernels ────────
    # Instead of the fused NF4 Triton kernel (which dequants tile-by-tile for
    # ALL experts), we:
    # 1. Identify active experts (count > 0) — typically ~250 of 260
    # 2. Dequant only active experts to bf16 (one projection at a time)
    # 3. Run the highly optimized baseline fused_lora_grouped_gemm
    # This trades ~2 GB temporary memory per projection for much faster
    # cuBLAS-level grouped GEMM vs the custom NF4 Triton kernel.
    x_in = sorted_x.to(dtype=module._nf4_compute_dtype)
    has_lora = getattr(module, "moe_lora_enabled", False)
    scaling = getattr(module, "moe_lora_scaling", 0.0)
    K = module.d_model
    N_hidden = module.d_hidden

    if not getattr(module, '_nf4_active_logged', False):
        logger.info("[NF4-V5] Using active-expert-only dequant + NF4 frozen LoRA autograd")
        module._nf4_active_logged = True

    try:
        return _run_active_expert_dequant(
            x_in, module, expert_counts, has_lora, scaling,
            K, N_hidden, liger_silu_mul, sorted_x.dtype,
        )
    except Exception as exc:
        if not getattr(module, '_nf4_active_fallback_logged', False):
            logger.warning(
                f"Active-expert dequant failed ({type(exc).__name__}: {exc}), "
                f"falling back to fused NF4 Triton kernel"
            )
            module._nf4_active_fallback_logged = True
        # Fall back to fused NF4 Triton kernel (original large-E path)
        return _run_fused_nf4_triton(
            x_in, module, expert_counts, has_lora, scaling,
            K, N_hidden, liger_silu_mul, sorted_x.dtype,
        )


def _run_active_expert_dequant(x_in, module, expert_counts, has_lora, scaling,
                              K, N_hidden, liger_silu_mul, out_dtype):
    """
    Active-expert-only dequant path (V5: NF4 frozen LoRA autograd).

    Instead of dequanting all E=260 experts, only dequant experts with count > 0.

    V5 improvement over V4: Uses NF4FrozenLoRAGroupedGEMMFn custom autograd that:
    1. Moves dequant INSIDE the autograd function — bf16 weights are never saved
       in the autograd graph (saves ~2 GB per projection × 3 × 20 layers = ~120 GB)
    2. Skips grad_W_base computation entirely — base weights are frozen in QLoRA
       (saves 60 _grouped_gemm_dweight calls per step)
    3. Re-dequants from NF4 in backward for dx = grad_output @ W^T (trades ~2ms
       compute for massive memory savings)

    Falls back to V4 (external dequant + fused_lora_grouped_gemm) if no LoRA
    or if the V5 path fails.

    Sequence: gate GEMM → up GEMM → SiLU(gate, up) → down GEMM
    (dequant happens inside each GEMM's autograd function)
    """
    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.to(device=x_in.device, dtype=torch.int64)
    else:
        counts = torch.tensor(expert_counts, device=x_in.device, dtype=torch.int64)

    E = counts.shape[0]
    active_mask = counts > 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]  # [E_active]
    E_active = active_indices.shape[0]
    active_counts = counts[active_indices]  # [E_active]

    block_size = module._nf4_block_size

    # ── V5 path: NF4 frozen LoRA autograd (dequant inside autograd) ────────
    if has_lora:
        if not getattr(module, '_nf4_v5_logged', False):
            logger.info("[NF4-V5] Using NF4FrozenLoRAGroupedGEMMFn — "
                        "no grad_W_base, no bf16 in autograd graph")
            module._nf4_v5_logged = True

        # Gate projection
        gate_out = nf4_frozen_lora_grouped_gemm(
            x_in,
            module.W_gate_nf4_packed[active_indices],
            module.W_gate_nf4_absmax[active_indices],
            module.lora_A_W_gate[active_indices],
            module.lora_B_W_gate[active_indices],
            active_counts, K, N_hidden, scaling, block_size,
        )

        # Up projection
        up_out = nf4_frozen_lora_grouped_gemm(
            x_in,
            module.W_up_nf4_packed[active_indices],
            module.W_up_nf4_absmax[active_indices],
            module.lora_A_W_up[active_indices],
            module.lora_B_W_up[active_indices],
            active_counts, K, N_hidden, scaling, block_size,
        )

        # SiLU activation
        h = liger_silu_mul(gate_out, up_out)
        del gate_out, up_out

        if module.training and module.dropout > 0:
            h = torch.nn.functional.dropout(h, p=module.dropout)

        # Down projection (note: down is [N_hidden, K], not [K, N_hidden])
        out = nf4_frozen_lora_grouped_gemm(
            h,
            module.W_down_nf4_packed[active_indices],
            module.W_down_nf4_absmax[active_indices],
            module.lora_A_W_down[active_indices],
            module.lora_B_W_down[active_indices],
            active_counts, N_hidden, K, scaling, block_size,
        )
        del h
        return out.to(dtype=out_dtype)

    # ── Fallback: no LoRA — dequant externally + triton grouped GEMM ───────
    compute_dtype = module._nf4_compute_dtype
    try:
        from .kernels.triton_moe_grouped_gemm import triton_grouped_gemm as _triton_gg
    except ImportError:
        _triton_gg = None

    if _triton_gg is None:
        raise RuntimeError("No grouped GEMM kernel available")

    W_gate_active = _dequant_nf4_batched(
        module.W_gate_nf4_packed[active_indices],
        module.W_gate_nf4_absmax[active_indices],
        (E_active, K, N_hidden), block_size, compute_dtype,
    )
    gate_out = _triton_gg(x_in, W_gate_active, active_counts)
    del W_gate_active

    W_up_active = _dequant_nf4_batched(
        module.W_up_nf4_packed[active_indices],
        module.W_up_nf4_absmax[active_indices],
        (E_active, K, N_hidden), block_size, compute_dtype,
    )
    up_out = _triton_gg(x_in, W_up_active, active_counts)
    del W_up_active

    h = liger_silu_mul(gate_out, up_out)
    del gate_out, up_out

    if module.training and module.dropout > 0:
        h = torch.nn.functional.dropout(h, p=module.dropout)

    W_down_active = _dequant_nf4_batched(
        module.W_down_nf4_packed[active_indices],
        module.W_down_nf4_absmax[active_indices],
        (E_active, N_hidden, K), block_size, compute_dtype,
    )
    out = _triton_gg(h, W_down_active, active_counts)
    del W_down_active

    return out.to(dtype=out_dtype)


def _run_fused_nf4_triton(x_in, module, expert_counts, has_lora, scaling,
                           K, N_hidden, liger_silu_mul, out_dtype):
    """
    Original fused NF4 Triton kernel path (fallback for V4).

    Uses the custom Triton kernel that dequants NF4 tile-by-tile in SRAM.
    Slower than V4 active-expert dequant but uses less peak memory.
    """
    if has_lora:
        from .kernels.triton_nf4_grouped_gemm import nf4_lora_grouped_gemm
        gate_out = nf4_lora_grouped_gemm(
            x_in,
            module.W_gate_nf4_packed, module.W_gate_nf4_absmax,
            module.lora_A_W_gate, module.lora_B_W_gate,
            expert_counts, K, N_hidden, scaling,
            module._nf4_block_size,
        )
        up_out = nf4_lora_grouped_gemm(
            x_in,
            module.W_up_nf4_packed, module.W_up_nf4_absmax,
            module.lora_A_W_up, module.lora_B_W_up,
            expert_counts, K, N_hidden, scaling,
            module._nf4_block_size,
        )
        h = liger_silu_mul(gate_out, up_out)
        if module.training and module.dropout > 0:
            h = torch.nn.functional.dropout(h, p=module.dropout)
        out = nf4_lora_grouped_gemm(
            h,
            module.W_down_nf4_packed, module.W_down_nf4_absmax,
            module.lora_A_W_down, module.lora_B_W_down,
            expert_counts, N_hidden, K, scaling,
            module._nf4_block_size,
        )
        return out.to(dtype=out_dtype)

    # No LoRA — dequant all and use baseline
    W_gate = _dequant_nf4_batched(
        module.W_gate_nf4_packed, module.W_gate_nf4_absmax,
        module.W_gate_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )
    W_up = _dequant_nf4_batched(
        module.W_up_nf4_packed, module.W_up_nf4_absmax,
        module.W_up_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )
    W_down = _dequant_nf4_batched(
        module.W_down_nf4_packed, module.W_down_nf4_absmax,
        module.W_down_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )
    try:
        return _run_dequant_grouped(
            x_in, W_gate, W_up, W_down, module, expert_counts,
            False, scaling, liger_silu_mul, out_dtype
        )
    finally:
        del W_gate, W_up, W_down


def _run_baseline_grouped(module, sorted_x, expert_counts, liger_silu_mul):
    """
    Run the EXACT same code path as the original MoEFFN._moe_grouped.

    Uses cached bf16 buffers (module.W_gate/W_up/W_down) with the same
    kernel priority as the baseline model. LoRA handling matches the model's
    original _moe_grouped — for 8B this means NO inline LoRA; for 70B it
    means fused LoRA grouped GEMM.
    """
    x_in = sorted_x.to(dtype=module.W_gate.dtype)
    has_lora = getattr(module, "moe_lora_enabled", False)
    scaling = getattr(module, "moe_lora_scaling", 0.0)

    # Import kernels — same resolution as the model file
    try:
        from .kernels.triton_moe_grouped_gemm import triton_grouped_gemm as _triton_gg
    except ImportError:
        _triton_gg = None
    try:
        from .kernels.triton_moe_fused_gate_up import fused_moe_gate_up_silu as _fused_gus
    except ImportError:
        _fused_gus = None
    try:
        from .kernels.fused_lora_grouped_gemm import (
            fused_lora_gate_up_silu as _fused_lora_gus,
            fused_lora_grouped_gemm as _fused_lora_gg,
        )
    except ImportError:
        _fused_lora_gus = None
        _fused_lora_gg = None

    # Gate + Up: mirror the model's kernel priority chain
    if has_lora and _fused_lora_gus is not None:
        h = _fused_lora_gus(
            x_in, module.W_gate, module.W_up,
            module.lora_A_W_gate, module.lora_B_W_gate,
            module.lora_A_W_up, module.lora_B_W_up,
            expert_counts, scaling,
        )
    elif has_lora and _fused_lora_gg is not None:
        gate_out = _fused_lora_gg(
            x_in, module.W_gate, module.lora_A_W_gate, module.lora_B_W_gate,
            expert_counts, scaling,
        )
        up_out = _fused_lora_gg(
            x_in, module.W_up, module.lora_A_W_up, module.lora_B_W_up,
            expert_counts, scaling,
        )
        h = liger_silu_mul(gate_out, up_out)
    elif _fused_gus is not None and not has_lora:
        h = _fused_gus(x_in, module.W_gate, module.W_up, expert_counts)
    elif _triton_gg is not None:
        gate_out = _triton_gg(x_in, module.W_gate, expert_counts)
        up_out = _triton_gg(x_in, module.W_up, expert_counts)
        h = liger_silu_mul(gate_out, up_out)
    else:
        raise RuntimeError("No grouped GEMM kernel available for NF4 cached path")

    if module.training and module.dropout > 0:
        h = torch.nn.functional.dropout(h, p=module.dropout)

    # Down projection
    if has_lora and _fused_lora_gg is not None:
        out = _fused_lora_gg(
            h, module.W_down, module.lora_A_W_down, module.lora_B_W_down,
            expert_counts, scaling,
        )
    elif _triton_gg is not None:
        out = _triton_gg(h, module.W_down, expert_counts)
    else:
        raise RuntimeError("No grouped GEMM kernel available for NF4 cached path")

    return out.to(dtype=sorted_x.dtype)


def _run_dequant_grouped(x_in, W_gate, W_up, W_down, module, expert_counts,
                         has_lora, scaling, liger_silu_mul, out_dtype):
    """Run grouped GEMM with on-the-fly dequantized weights (large-E fallback)."""
    try:
        from .kernels.fused_lora_grouped_gemm import fused_lora_gate_up_silu as fused_lora_gate_up_silu_fn
        from .kernels.fused_lora_grouped_gemm import fused_lora_grouped_gemm as fused_lora_grouped_gemm_fn
    except ImportError:
        fused_lora_gate_up_silu_fn = None
        fused_lora_grouped_gemm_fn = None

    if has_lora and fused_lora_gate_up_silu_fn is not None:
        h = fused_lora_gate_up_silu_fn(
            x_in, W_gate, W_up,
            module.lora_A_W_gate, module.lora_B_W_gate,
            module.lora_A_W_up, module.lora_B_W_up,
            expert_counts, scaling,
        )
    elif has_lora and fused_lora_grouped_gemm_fn is not None:
        gate_out = fused_lora_grouped_gemm_fn(
            x_in, W_gate, module.lora_A_W_gate, module.lora_B_W_gate,
            expert_counts, scaling,
        )
        up_out = fused_lora_grouped_gemm_fn(
            x_in, W_up, module.lora_A_W_up, module.lora_B_W_up,
            expert_counts, scaling,
        )
        h = liger_silu_mul(gate_out, up_out)
    else:
        from .kernels.triton_moe_grouped_gemm import triton_grouped_gemm
        gate_out = triton_grouped_gemm(x_in, W_gate, expert_counts)
        up_out = triton_grouped_gemm(x_in, W_up, expert_counts)
        h = liger_silu_mul(gate_out, up_out)

    if module.training and module.dropout > 0:
        h = torch.nn.functional.dropout(h, p=module.dropout)

    if has_lora and fused_lora_grouped_gemm_fn is not None:
        out = fused_lora_grouped_gemm_fn(
            h, W_down, module.lora_A_W_down, module.lora_B_W_down,
            expert_counts, scaling,
        )
    else:
        from .kernels.triton_moe_grouped_gemm import triton_grouped_gemm
        out = triton_grouped_gemm(h, W_down, expert_counts)

    return out.to(dtype=out_dtype)



def _moe_vectorized_nf4(module, sorted_x, sorted_expert_indices):
    """
    NF4-aware MoE vectorized fallback path.

    Replaces MoEFFN._moe_vectorized when expert weights are NF4-quantized.
    Uses cached bf16 weights when available (E <= threshold), otherwise
    dequantizes on-the-fly.
    """
    m = sorted_x.size(0)
    if m == 0:
        return torch.empty_like(sorted_x)

    # Get weights: cached bf16 → use original method, otherwise dequantize
    if getattr(module, '_nf4_use_cache', False):
        original_fn = getattr(module, '_nf4_original_moe_vectorized', None)
        if original_fn is not None:
            return original_fn(sorted_x, sorted_expert_indices)

    try:
        from .models.liger_ops import liger_silu_mul
    except ImportError:
        liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

    # No cache — dequantize on-the-fly
    W_gate = _dequant_nf4_batched(
        module.W_gate_nf4_packed, module.W_gate_nf4_absmax,
        module.W_gate_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )
    W_up = _dequant_nf4_batched(
        module.W_up_nf4_packed, module.W_up_nf4_absmax,
        module.W_up_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )
    W_down = _dequant_nf4_batched(
        module.W_down_nf4_packed, module.W_down_nf4_absmax,
        module.W_down_nf4_shape, module._nf4_block_size, module._nf4_compute_dtype
    )

    out = torch.empty((m, module.d_model), device=sorted_x.device, dtype=sorted_x.dtype)
    chunk = getattr(module, 'vectorized_chunk_size', 64)

    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        x_chunk = sorted_x[start:end]
        idx_chunk = sorted_expert_indices[start:end]

        x_expanded = x_chunk.unsqueeze(1)  # [C, 1, D]
        w_gate_sel = W_gate[idx_chunk]  # [C, D, H]
        w_up_sel = W_up[idx_chunk]      # [C, D, H]
        w_down_sel = W_down[idx_chunk]  # [C, H, D]

        gate_out = torch.bmm(x_expanded, w_gate_sel).squeeze(1)
        up_out = torch.bmm(x_expanded, w_up_sel).squeeze(1)
        h = liger_silu_mul(gate_out, up_out)
        if module.training and module.dropout > 0:
            h = torch.nn.functional.dropout(h, p=module.dropout)
        out[start:end] = torch.bmm(h.unsqueeze(1), w_down_sel).squeeze(1)

    del W_gate, W_up, W_down
    return out


def patch_moe_nf4_forward(model: nn.Module) -> int:
    """
    Patch MoEFFN modules to use NF4-aware forward path.

    After quantize_moe_experts() has converted weights to NF4, this function
    monkey-patches _moe_grouped AND _moe_vectorized on each quantized MoEFFN
    to use the NF4 path. Both must be patched because MoEFFN.forward falls
    back to _moe_vectorized if _moe_grouped raises.

    Returns: number of modules patched
    """
    import types
    count = 0
    for mod_name, module in model.named_modules():
        if not getattr(module, '_nf4_quantized', False):
            continue

        # Save original methods BEFORE patching (for cached bf16 fast path)
        module._nf4_original_moe_grouped = module._moe_grouped
        module._nf4_original_moe_vectorized = module._moe_vectorized

        def _make_nf4_grouped(mod):
            def _patched_moe_grouped(self, sorted_x, expert_counts):
                return _moe_grouped_nf4(self, sorted_x, expert_counts)
            return types.MethodType(_patched_moe_grouped, mod)

        def _make_nf4_vectorized(mod):
            def _patched_moe_vectorized(self, sorted_x, sorted_expert_indices):
                return _moe_vectorized_nf4(self, sorted_x, sorted_expert_indices)
            return types.MethodType(_patched_moe_vectorized, mod)

        module._moe_grouped = _make_nf4_grouped(module)
        module._moe_vectorized = _make_nf4_vectorized(module)
        count += 1
        cache_status = "cached bf16" if getattr(module, '_nf4_use_cache', False) else "on-the-fly dequant"
        logger.info(f"  [NF4-V3] forward patched: {mod_name} [{cache_status}]")

    return count


def print_nf4_summary(model: nn.Module):
    """Print NF4 quantization summary."""
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    nf4_bytes = 0
    bf16_equiv_bytes = 0
    bf16_cache_bytes = 0
    nf4_count = 0
    cached_count = 0

    for mod_name, module in model.named_modules():
        if not getattr(module, '_nf4_quantized', False):
            continue
        use_cache = getattr(module, '_nf4_use_cache', False)
        for pname in ['W_gate', 'W_up', 'W_down']:
            packed = getattr(module, f'{pname}_nf4_packed', None)
            absmax = getattr(module, f'{pname}_nf4_absmax', None)
            shape = getattr(module, f'{pname}_nf4_shape', None)
            if packed is not None and shape is not None:
                nf4_bytes += packed.nbytes + (absmax.nbytes if absmax is not None else 0)
                E, K, N = shape
                bf16_equiv_bytes += E * K * N * 2
                nf4_count += 1
                if use_cache:
                    bf16_cache_bytes += E * K * N * 2
                    cached_count += 1

    if nf4_count == 0:
        print("  No NF4-quantized expert weights found.")
        return

    total_stored = nf4_bytes + bf16_cache_bytes
    print("\n" + "=" * 70)
    print("  NF4 QUANTIZATION SUMMARY")
    print("=" * 70)
    print(f"  Expert weight tensors quantized: {nf4_count}")
    print(f"  Cached bf16 tensors (E<={_NF4_CACHE_THRESHOLD}): {cached_count}")
    print(f"  Original bf16 size:  {bf16_equiv_bytes / 1e9:.2f} GB")
    print(f"  NF4 packed size:     {nf4_bytes / 1e9:.2f} GB")
    print(f"  Cached bf16 size:    {bf16_cache_bytes / 1e9:.2f} GB")
    print(f"  Total stored:        {total_stored / 1e9:.2f} GB")
    print(f"  Net savings:         {(bf16_equiv_bytes - total_stored) / 1e9:.2f} GB "
          f"({(1 - total_stored / bf16_equiv_bytes) * 100:.1f}%)")
    print("=" * 70 + "\n")
