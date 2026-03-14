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


def _dequant_nf4_batched(packed, absmax, shape, block_size, compute_dtype):
    """Batched vectorized NF4 dequantization — all experts at once.

    Args:
        packed: [E, K*N//2] uint8
        absmax: [E, blocks_per_expert] float
        shape: (E, K, N)
        block_size: int
        compute_dtype: torch.dtype

    Returns: [E, K, N] tensor in compute_dtype
    """
    from .kernels.nf4_quantize import NF4_LEVELS

    E, K, N = shape
    numel = K * N
    nf4_levels = NF4_LEVELS.to(packed.device)

    high = ((packed >> 4) & 0x0F).to(torch.int16)
    low = (packed & 0x0F).to(torch.int16)
    indices = torch.stack([high, low], dim=2).reshape(E, -1)
    del high, low
    indices = indices[:, :numel]

    values = nf4_levels[indices.long()].float()
    del indices

    pad = (block_size - numel % block_size) % block_size
    padded_total = numel + pad
    if pad > 0:
        values = torch.nn.functional.pad(values, (0, pad))
    num_blocks = padded_total // block_size
    values = values.view(E, num_blocks, block_size)
    values = values * absmax[:, :num_blocks].unsqueeze(2)
    values = values.reshape(E, -1)[:, :numel]

    return values.view(E, K, N).to(compute_dtype)


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
            return original_fn(sorted_x, expert_counts)
        # Fallback if original wasn't saved (shouldn't happen)
        return _run_baseline_grouped(
            module, sorted_x, expert_counts, liger_silu_mul
        )

    # ── Large-E path: fused NF4 Triton kernel ──────────────────────────────
    x_in = sorted_x.to(dtype=module._nf4_compute_dtype)
    has_lora = getattr(module, "moe_lora_enabled", False)
    scaling = getattr(module, "moe_lora_scaling", 0.0)
    K = module.d_model
    N_hidden = module.d_hidden

    if has_lora and not getattr(module, '_nf4_fused_failed', False):
        try:
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
            return out.to(dtype=sorted_x.dtype)
        except Exception as exc:
            module._nf4_fused_failed = True
            logger.warning(
                f"Fused NF4 GEMM failed ({type(exc).__name__}: {exc}), "
                f"falling back to dequant-then-GEMM"
            )

    # ── Fallback: dequantize then use baseline kernels ──────────────────────
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
            has_lora, scaling, liger_silu_mul, sorted_x.dtype
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
        logger.info(f"  NF4 forward patched: {mod_name} [{cache_status}]")

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
