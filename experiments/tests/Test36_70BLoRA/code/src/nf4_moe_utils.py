"""
NF4 MoE Integration Utilities.

Provides functions to:
1. Quantize existing MoE expert weights (W_gate, W_up, W_down) to NF4
2. Patch MoEFFN modules to use NF4 base weights + bf16 LoRA
3. Wire the fused NF4 grouped GEMM kernel into the MoE forward path

Usage:
    from src.nf4_moe_utils import quantize_moe_experts, patch_moe_nf4_forward

    # After model creation and LoRA injection:
    quantize_moe_experts(model, config)
    patch_moe_nf4_forward(model)

    # MoE forward now uses NF4 dequant-on-the-fly for base weights
    # LoRA adapters remain in bf16

Memory savings for 70B model:
    - 20 layers × 260 experts × 3 weights × [4096, 1024] × 2 bytes = ~128 GB (bf16)
    - With NF4: ~32 GB (4× reduction)
    - Net savings: ~96 GB across all layers
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


def quantize_moe_experts(
    model: nn.Module,
    config: Optional[NF4QuantConfig] = None,
) -> int:
    """
    Quantize MoE expert base weights to NF4 in-place.

    Finds all modules with W_gate, W_up, W_down nn.Parameters (MoEFFN layers)
    and replaces them with NF4 quantized buffers. The original bf16 parameters
    are deleted to free memory.

    LoRA parameters (lora_A_W_gate, etc.) are NOT touched — they stay in bf16.

    Args:
        model: model with MoEFFN modules
        config: NF4QuantConfig (defaults to block_size=64, double_quant=True)

    Returns:
        Number of expert weight tensors quantized
    """
    if config is None:
        config = NF4QuantConfig()

    count = 0
    for mod_name, module in model.named_modules():
        expert_params = ['W_gate', 'W_up', 'W_down']
        has_experts = all(hasattr(module, p) for p in expert_params)
        if not has_experts:
            continue

        # Check these are 3D expert weight tensors [E, K, N]
        W_gate = getattr(module, 'W_gate')
        if not isinstance(W_gate, nn.Parameter):
            continue

        # Get original shapes (handle ZeRO-3 partitioned params)
        shapes = {}
        for pname in expert_params:
            param = getattr(module, pname)
            shape = getattr(param, 'ds_shape', param.shape)
            if len(shape) != 3:
                break
            shapes[pname] = shape
        else:
            # All three are 3D — proceed with quantization
            for pname in expert_params:
                param = getattr(module, pname)
                original_shape = shapes[pname]

                # For ZeRO-3: need to gather the full parameter first
                if hasattr(param, 'ds_id'):
                    # ZeRO-3 partitioned — gather full param
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([param]):
                        full_data = param.data.clone()
                else:
                    full_data = param.data

                # Quantize to NF4
                nf4_param = quantize_tensor_nf4(full_data, config)

                # Store NF4 data as buffers on the module
                module.register_buffer(
                    f'{pname}_nf4_packed', nf4_param.packed
                )
                module.register_buffer(
                    f'{pname}_nf4_absmax', nf4_param.absmax
                )
                # Store metadata
                setattr(module, f'{pname}_nf4_shape', nf4_param.original_shape)
                setattr(module, f'{pname}_nf4_numel', nf4_param.original_numel)
                setattr(module, f'{pname}_nf4_block_size', nf4_param.block_size)

                # Delete original bf16 parameter to free memory
                delattr(module, pname)

                count += 1
                E, K, N = original_shape
                bf16_mb = E * K * N * 2 / 1e6
                nf4_mb = nf4_param.nbytes() / 1e6
                logger.info(
                    f"  NF4: {mod_name}.{pname} [{E}×{K}×{N}] "
                    f"bf16={bf16_mb:.1f}MB → nf4={nf4_mb:.1f}MB "
                    f"({(1 - nf4_mb/bf16_mb)*100:.0f}% savings)"
                )

            # Mark module as NF4-quantized
            module._nf4_quantized = True
            module._nf4_block_size = config.block_size
            module._nf4_compute_dtype = config.compute_dtype

    return count


def _get_nf4_weight(module, param_name):
    """Dequantize an NF4 expert weight back to compute dtype."""
    from .kernels.nf4_quantize import NF4_LEVELS, _dequantize_block_nf4

    packed = getattr(module, f'{param_name}_nf4_packed')
    absmax = getattr(module, f'{param_name}_nf4_absmax')
    shape = getattr(module, f'{param_name}_nf4_shape')
    numel = getattr(module, f'{param_name}_nf4_numel')
    block_size = getattr(module, f'{param_name}_nf4_block_size')
    compute_dtype = module._nf4_compute_dtype

    nf4_levels = NF4_LEVELS.to(packed.device)
    flat = _dequantize_block_nf4(
        packed, absmax, numel, block_size, nf4_levels, compute_dtype
    )
    return flat.view(shape)


def _moe_grouped_nf4(module, sorted_x, expert_counts):
    """
    NF4-aware MoE grouped forward path.

    Replaces MoEFFN._moe_grouped when expert weights are NF4-quantized.
    Uses fused NF4 dequant+GEMM kernel when available, falls back to
    dequant-then-GEMM otherwise.
    """
    x_in = sorted_x.to(dtype=module._nf4_compute_dtype)
    has_lora = getattr(module, "moe_lora_enabled", False)
    scaling = getattr(module, "moe_lora_scaling", 0.0)

    # Try fused NF4+LoRA grouped GEMM kernel
    try:
        from .kernels.triton_nf4_grouped_gemm import nf4_lora_grouped_gemm
        has_fused_nf4 = True
    except ImportError:
        has_fused_nf4 = False

    K = module.d_model
    N_hidden = module.d_hidden

    try:
        from .models.liger_ops import liger_silu_mul
    except ImportError:
        liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

    if has_fused_nf4 and has_lora:
        try:
            # Gate + Up with fused NF4 dequant + LoRA
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

            # Down projection with fused NF4 + LoRA
            out = nf4_lora_grouped_gemm(
                h,
                module.W_down_nf4_packed, module.W_down_nf4_absmax,
                module.lora_A_W_down, module.lora_B_W_down,
                expert_counts, N_hidden, K, scaling,
                module._nf4_block_size,
            )
            return out.to(dtype=sorted_x.dtype)
        except Exception as fused_exc:
            logger.warning(
                f"Fused NF4 GEMM failed ({type(fused_exc).__name__}: {fused_exc}), "
                f"falling back to dequant-then-GEMM"
            )

    # Fallback: dequantize then use existing grouped GEMM path
    W_gate = _get_nf4_weight(module, 'W_gate')
    W_up = _get_nf4_weight(module, 'W_up')
    W_down = _get_nf4_weight(module, 'W_down')

    try:
        # Use the existing fused LoRA grouped GEMM with dequantized weights
        from .kernels.fused_lora_grouped_gemm import fused_lora_gate_up_silu as fused_lora_gate_up_silu_fn
        from .kernels.fused_lora_grouped_gemm import fused_lora_grouped_gemm as fused_lora_grouped_gemm_fn

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

        return out.to(dtype=sorted_x.dtype)
    finally:
        # Dequantized weights are local tensors, no cleanup needed
        del W_gate, W_up, W_down


def _moe_vectorized_nf4(module, sorted_x, sorted_expert_indices):
    """
    NF4-aware MoE vectorized fallback path.

    Replaces MoEFFN._moe_vectorized when expert weights are NF4-quantized.
    Dequantizes full expert weights then indexes per-token.
    """
    m = sorted_x.size(0)
    if m == 0:
        return torch.empty_like(sorted_x)

    try:
        from .models.liger_ops import liger_silu_mul
    except ImportError:
        liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

    # Dequantize all expert weights
    W_gate = _get_nf4_weight(module, 'W_gate')
    W_up = _get_nf4_weight(module, 'W_up')
    W_down = _get_nf4_weight(module, 'W_down')

    out = torch.empty((m, module.d_model), device=sorted_x.device, dtype=sorted_x.dtype)
    chunk = getattr(module, 'vectorized_chunk_size', 64)

    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        x_chunk = sorted_x[start:end]
        idx_chunk = sorted_expert_indices[start:end]

        x_expanded = x_chunk.unsqueeze(1)  # [C, 1, D]
        w_gate_sel = W_gate[idx_chunk]  # [C, D, H]
        w_up_sel = W_up[idx_chunk]  # [C, D, H]
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
    to use the NF4 dequant-on-the-fly path. Both must be patched because
    MoEFFN.forward falls back to _moe_vectorized if _moe_grouped raises.

    Args:
        model: model with NF4-quantized MoEFFN modules

    Returns:
        Number of modules patched
    """
    import types
    count = 0
    for mod_name, module in model.named_modules():
        if not getattr(module, '_nf4_quantized', False):
            continue

        # Monkey-patch _moe_grouped to use NF4 path
        def _make_nf4_grouped(mod):
            def _patched_moe_grouped(self, sorted_x, expert_counts):
                return _moe_grouped_nf4(self, sorted_x, expert_counts)
            return types.MethodType(_patched_moe_grouped, mod)

        # Monkey-patch _moe_vectorized to use NF4 path (fallback safety)
        def _make_nf4_vectorized(mod):
            def _patched_moe_vectorized(self, sorted_x, sorted_expert_indices):
                return _moe_vectorized_nf4(self, sorted_x, sorted_expert_indices)
            return types.MethodType(_patched_moe_vectorized, mod)

        module._moe_grouped = _make_nf4_grouped(module)
        module._moe_vectorized = _make_nf4_vectorized(module)
        count += 1
        logger.info(f"  NF4 forward patched: {mod_name}")

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
    nf4_count = 0

    for mod_name, module in model.named_modules():
        if not getattr(module, '_nf4_quantized', False):
            continue
        for pname in ['W_gate', 'W_up', 'W_down']:
            packed = getattr(module, f'{pname}_nf4_packed', None)
            absmax = getattr(module, f'{pname}_nf4_absmax', None)
            shape = getattr(module, f'{pname}_nf4_shape', None)
            if packed is not None and shape is not None:
                nf4_bytes += packed.nbytes + (absmax.nbytes if absmax is not None else 0)
                E, K, N = shape
                bf16_equiv_bytes += E * K * N * 2
                nf4_count += 1

    if nf4_count == 0:
        print("  No NF4-quantized expert weights found.")
        return

    print("\n" + "=" * 70)
    print("  NF4 QUANTIZATION SUMMARY")
    print("=" * 70)
    print(f"  Expert weight tensors quantized: {nf4_count}")
    print(f"  Original bf16 size:  {bf16_equiv_bytes / 1e9:.2f} GB")
    print(f"  NF4 quantized size:  {nf4_bytes / 1e9:.2f} GB")
    print(f"  Memory savings:      {(bf16_equiv_bytes - nf4_bytes) / 1e9:.2f} GB "
          f"({(1 - nf4_bytes / bf16_equiv_bytes) * 100:.1f}%)")
    print("=" * 70 + "\n")
