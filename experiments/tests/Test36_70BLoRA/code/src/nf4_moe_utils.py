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

    E_gate = module.W_gate_nf4_packed.shape[0] if hasattr(module, 'W_gate_nf4_packed') else 0
    K = module.d_model
    N_hidden = module.d_hidden

    if has_fused_nf4 and has_lora:
        from .kernels.triton_nf4_grouped_gemm import nf4_lora_grouped_gemm

        try:
            from .models.liger_ops import liger_silu_mul
        except ImportError:
            liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

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

    # Fallback: dequantize then use existing grouped GEMM path
    W_gate = _get_nf4_weight(module, 'W_gate')
    W_up = _get_nf4_weight(module, 'W_up')
    W_down = _get_nf4_weight(module, 'W_down')

    # Temporarily set the dequantized weights on the module
    # so the existing _moe_grouped code path works
    module.W_gate = nn.Parameter(W_gate, requires_grad=False)
    module.W_up = nn.Parameter(W_up, requires_grad=False)
    module.W_down = nn.Parameter(W_down, requires_grad=False)

    try:
        # Use the original _moe_grouped with dequantized weights
        from .kernels.fused_lora_grouped_gemm import fused_lora_gate_up_silu as fused_lora_gate_up_silu_fn
        from .kernels.fused_lora_grouped_gemm import fused_lora_grouped_gemm as fused_lora_grouped_gemm_fn

        try:
            from .models.liger_ops import liger_silu_mul
        except ImportError:
            liger_silu_mul = lambda g, u: torch.nn.functional.silu(g) * u

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
        # Clean up temporary dequantized weights to free memory
        if hasattr(module, 'W_gate') and isinstance(module.W_gate, nn.Parameter):
            del module.W_gate, module.W_up, module.W_down
            torch.cuda.empty_cache()


def patch_moe_nf4_forward(model: nn.Module) -> int:
    """
    Patch MoEFFN modules to use NF4-aware forward path.

    After quantize_moe_experts() has converted weights to NF4, this function
    monkey-patches _moe_grouped on each quantized MoEFFN to use the NF4
    dequant-on-the-fly path.

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

        module._moe_grouped = _make_nf4_grouped(module)
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
