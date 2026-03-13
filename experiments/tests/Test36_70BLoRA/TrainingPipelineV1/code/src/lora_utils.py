"""
Reusable LoRA (Low-Rank Adaptation) utilities for DeepSpeed ZeRO-3 training.

Model-agnostic: works with any nn.Module by targeting named Linear modules.
ZeRO-3 compatible: LoRA params created AFTER zero.Init() so they are NOT
sharded during construction. Base model params keep their ds_id partition.

Supports two LoRA targets:
1. Attention nn.Linear modules — inject_lora() wraps with LoRALinear/FusedLoRALinear
2. MoE expert nn.Parameter weights — inject_moe_lora() adds stacked LoRA params
   [E, rank, K] / [E, N, rank] and wires fused_lora_grouped_gemm into _moe_grouped()

Usage:
    from src.lora_utils import LoRAConfig, inject_lora, inject_moe_lora, freeze_non_lora

    lora_cfg = LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "k_proj", ...])
    inject_lora(model, lora_cfg)
    inject_moe_lora(model, lora_cfg)
    freeze_non_lora(model)
    trainable = get_lora_params(model)
    # Pass trainable to deepspeed.initialize(model_parameters=trainable)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA injection."""
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "W_q", "W_k", "W_v",
    ])
    # Use fused autograd (FusedLoRALinear) for attention LoRA — saves activation memory
    use_fused: bool = True
    # Use manual backward (ManualLoRALinear) — bypasses autograd overhead entirely
    use_manual_backward: bool = False
    # MoE expert LoRA settings
    moe_rank: Optional[int] = None  # defaults to rank if None
    moe_alpha: Optional[float] = None  # defaults to alpha if None
    moe_target_params: List[str] = field(default_factory=lambda: [
        "W_gate", "W_up", "W_down",
    ])


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer.

    Wraps an existing nn.Linear (frozen) and adds trainable low-rank matrices:
        output = frozen_linear(x) + (x @ A^T @ B^T) * (alpha / rank)

    B is initialized to zeros so LoRA starts as identity (no change to base model).
    """

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep the original linear as a submodule — preserves ZeRO-3 ds_id.
        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA matrices: A (rank x in_features), B (out_features x rank)
        dtype = getattr(original_linear.weight, 'dtype', torch.bfloat16)
        # If weight is ZeRO-partitioned, dtype detection may fail; default bf16
        if dtype is None:
            dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize: A with Kaiming, B with zeros (LoRA starts at zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        """Expose base linear weight for code that accesses .weight (e.g. dtype checks)."""
        return self.linear.weight

    @property
    def bias(self):
        """Expose base linear bias for code that accesses .bias."""
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base path (ZeRO-3 gathers weights automatically)
        result = self.linear(x)

        # LoRA path: x @ A^T @ B^T * scaling
        lora_x = self.lora_dropout(x)
        lora_x = lora_x.to(self.lora_A.dtype)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)
        result = result + lora_out * self.scaling

        return result

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"
        )


def _set_submodule(model: nn.Module, target_key: str, new_module: nn.Module):
    """Set a submodule by dot-separated key."""
    parts = target_key.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def inject_lora(model: nn.Module, config: LoRAConfig) -> int:
    """
    Inject LoRA adapters into target modules of the model (in-place).

    Scans model.named_modules() for nn.Linear layers whose leaf name
    matches one of config.target_modules.

    If config.use_fused=True, uses FusedLoRALinear (memory-efficient backward)
    instead of LoRALinear.

    Returns number of LoRA adapters injected.
    """
    # Select LoRA class
    lora_cls = LoRALinear
    if config.use_manual_backward:
        try:
            from .kernels.manual_lora_backward import ManualLoRALinear
            lora_cls = ManualLoRALinear
        except ImportError:
            logger.warning("ManualLoRALinear not available, falling back to FusedLoRALinear")
            config.use_manual_backward = False

    if config.use_fused and not config.use_manual_backward:
        try:
            from .kernels.fused_lora_linear import FusedLoRALinear
            lora_cls = FusedLoRALinear
        except ImportError:
            logger.warning("FusedLoRALinear not available, falling back to LoRALinear")

    # Collect targets first to avoid modifying dict during iteration.
    targets = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in config.target_modules:
            continue
        targets.append((full_name, module))

    for full_name, original_linear in targets:
        lora_module = lora_cls(
            original_linear=original_linear,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        _set_submodule(model, full_name, lora_module)
        fused_tag = " [fused]" if lora_cls is not LoRALinear else ""
        logger.info(f"  LoRA{fused_tag}: {full_name} ({original_linear.in_features}x{original_linear.out_features}) rank={config.rank}")

    return len(targets)


def freeze_non_lora(model: nn.Module) -> tuple:
    """
    Freeze all parameters except LoRA A/B matrices.

    Returns (total_params, trainable_params, frozen_params).
    """
    total = 0
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    return total, trainable, frozen


def get_lora_params(model: nn.Module) -> list:
    """Return list of trainable LoRA parameters for deepspeed.initialize()."""
    return [p for n, p in model.named_parameters() if p.requires_grad]


# ============================================================================
# MoE Expert LoRA Injection
# ============================================================================

def inject_moe_lora(model: nn.Module, config: LoRAConfig) -> int:
    """
    Inject LoRA adapters into MoE expert nn.Parameter weights (in-place).

    MoE expert weights are stacked [E, K, N] nn.Parameters (W_gate, W_up, W_down),
    NOT nn.Linear modules. This function:
    1. Finds modules containing target params (e.g. MoEFFN with W_gate, W_up, W_down)
    2. Creates stacked LoRA params: lora_A [E, rank, in_f], lora_B [E, out_f, rank]
    3. Registers them on the module as e.g. lora_A_W_gate, lora_B_W_gate
    4. Sets moe_lora_scaling on the module for the forward path

    Returns number of MoE LoRA adapters injected.
    """
    moe_rank = config.moe_rank if config.moe_rank is not None else config.rank
    moe_alpha = config.moe_alpha if config.moe_alpha is not None else config.alpha
    scaling = moe_alpha / moe_rank

    count = 0
    for mod_name, module in model.named_modules():
        for param_name in config.moe_target_params:
            if not hasattr(module, param_name):
                continue
            base_param = getattr(module, param_name)
            if not isinstance(base_param, nn.Parameter):
                continue

            # ZeRO-3 partitions params to 1D — use ds_shape for original dimensions
            original_shape = getattr(base_param, "ds_shape", base_param.shape)
            if len(original_shape) != 3:
                continue

            # base_param shape: [E, in_f, out_f] for gate/up, [E, out_f, in_f] for down
            E, dim1, dim2 = original_shape
            in_f = dim1   # tokens come in on dim1
            out_f = dim2  # output on dim2

            dtype = base_param.dtype
            if dtype is None:
                dtype = torch.bfloat16

            # lora_A: [E, rank, in_f] — down-projection per expert
            lora_A = nn.Parameter(torch.empty(E, moe_rank, in_f, dtype=dtype))
            nn.init.kaiming_uniform_(lora_A.view(E * moe_rank, in_f), a=math.sqrt(5))

            # lora_B: [E, out_f, rank] — up-projection per expert (init zeros)
            lora_B = nn.Parameter(torch.zeros(E, out_f, moe_rank, dtype=dtype))

            # Register as named params on the module
            module.register_parameter(f"lora_A_{param_name}", lora_A)
            module.register_parameter(f"lora_B_{param_name}", lora_B)

            # Freeze base weight
            base_param.requires_grad = False

            count += 1
            logger.info(
                f"  MoE LoRA: {mod_name}.{param_name} "
                f"[{E}x{in_f}x{out_f}] rank={moe_rank}"
            )

    # Set scaling on all modules that got LoRA params
    for mod_name, module in model.named_modules():
        has_moe_lora = any(
            hasattr(module, f"lora_A_{p}") for p in config.moe_target_params
        )
        if has_moe_lora:
            module.moe_lora_scaling = scaling
            module.moe_lora_enabled = True

    return count


def print_lora_summary(model: nn.Module):
    """Print LoRA injection summary (rank 0 only)."""
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    lora_count = sum(1 for _, m in model.named_modules() if isinstance(m, LoRALinear))
    moe_lora_count = sum(
        1 for n, _ in model.named_parameters() if "lora_A_W_" in n
    )

    print("\n" + "=" * 70)
    print("  LoRA INJECTION SUMMARY")
    print("=" * 70)
    print(f"  Total parameters:     {total:>15,} ({total / 1e9:.3f}B)")
    print(f"  Trainable (LoRA):     {trainable:>15,} ({trainable / 1e6:.2f}M)")
    print(f"  Frozen (base model):  {frozen:>15,} ({frozen / 1e9:.3f}B)")
    print(f"  Trainable fraction:   {100 * trainable / max(total, 1):.4f}%")
    print(f"  Attention LoRA:       {lora_count}")
    print(f"  MoE expert LoRA:      {moe_lora_count}")
    optim_mb = trainable * 12 / 1e6  # Adam: param(4) + momentum(4) + variance(4)
    print(f"  LoRA optimizer est:   {optim_mb:.1f} MB total")
    print("=" * 70 + "\n")
