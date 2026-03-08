"""
Reusable LoRA (Low-Rank Adaptation) utilities for DeepSpeed ZeRO-3 training.

Model-agnostic: works with any nn.Module by targeting named Linear modules.
ZeRO-3 compatible: LoRA params created AFTER zero.Init() so they are NOT
sharded during construction. Base model params keep their ds_id partition.

Usage:
    from src.lora_utils import LoRAConfig, inject_lora, freeze_non_lora, get_lora_params

    lora_cfg = LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "k_proj", ...])
    inject_lora(model, lora_cfg)
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
    fused_kernels: bool = False  # Unsloth-style: one autograd node, recompute intermediate in backward
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "W_q", "W_k", "W_v",
    ])


# ---------------------------------------------------------------------------
# Fused LoRA (Unsloth-style): one autograd node, recompute A(x) in backward
# ---------------------------------------------------------------------------

class _FusedLoRAFullFunc(torch.autograd.Function):
    """
    Fully fused base + LoRA in one node. Saves x, W, A, B; recomputes A(x) in backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, W, A, B, scaling):
        base_out = F.linear(x, W)
        inter = F.linear(x, A)
        lora_out = F.linear(inter, B) * scaling
        ctx.save_for_backward(x, W, A, B)
        ctx.scaling = scaling
        return base_out + lora_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, W, A, B = ctx.saved_tensors
        scaling = ctx.scaling
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        inter_2d = x_2d @ A.t()
        grad_B = scaling * (go_2d.t() @ inter_2d)
        grad_inter = scaling * (go_2d @ B)
        grad_A = grad_inter.t() @ x_2d
        grad_x_2d = go_2d @ W
        grad_x_2d.addmm_(grad_inter, A)
        grad_x = grad_x_2d.reshape(orig_shape)
        return grad_x, None, grad_A, grad_B, None


class _FusedLoRADeltaFunc(torch.autograd.Function):
    """LoRA delta only (for fused path when dropout > 0). Recomputes A(x) in backward."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, A, B, scaling):
        inter = F.linear(x, A)
        delta = F.linear(inter, B) * scaling
        ctx.save_for_backward(x, A, B)
        ctx.scaling = scaling
        return delta

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, A, B = ctx.saved_tensors
        scaling = ctx.scaling
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        inter_2d = x_2d @ A.t()
        grad_B = scaling * (go_2d.t() @ inter_2d)
        grad_inter = scaling * (go_2d @ B)
        grad_A = grad_inter.t() @ x_2d
        grad_x = (grad_inter @ A).reshape(orig_shape)
        return grad_x, grad_A, grad_B, None


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer (unfused).

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


class FusedLoRALinear(nn.Module):
    """
    LoRA-augmented linear with fused forward/backward (Unsloth-style).

    Same interface as LoRALinear; when dropout=0 uses a single autograd node
    (base + LoRA) and recomputes A(x) in backward for lower peak memory.
    When dropout > 0 uses base + fused delta (two nodes).
    """

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.linear = original_linear
        for p in self.linear.parameters():
            p.requires_grad = False

        dtype = getattr(original_linear.weight, 'dtype', torch.bfloat16)
        if dtype is None:
            dtype = torch.bfloat16

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, rank, dtype=dtype))
        self._dropout = dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._dropout > 0:
            base_out = self.linear(x)
            lora_x = self.lora_dropout(x).to(self.lora_A.dtype)
            delta = _FusedLoRADeltaFunc.apply(lora_x, self.lora_A, self.lora_B, self.scaling)
            return base_out + delta
        return _FusedLoRAFullFunc.apply(
            x, self.linear.weight, self.lora_A, self.lora_B, self.scaling
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f} (fused)"
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

    Returns number of LoRA adapters injected.
    """
    # Collect targets first to avoid modifying dict during iteration.
    targets = []
    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = full_name.split(".")[-1]
        if leaf_name not in config.target_modules:
            continue
        targets.append((full_name, module))

    lora_cls = FusedLoRALinear if config.fused_kernels else LoRALinear
    for full_name, original_linear in targets:
        lora_module = lora_cls(
            original_linear=original_linear,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        _set_submodule(model, full_name, lora_module)
        logger.info(
            f"  LoRA: {full_name} ({original_linear.in_features}x{original_linear.out_features}) "
            f"rank={config.rank} {'(fused)' if config.fused_kernels else ''}"
        )

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

    lora_count = sum(
        1 for _, m in model.named_modules()
        if isinstance(m, (LoRALinear, FusedLoRALinear))
    )

    print("\n" + "=" * 70)
    print("  LoRA INJECTION SUMMARY")
    print("=" * 70)
    print(f"  Total parameters:     {total:>15,} ({total / 1e9:.3f}B)")
    print(f"  Trainable (LoRA):     {trainable:>15,} ({trainable / 1e6:.2f}M)")
    print(f"  Frozen (base model):  {frozen:>15,} ({frozen / 1e9:.3f}B)")
    print(f"  Trainable fraction:   {100 * trainable / max(total, 1):.4f}%")
    print(f"  LoRA modules:         {lora_count}")
    optim_mb = trainable * 12 / 1e6  # Adam: param(4) + momentum(4) + variance(4)
    print(f"  LoRA optimizer est:   {optim_mb:.1f} MB total")
    print("=" * 70 + "\n")
