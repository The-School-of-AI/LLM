"""
MoE utilities that are safe across DeepSpeed versions and won't break imports.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import torch


def is_moe_model(model: Any) -> bool:
    for m in model.modules():
        name = type(m).__name__
        if "MoE" in name or "MixtureOfExperts" in name:
            return True
        if hasattr(m, "_z3_leaf") and getattr(m, "_z3_leaf") is True:
            return True
        # DeepSpeed MoE layer commonly has `deepspeed_moe`
        if hasattr(m, "deepspeed_moe"):
            return True
    return False


def create_moe_param_groups(model: Any):
    """
    DeepSpeed MoE wants expert params separated for optimizer grouping.
    If DS MoE utils are unavailable, fallback to dense params.
    """
    try:
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

        params = {"params": list(model.parameters()), "name": "parameters"}
        return split_params_into_different_moe_groups_for_optimizer(params)
    except Exception as e:
        print(f"[moe_utils] MoE param group fallback -> using model.parameters() | reason: {e}")
        return model.parameters()


def find_moe_modules(model: Any) -> List[Any]:
    """Return a list of modules that look like DeepSpeed MoE layers."""
    moe_modules = []
    for m in model.modules():
        name = type(m).__name__
        if "MoE" in name or hasattr(m, "deepspeed_moe"):
            moe_modules.append(m)
    return moe_modules


def try_get_moe_expert_counts(model: Any) -> Optional[torch.Tensor]:
    """
    Attempts to extract expert token counts from the first MoE layer.
    Returns a 1D tensor [num_experts] if available.
    """
    moe_modules = find_moe_modules(model)
    if not moe_modules:
        return None

    m = moe_modules[0]

    # DeepSpeed MoE sharded_moe sets `exp_counts` on the MoE layer (seen in your traceback)
    exp_counts = getattr(m, "exp_counts", None)
    if exp_counts is None and hasattr(m, "deepspeed_moe"):
        exp_counts = getattr(m.deepspeed_moe, "exp_counts", None)

    if exp_counts is None:
        return None

    try:
        t = torch.as_tensor(exp_counts).detach()
        if t.ndim == 0:
            return None
        return t.cpu()
    except Exception:
        return None


def expert_imbalance_ratio(expert_counts: Optional[torch.Tensor]) -> Optional[float]:
    """
    Simple imbalance metric: max(count)/mean(count).
    1.0 is perfectly balanced. Higher = more skew.
    """
    if expert_counts is None:
        return None
    counts = expert_counts.float()
    mean = counts.mean().item()
    if mean <= 0:
        return None
    return (counts.max().item() / mean)