# ============================================================================
# MoE Configuration Templates
# ============================================================================
# This package provides canonical configurations for all MoE model stages.
# ============================================================================

from .config_3b_moe import MoE3BConfig, MoE3BFallbackConfig, get_3b_config
from .config_70b_moe import (
    MoE70BConfig, 
    MoE70BFallbackConfig, 
    MoE70BHighSparsityConfig,
    get_70b_config
)

__all__ = [
    # 3B Configurations
    'MoE3BConfig',
    'MoE3BFallbackConfig',
    'get_3b_config',
    
    # 70B Configurations
    'MoE70BConfig',
    'MoE70BFallbackConfig',
    'MoE70BHighSparsityConfig',
    'get_70b_config',
]


def get_config(model_size: str, variant: str = "primary"):
    """
    Get configuration by model size.
    
    Args:
        model_size: "3b" or "70b"
        variant: "primary", "fallback", or "high_sparsity" (70b only)
    
    Returns:
        Configuration dataclass
    """
    if model_size.lower() == "3b":
        return get_3b_config(fallback=(variant == "fallback"))
    elif model_size.lower() == "70b":
        return get_70b_config(variant=variant)
    else:
        raise ValueError(f"Unknown model size: {model_size}")
