"""
Configuration Presets Package
=============================

Pre-defined configurations for all 4 stages of the growth cadence:

- config_1b_dense.py: Stage 1 - 1B Dense Foundation
- config_3b_moe.py: Stage 2 - 3B MoE-8 (Learn Routing)
- config_8b_moe.py: Stage 3 - 8B MoE-8 (Scale Dimensions)
- config_70b_moe.py: Stage 4 - 70B MoE-64 (Expert Expansion)

Usage:
    from moe_architecture.configs import config_3b_moe
    config = config_3b_moe.get_config()
"""

from . import config_1b_dense
from . import config_3b_moe
from . import config_8b_moe
from . import config_70b_moe

# Quick access to configs
CONFIGS = {
    '1b_dense': config_1b_dense.get_config,
    '3b_moe': config_3b_moe.get_config,
    '8b_moe': config_8b_moe.get_config,
    '70b_moe': config_70b_moe.get_config,
}


def get_config(name: str):
    """Get configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


__all__ = [
    'config_1b_dense',
    'config_3b_moe', 
    'config_8b_moe',
    'config_70b_moe',
    'CONFIGS',
    'get_config',
]
