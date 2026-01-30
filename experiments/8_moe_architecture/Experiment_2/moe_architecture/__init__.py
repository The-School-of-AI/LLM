"""
MoE Architecture Package
========================

Team 8 - Mixture of Experts Architecture for LLM Development

This package provides a complete, production-ready MoE implementation
supporting the 4-stage growth cadence:

1. Stage 1: 1B Dense (Foundation)
2. Stage 2: 3B MoE-8 (Learn Routing)
3. Stage 3: 8B MoE-8 (Scale Dimensions)
4. Stage 4: 70B MoE-64 (Expert Expansion)

Key Features:
- GSA-style gated lightning router with sigmoid scoring
- Dual gating (G1+G2) for collapse prevention
- Null experts for junk token absorption
- Loss-free load balancing via bias adjustment
- Comprehensive telemetry (Team 7 integration)
- CUDA kernels for high performance

Usage:
    from moe_architecture import create_model, get_config
    
    # Create 3B MoE model
    config = get_config('3b_moe')
    model = create_model(config)
    
    # Forward pass
    outputs = model(input_ids)
    logits = outputs['logits']

Configuration:
    from moe_architecture.config import get_config, print_config_summary
    
    config = get_config('70b_moe')
    print_config_summary(config)

Team Integration:
- Team 6: Tokenizer constraints (junk token IDs, special tokens)
- Team 7: Null-routing telemetry, health gates, plugin interface

References:
- DeepSeek-V3: MoE architecture and loss-free load balancing
- GSA Paper (arXiv:2601.15305v1): Gated attention and adaptive sparsity
"""

__version__ = "1.0.0"
__author__ = "Team 8 - MoE Architecture"

# Import main components
from .config import (
    MoEModelConfig,
    get_config,
    print_config_summary,
    ModelStage,
    RouterType,
    LoadBalanceStrategy,
)

from .model.transformer import (
    MoETransformer,
    create_model,
    load_model,
)

from .model.router import GSARouter
from .model.expert import GatedExpert, NullExpert, SharedExpert, ExpertContainer
from .model.moe_block import MoEBlock
from .model.attention import GQAttention, RMSNorm
from .model.load_balancer import LoadBalancer, LoadBalanceConfig

# Utilities
from .utils.model_utils import (
    expand_dense_to_moe,
    expand_moe_experts,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    print_parameter_summary,
)

from .utils.telemetry import (
    MoETelemetrySystem,
    TelemetryConfig,
    create_default_telemetry,
)

# Kernels
from .kernels.moe_kernels import MoEKernels, ExpertParallelExecutor


__all__ = [
    # Config
    'MoEModelConfig',
    'get_config',
    'print_config_summary',
    'ModelStage',
    'RouterType',
    'LoadBalanceStrategy',
    
    # Model
    'MoETransformer',
    'create_model',
    'load_model',
    
    # Components
    'GSARouter',
    'GatedExpert',
    'NullExpert',
    'SharedExpert',
    'ExpertContainer',
    'MoEBlock',
    'GQAttention',
    'RMSNorm',
    'LoadBalancer',
    'LoadBalanceConfig',
    
    # Utils
    'expand_dense_to_moe',
    'expand_moe_experts',
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    'print_parameter_summary',
    
    # Telemetry
    'MoETelemetrySystem',
    'TelemetryConfig',
    'create_default_telemetry',
    
    # Kernels
    'MoEKernels',
    'ExpertParallelExecutor',
]
