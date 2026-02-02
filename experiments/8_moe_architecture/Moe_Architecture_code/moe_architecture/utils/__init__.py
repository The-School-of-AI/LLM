"""
Utilities Package
=================

Utility functions for MoE models:

- model_utils.py: Expansion, checkpointing, parameter analysis
- telemetry.py: Team 7 telemetry interface
"""

from .model_utils import (
    expand_dense_to_moe,
    expand_moe_experts,
    scale_model_dimensions,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    analyze_parameters,
    print_parameter_summary,
    validate_config,
    validate_expansion_compatibility,
    verify_lossless_init,
    build_optimizer,
)

from .telemetry import (
    MoETelemetrySystem,
    TelemetryConfig,
    RoutingEvent,
    create_default_telemetry,
)
from .dashboard_logger import DashboardLogger

__all__ = [
    # Model utils
    'expand_dense_to_moe',
    'expand_moe_experts',
    'scale_model_dimensions',
    'save_checkpoint',
    'load_checkpoint',
    'count_parameters',
    'analyze_parameters',
    'print_parameter_summary',
    'validate_config',
    'validate_expansion_compatibility',
    'verify_lossless_init',
    'build_optimizer',
    
    # Telemetry
    'MoETelemetrySystem',
    'TelemetryConfig',
    'RoutingEvent',
    'create_default_telemetry',
    'DashboardLogger',
]
