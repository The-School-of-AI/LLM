"""
MoE (Mixture of Experts) utilities for DeepSpeed training.

This module provides helper functions to properly initialize MoE models
with DeepSpeed, ensuring correct parameter grouping for the optimizer.
"""

from typing import Any


def is_moe_model(model: Any) -> bool:
    """
    Check if a model contains MoE layers.
    
    This function recursively checks if any module in the model has the
    `_z3_leaf` attribute set to True (indicating it's an MoE leaf module)
    or if it's a DeepSpeed MoE layer.
    
    Args:
        model: The PyTorch model to check.
        
    Returns:
        True if the model contains MoE layers, False otherwise.
    """
    for module in model.modules():
        # Check for DeepSpeed MoE layer marker
        if hasattr(module, '_z3_leaf') and module._z3_leaf:
            return True
        # Check for DeepSpeed native MoE layer
        module_name = type(module).__name__
        if 'MoE' in module_name or 'MixtureOfExperts' in module_name:
            return True
    return False


def create_moe_param_groups(model: Any) -> list:
    """
    Create parameter groups for MoE models.
    
    For MoE models, DeepSpeed requires separating expert parameters from
    non-expert parameters. This function uses DeepSpeed's utility to
    split parameters into the appropriate groups.
    
    Args:
        model: The MoE model.
        
    Returns:
        A list of parameter groups suitable for DeepSpeed optimizer.
        
    Example:
        >>> model = MyMoEModel()
        >>> if is_moe_model(model):
        ...     param_groups = create_moe_param_groups(model)
        ... else:
        ...     param_groups = model.parameters()
        >>> model_engine, optimizer, _, _ = deepspeed.initialize(
        ...     model=model,
        ...     model_parameters=param_groups,
        ...     config=ds_config
        ... )
    """
    try:
        from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
        
        parameters = {
            'params': [p for p in model.parameters()],
            'name': 'parameters'
        }
        return split_params_into_different_moe_groups_for_optimizer(parameters)
    except ImportError:
        # Fallback if DeepSpeed MoE utils not available
        print("Warning: DeepSpeed MoE utils not available. Using default parameter groups.")
        return model.parameters()


def get_moe_config_recommendations() -> dict:
    """
    Get recommended DeepSpeed configuration for MoE models.
    
    Returns:
        A dictionary with recommended settings and explanations.
    """
    return {
        "recommended_zero_stage": 2,
        "reason": "ZeRO-2 is more stable with MoE. ZeRO-3 has a known race condition (GitHub #7824).",
        "memory_optimization": {
            "fp16_master_weights_and_grads": True,
            "reason": "Keeps optimizer master weights in FP16, saving ~50% memory."
        },
        "known_issues": [
            {
                "issue": "ZeRO-3 + MoE race condition",
                "github_issue": "https://github.com/deepspeedai/DeepSpeed/issues/7824",
                "fix_pr": "https://github.com/deepspeedai/DeepSpeed/pull/7825",
                "status": "Fixed in DeepSpeed >= 0.18.6"
            }
        ]
    }
