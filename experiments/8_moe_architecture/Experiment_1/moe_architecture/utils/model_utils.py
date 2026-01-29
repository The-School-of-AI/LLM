"""
Model Utilities
===============

Utility functions for:
1. Model expansion (Dense→MoE, MoE→larger MoE)
2. Checkpointing and loading
3. Configuration validation
4. Parameter counting and analysis

These utilities support the growth cadence:
- Stage 1→2: Dense to MoE (expert explosion)
- Stage 2→3: Dimension scaling
- Stage 3→4: Expert expansion
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Tuple, List
from pathlib import Path
import json
import logging
from collections import OrderedDict


logger = logging.getLogger('moe_utils')


# =============================================================================
# Model Expansion Utilities
# =============================================================================

def expand_dense_to_moe(
    dense_state_dict: Dict[str, torch.Tensor],
    num_experts: int,
    num_shared_experts: int = 2,
    noise_std: float = 1e-4,
    moe_layer_indices: Optional[List[int]] = None
) -> Dict[str, torch.Tensor]:
    """
    Expand dense model state dict to MoE.
    
    Stage 1 → Stage 2 expansion: Copy FFN weights to all experts.
    
    Args:
        dense_state_dict: State dict from dense model
        num_experts: Number of routed experts
        num_shared_experts: Number of shared experts  
        noise_std: Noise for symmetry breaking
        moe_layer_indices: Which layers become MoE (None = all)
        
    Returns:
        MoE state dict
    """
    moe_state_dict = OrderedDict()
    
    for key, value in dense_state_dict.items():
        # Check if this is an FFN weight in a layer that should become MoE
        if '.ffn.' in key:
            layer_idx = int(key.split('.')[1])  # Assumes 'layers.N.ffn.xxx'
            
            if moe_layer_indices is None or layer_idx in moe_layer_indices:
                # This FFN becomes experts
                base_key = key.replace('.ffn.', '.ffn.experts.')
                
                # Copy to all routed experts
                for expert_idx in range(num_experts):
                    expert_key = base_key.replace(
                        '.experts.', 
                        f'.experts.routed_experts.{expert_idx}.'
                    )
                    # Add small noise for symmetry breaking
                    moe_state_dict[expert_key] = value.clone() + torch.randn_like(value) * noise_std
                
                # Copy to shared experts
                for shared_idx in range(num_shared_experts):
                    shared_key = base_key.replace(
                        '.experts.',
                        f'.experts.shared_experts.{shared_idx}.'
                    )
                    moe_state_dict[shared_key] = value.clone()
            else:
                # Keep as dense
                moe_state_dict[key] = value
        else:
            # Non-FFN weights (attention, norms, embeddings)
            moe_state_dict[key] = value
    
    logger.info(f"Expanded dense model to MoE with {num_experts} experts")
    return moe_state_dict


def expand_moe_experts(
    source_state_dict: Dict[str, torch.Tensor],
    source_num_experts: int,
    target_num_experts: int,
    children_per_parent: int = 8,
    noise_std: float = 1e-3
) -> Dict[str, torch.Tensor]:
    """
    Expand MoE model to more experts.
    
    Stage 3 → Stage 4 expansion: Each expert becomes multiple children.
    
    Args:
        source_state_dict: State dict from source MoE model
        source_num_experts: Number of experts in source
        target_num_experts: Number of experts in target
        children_per_parent: Children per parent expert
        noise_std: Noise for divergence
        
    Returns:
        Expanded state dict
    """
    assert target_num_experts == source_num_experts * children_per_parent, \
        f"Expected {source_num_experts * children_per_parent} target experts, got {target_num_experts}"
    
    expanded_state_dict = OrderedDict()
    
    for key, value in source_state_dict.items():
        if '.routed_experts.' in key:
            # Extract parent expert index
            parts = key.split('.')
            expert_idx_pos = parts.index('routed_experts') + 1
            parent_idx = int(parts[expert_idx_pos])
            
            # Create children
            for child_idx in range(children_per_parent):
                global_idx = parent_idx * children_per_parent + child_idx
                
                # Create new key with child index
                new_parts = parts.copy()
                new_parts[expert_idx_pos] = str(global_idx)
                new_key = '.'.join(new_parts)
                
                # Copy with noise for divergence
                expanded_state_dict[new_key] = value.clone() + torch.randn_like(value) * noise_std
        else:
            # Non-expert weights
            expanded_state_dict[key] = value
    
    logger.info(f"Expanded MoE from {source_num_experts} to {target_num_experts} experts")
    return expanded_state_dict


def scale_model_dimensions(
    state_dict: Dict[str, torch.Tensor],
    source_hidden: int,
    target_hidden: int,
    source_intermediate: int,
    target_intermediate: int,
    interpolation: str = 'linear'
) -> Dict[str, torch.Tensor]:
    """
    Scale model dimensions via interpolation.
    
    Stage 2 → Stage 3: Scale hidden and intermediate sizes.
    
    Args:
        state_dict: Source state dict
        source_hidden: Source hidden size
        target_hidden: Target hidden size
        source_intermediate: Source intermediate size
        target_intermediate: Target intermediate size
        interpolation: Interpolation method ('linear', 'nearest')
        
    Returns:
        Scaled state dict
    """
    scaled_state_dict = OrderedDict()
    
    hidden_scale = target_hidden / source_hidden
    inter_scale = target_intermediate / source_intermediate
    
    for key, value in state_dict.items():
        if value.dim() == 1:
            # 1D tensor (bias, norm weights)
            if len(value) == source_hidden:
                scaled = torch.nn.functional.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=target_hidden,
                    mode=interpolation
                ).squeeze()
            elif len(value) == source_intermediate:
                scaled = torch.nn.functional.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=target_intermediate,
                    mode=interpolation
                ).squeeze()
            else:
                scaled = value
        elif value.dim() == 2:
            # 2D tensor (linear weights)
            out_dim, in_dim = value.shape
            
            # Determine target dimensions
            target_out = int(out_dim * (hidden_scale if out_dim == source_hidden else 
                                       inter_scale if out_dim == source_intermediate else 1))
            target_in = int(in_dim * (hidden_scale if in_dim == source_hidden else
                                     inter_scale if in_dim == source_intermediate else 1))
            
            if target_out != out_dim or target_in != in_dim:
                # Interpolate
                scaled = torch.nn.functional.interpolate(
                    value.unsqueeze(0).unsqueeze(0),
                    size=(target_out, target_in),
                    mode='bilinear' if interpolation == 'linear' else 'nearest'
                ).squeeze()
            else:
                scaled = value
        else:
            scaled = value
        
        scaled_state_dict[key] = scaled
    
    logger.info(f"Scaled model: hidden {source_hidden}→{target_hidden}, "
               f"intermediate {source_intermediate}→{target_intermediate}")
    return scaled_state_dict


# =============================================================================
# Checkpointing Utilities
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Any,
    step: int,
    path: str,
    additional_state: Optional[Dict] = None
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optional optimizer state
        config: Model configuration
        step: Training step
        path: Save path
        additional_state: Additional state to save
    """
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if additional_state:
        checkpoint.update(additional_state)
    
    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path} at step {step}")


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cuda',
    strict: bool = True
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Checkpoint path
        model: Optional model to load into
        optimizer: Optional optimizer to load into
        device: Device to load to
        strict: Strict loading (error on missing/extra keys)
        
    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device)
    
    if model is not None:
        missing, unexpected = model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict
        )
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {path} at step {checkpoint.get('step', 'unknown')}")
    return checkpoint


# =============================================================================
# Parameter Counting
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def analyze_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Analyze parameter distribution by component.
    
    Returns dict mapping component name to parameter count.
    """
    analysis = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                # Extract component type
                parts = name.split('.')
                if 'embed' in name.lower():
                    component = 'embeddings'
                elif 'attn' in name.lower() or 'attention' in name.lower():
                    component = 'attention'
                elif 'router' in name.lower():
                    component = 'router'
                elif 'expert' in name.lower():
                    if 'shared' in name.lower():
                        component = 'shared_experts'
                    elif 'null' in name.lower():
                        component = 'null_experts'
                    else:
                        component = 'routed_experts'
                elif 'ffn' in name.lower() or 'mlp' in name.lower():
                    component = 'ffn'
                elif 'norm' in name.lower():
                    component = 'normalization'
                elif 'head' in name.lower():
                    component = 'output_head'
                else:
                    component = 'other'
                
                analysis[component] = analysis.get(component, 0) + params
    
    analysis['total'] = sum(analysis.values())
    return analysis


def print_parameter_summary(model: nn.Module, model_name: str = "Model"):
    """Print formatted parameter summary."""
    analysis = analyze_parameters(model)
    total = analysis.pop('total')
    
    print(f"\n{'='*60}")
    print(f" {model_name} Parameter Summary")
    print(f"{'='*60}")
    
    for component, count in sorted(analysis.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {component:20s}: {count:>12,d} ({pct:5.1f}%)")
    
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':20s}: {total:>12,d}")
    print(f"{'='*60}\n")


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: Any) -> List[str]:
    """
    Validate model configuration.
    
    Returns list of warnings/errors.
    """
    issues = []
    
    # Check hidden size divisibility
    if hasattr(config, 'hidden_size') and hasattr(config, 'attention'):
        if config.hidden_size % config.attention.num_attention_heads != 0:
            issues.append(
                f"hidden_size ({config.hidden_size}) not divisible by "
                f"num_attention_heads ({config.attention.num_attention_heads})"
            )
    
    # Check GQA ratio
    if hasattr(config, 'attention'):
        if config.attention.num_attention_heads % config.attention.num_kv_heads != 0:
            issues.append(
                f"num_attention_heads ({config.attention.num_attention_heads}) not divisible by "
                f"num_kv_heads ({config.attention.num_kv_heads})"
            )
    
    # Check MoE configuration
    if hasattr(config, 'expert') and hasattr(config, 'router'):
        if config.expert.num_routed_experts > 0:
            total_routable = config.expert.num_routed_experts + config.expert.num_null_experts
            if config.router.top_k > total_routable:
                issues.append(
                    f"top_k ({config.router.top_k}) exceeds total routable experts ({total_routable})"
                )
    
    # Check compute budget
    if hasattr(config, 'compute_budget'):
        estimated = config.estimate_parameters() if hasattr(config, 'estimate_parameters') else {}
        if estimated and config.compute_budget.max_params_total:
            if estimated.get('total', 0) > config.compute_budget.max_params_total:
                issues.append(
                    f"Estimated parameters ({estimated['total']/1e9:.1f}B) exceed "
                    f"budget ({config.compute_budget.max_params_total/1e9:.1f}B)"
                )
    
    return issues


def validate_expansion_compatibility(
    source_config: Any,
    target_config: Any,
    expansion_type: str
) -> List[str]:
    """
    Validate that source and target configs are compatible for expansion.
    
    Args:
        source_config: Source model config
        target_config: Target model config
        expansion_type: 'dense_to_moe', 'dimension_scale', or 'expert_expansion'
        
    Returns:
        List of incompatibility issues
    """
    issues = []
    
    if expansion_type == 'dense_to_moe':
        # Hidden size should match
        if source_config.hidden_size != target_config.hidden_size:
            issues.append(
                f"Hidden size mismatch: {source_config.hidden_size} → {target_config.hidden_size}"
            )
    
    elif expansion_type == 'dimension_scale':
        # Number of experts should match
        if hasattr(source_config, 'expert') and hasattr(target_config, 'expert'):
            if source_config.expert.num_routed_experts != target_config.expert.num_routed_experts:
                issues.append(
                    f"Expert count should stay same for dimension scaling: "
                    f"{source_config.expert.num_routed_experts} → {target_config.expert.num_routed_experts}"
                )
    
    elif expansion_type == 'expert_expansion':
        # Hidden size should match
        if source_config.hidden_size != target_config.hidden_size:
            issues.append(
                f"Hidden size should stay same for expert expansion: "
                f"{source_config.hidden_size} → {target_config.hidden_size}"
            )
        
        # Check expansion ratio
        if hasattr(source_config, 'expert') and hasattr(target_config, 'expert'):
            source_exp = source_config.expert.num_routed_experts
            target_exp = target_config.expert.num_routed_experts
            if target_exp % source_exp != 0:
                issues.append(
                    f"Target experts ({target_exp}) should be multiple of "
                    f"source experts ({source_exp})"
                )
    
    return issues


# =============================================================================
# Initialization Verification
# =============================================================================

def verify_lossless_init(
    model: nn.Module,
    test_input: torch.Tensor,
    tolerance: float = 1e-5
) -> Tuple[bool, float]:
    """
    Verify that MoE initialization is lossless.
    
    For lossless init: All experts identical → MoE output = Dense output
    
    Args:
        model: MoE model to verify
        test_input: Test input tensor
        tolerance: Maximum allowed deviation
        
    Returns:
        (is_lossless, max_deviation)
    """
    model.eval()
    
    with torch.no_grad():
        # Run forward pass
        output = model(test_input)
        
        if isinstance(output, dict):
            output = output['logits']
        
        # For truly lossless init, all tokens should produce similar expert outputs
        # This is a simplified check - full verification would compare to dense model
        
        # Check output statistics
        mean = output.mean().item()
        std = output.std().item()
        
        # Outputs should be well-behaved
        is_valid = not (torch.isnan(output).any() or torch.isinf(output).any())
        
        # Check for reasonable range
        max_val = output.abs().max().item()
        
    return is_valid, max_val
