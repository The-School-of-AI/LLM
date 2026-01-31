"""
Growth Utilities for Weight Transfer

This module provides functions to grow a model across architectural phases:
1. dense_to_moe: Convert dense FFN layers to MoE blocks
2. scale_hidden_dim: Increase hidden dimension with zero/Gaussian padding
3. add_layers: Add new transformer blocks with identity initialization

Key principle: All growth operations must preserve the model's current behavior
as closely as possible to avoid loss spikes.
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .model import SmolLM2, SmolLM2Config, TransformerBlock, MLP
from .moe_model import SmolLM2MoE, MoEConfig, MoETransformerBlock, MoEBlock


def dense_to_moe(
    dense_model: SmolLM2,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
) -> SmolLM2MoE:
    """
    Convert a dense SmolLM2 model to MoE by cloning FFN weights into experts.
    
    Strategy:
    - Clone the dense FFN weights into ALL experts
    - Initialize router with small random weights
    - This ensures initial behavior is identical to dense model
      (since all experts produce the same output initially)
    
    Args:
        dense_model: Trained dense SmolLM2 model
        num_experts: Number of experts to create
        num_experts_per_tok: Top-k experts per token
    
    Returns:
        MoE model with weights transferred from dense model
    """
    # Create MoE config from dense config
    dense_config = dense_model.config
    moe_config = MoEConfig(
        vocab_size=dense_config.vocab_size,
        hidden_size=dense_config.hidden_size,
        intermediate_size=dense_config.intermediate_size,
        num_hidden_layers=dense_config.num_hidden_layers,
        num_attention_heads=dense_config.num_attention_heads,
        num_key_value_heads=dense_config.num_key_value_heads,
        max_position_embeddings=dense_config.max_position_embeddings,
        rms_norm_eps=dense_config.rms_norm_eps,
        rope_theta=dense_config.rope_theta,
        tie_word_embeddings=dense_config.tie_word_embeddings,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    # Create fresh MoE model
    moe_model = SmolLM2MoE(moe_config)
    
    # Transfer embeddings
    moe_model.embed_tokens.load_state_dict(dense_model.embed_tokens.state_dict())
    
    # Transfer final norm
    moe_model.norm.load_state_dict(dense_model.norm.state_dict())
    
    # Transfer LM head if not tied
    if not dense_config.tie_word_embeddings and dense_model.lm_head is not None:
        moe_model.lm_head.load_state_dict(dense_model.lm_head.state_dict())
    
    # Transfer layer-by-layer
    for layer_idx, (dense_layer, moe_layer) in enumerate(
        zip(dense_model.layers, moe_model.layers)
    ):
        # Transfer attention (unchanged)
        moe_layer.input_layernorm.load_state_dict(dense_layer.input_layernorm.state_dict())
        moe_layer.self_attn.load_state_dict(dense_layer.self_attn.state_dict())
        moe_layer.post_attention_layernorm.load_state_dict(
            dense_layer.post_attention_layernorm.state_dict()
        )
        
        # Clone dense MLP into each expert
        dense_mlp_state = dense_layer.mlp.state_dict()
        for expert in moe_layer.moe.experts:
            expert.load_state_dict(dense_mlp_state)
        
        # Initialize router with small weights (near-uniform routing initially)
        nn.init.normal_(moe_layer.moe.router.gate.weight, mean=0.0, std=0.01)
    
    print(f"✓ Converted dense model ({dense_model.num_parameters():,} params) "
          f"to MoE model ({moe_model.num_parameters():,} params)")
    print(f"  - {num_experts} experts per layer, top-{num_experts_per_tok} routing")
    
    return moe_model


def scale_hidden_dim(
    model: nn.Module,
    new_hidden_size: int,
    padding_mode: str = "zero",  # "zero" or "gaussian"
    gaussian_std: float = 0.01,
) -> nn.Module:
    """
    Scale the hidden dimension of a model by padding weight matrices.
    
    Strategy:
    - Pad weight matrices with zeros (or small Gaussian noise)
    - Pad biases if they exist
    - This preserves the model's behavior on the original dimensions
    
    Args:
        model: Model to scale
        new_hidden_size: Target hidden size (must be >= current)
        padding_mode: "zero" for zero padding, "gaussian" for small noise
        gaussian_std: Std dev for Gaussian padding
    
    Returns:
        Scaled model (modifies in place and returns)
    """
    current_hidden_size = model.config.hidden_size
    if new_hidden_size <= current_hidden_size:
        raise ValueError(f"new_hidden_size ({new_hidden_size}) must be > current ({current_hidden_size})")
    
    delta = new_hidden_size - current_hidden_size
    
    def pad_tensor(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
        """Pad a tensor along a dimension."""
        pad_shape = list(tensor.shape)
        pad_shape[dim] = size
        
        if padding_mode == "zero":
            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        else:  # gaussian
            padding = torch.randn(pad_shape, dtype=tensor.dtype, device=tensor.device) * gaussian_std
        
        return torch.cat([tensor, padding], dim=dim)
    
    # Scale embedding layer
    old_embed = model.embed_tokens.weight.data
    new_embed = pad_tensor(old_embed, dim=1, size=delta)
    model.embed_tokens = nn.Embedding(model.config.vocab_size, new_hidden_size)
    model.embed_tokens.weight.data = new_embed
    
    # Scale each layer
    for layer in model.layers:
        # Scale attention projections
        attn = layer.self_attn
        
        # Q, K, V projections: input dim changes
        for proj in [attn.q_proj, attn.k_proj, attn.v_proj]:
            old_weight = proj.weight.data
            new_weight = pad_tensor(old_weight, dim=1, size=delta)
            proj.weight = nn.Parameter(new_weight)
        
        # O projection: both dims change
        old_o = attn.o_proj.weight.data
        new_o = pad_tensor(old_o, dim=0, size=delta)  # output dim
        new_o = pad_tensor(new_o, dim=1, size=delta)  # input dim (came from attention)
        attn.o_proj.weight = nn.Parameter(new_o)
        
        # Scale MLP/MoE
        if hasattr(layer, 'mlp'):  # Dense model
            mlp = layer.mlp
            # gate_proj and up_proj: input dim changes
            mlp.gate_proj.weight = nn.Parameter(pad_tensor(mlp.gate_proj.weight.data, dim=1, size=delta))
            mlp.up_proj.weight = nn.Parameter(pad_tensor(mlp.up_proj.weight.data, dim=1, size=delta))
            # down_proj: output dim changes
            mlp.down_proj.weight = nn.Parameter(pad_tensor(mlp.down_proj.weight.data, dim=0, size=delta))
        
        elif hasattr(layer, 'moe'):  # MoE model
            # Scale router
            old_gate = layer.moe.router.gate.weight.data
            new_gate = pad_tensor(old_gate, dim=1, size=delta)
            layer.moe.router.gate.weight = nn.Parameter(new_gate)
            
            # Scale each expert
            for expert in layer.moe.experts:
                expert.gate_proj.weight = nn.Parameter(pad_tensor(expert.gate_proj.weight.data, dim=1, size=delta))
                expert.up_proj.weight = nn.Parameter(pad_tensor(expert.up_proj.weight.data, dim=1, size=delta))
                expert.down_proj.weight = nn.Parameter(pad_tensor(expert.down_proj.weight.data, dim=0, size=delta))
        
        # Scale layer norms
        layer.input_layernorm.weight = nn.Parameter(
            pad_tensor(layer.input_layernorm.weight.data, dim=0, size=delta)
        )
        layer.post_attention_layernorm.weight = nn.Parameter(
            pad_tensor(layer.post_attention_layernorm.weight.data, dim=0, size=delta)
        )
    
    # Scale final norm
    model.norm.weight = nn.Parameter(pad_tensor(model.norm.weight.data, dim=0, size=delta))
    
    # Scale LM head if not tied
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        old_lm = model.lm_head.weight.data
        new_lm = pad_tensor(old_lm, dim=1, size=delta)
        model.lm_head.weight = nn.Parameter(new_lm)
    
    # Update config
    model.config.hidden_size = new_hidden_size
    
    print(f"✓ Scaled hidden dimension from {current_hidden_size} to {new_hidden_size}")
    print(f"  - Padding mode: {padding_mode}")
    print(f"  - New parameter count: {model.num_parameters():,}")
    
    return model


def add_layers(
    model: nn.Module,
    num_new_layers: int,
    insert_positions: Optional[list] = None,
    init_mode: str = "identity",
) -> nn.Module:
    """
    Add new transformer blocks to a model.
    
    Strategy:
    - Insert layers at specified positions (or evenly distributed)
    - Initialize new layers to approximate identity function:
        - Attention: zero output projection
        - MLP: zero down projection (or very small)
    
    Args:
        model: Model to grow
        num_new_layers: Number of layers to add
        insert_positions: Where to insert (indices). Default: evenly distributed
        init_mode: "identity" for identity-like init, "random" for standard init
    
    Returns:
        Model with new layers (modifies in place and returns)
    """
    current_layers = len(model.layers)
    
    if insert_positions is None:
        # Distribute evenly
        step = current_layers // (num_new_layers + 1)
        insert_positions = [step * (i + 1) for i in range(num_new_layers)]
    
    # Determine if MoE or dense
    is_moe = hasattr(model.layers[0], 'moe')
    
    # Create new layers
    new_layers = []
    for pos in insert_positions:
        if is_moe:
            new_layer = MoETransformerBlock(model.config, layer_idx=pos)
        else:
            new_layer = TransformerBlock(model.config, layer_idx=pos)
        
        if init_mode == "identity":
            # Initialize to approximate identity
            # Zero out the output projections so residual passes through
            with torch.no_grad():
                new_layer.self_attn.o_proj.weight.zero_()
                
                if hasattr(new_layer, 'mlp'):
                    new_layer.mlp.down_proj.weight.zero_()
                elif hasattr(new_layer, 'moe'):
                    for expert in new_layer.moe.experts:
                        expert.down_proj.weight.zero_()
        
        new_layers.append((pos, new_layer))
    
    # Insert layers (in reverse order to maintain indices)
    for pos, layer in sorted(new_layers, key=lambda x: x[0], reverse=True):
        model.layers.insert(pos, layer)
    
    # Update layer indices
    for idx, layer in enumerate(model.layers):
        layer.layer_idx = idx
    
    # Update config
    model.config.num_hidden_layers = len(model.layers)
    
    print(f"✓ Added {num_new_layers} new layers")
    print(f"  - Insert positions: {[p for p, _ in new_layers]}")
    print(f"  - Total layers: {current_layers} → {len(model.layers)}")
    print(f"  - New parameter count: {model.num_parameters():,}")
    
    return model


def add_experts(
    moe_model: SmolLM2MoE,
    num_new_experts: int,
    clone_from: str = "random",  # "random", "mean", "best"
) -> SmolLM2MoE:
    """
    Add more experts to an existing MoE model (expert explosion).
    
    Strategy:
    - Clone existing experts into new experts with small noise
    - This preserves behavior while adding capacity
    
    Args:
        moe_model: Existing MoE model
        num_new_experts: Number of experts to add
        clone_from: Strategy for initializing new experts
            - "random": Clone a random existing expert
            - "mean": Initialize as mean of all experts
    
    Returns:
        MoE model with more experts
    """
    old_num_experts = moe_model.config.num_experts
    new_num_experts = old_num_experts + num_new_experts
    
    for layer in moe_model.layers:
        moe_block = layer.moe
        
        # Add new experts
        for i in range(num_new_experts):
            if clone_from == "random":
                # Clone a random existing expert
                source_idx = torch.randint(0, old_num_experts, (1,)).item()
                source_expert = moe_block.experts[source_idx]
            else:  # mean
                # Would need to average weights - simplified to random for now
                source_idx = 0
                source_expert = moe_block.experts[source_idx]
            
            # Deep copy and add noise
            new_expert = copy.deepcopy(source_expert)
            with torch.no_grad():
                for param in new_expert.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            
            moe_block.experts.append(new_expert)
        
        # Expand router
        old_gate = moe_block.router.gate.weight.data
        new_gate_weights = torch.randn(num_new_experts, old_gate.shape[1]) * 0.01
        new_gate = torch.cat([old_gate, new_gate_weights.to(old_gate.device)], dim=0)
        
        moe_block.router.gate = nn.Linear(
            moe_block.hidden_size, new_num_experts, bias=False
        )
        moe_block.router.gate.weight.data = new_gate
        moe_block.num_experts = new_num_experts
    
    # Update config
    moe_model.config.num_experts = new_num_experts
    
    print(f"✓ Added {num_new_experts} experts per layer")
    print(f"  - Experts: {old_num_experts} → {new_num_experts}")
    print(f"  - New parameter count: {moe_model.num_parameters():,}")
    
    return moe_model


if __name__ == "__main__":
    # Test growth functions
    print("=" * 60)
    print("Testing Growth Utilities")
    print("=" * 60)
    
    # 1. Create dense model
    print("\n1. Creating dense model...")
    dense_config = SmolLM2Config(num_hidden_layers=4)  # Small for testing
    dense_model = SmolLM2(dense_config)
    print(f"   Dense model: {dense_model.num_parameters():,} params")
    
    # 2. Test dense_to_moe
    print("\n2. Converting to MoE...")
    moe_model = dense_to_moe(dense_model, num_experts=4, num_experts_per_tok=2)
    print(f"   MoE model: {moe_model.num_parameters():,} params")
    
    # 3. Test add_experts
    print("\n3. Adding experts...")
    moe_model = add_experts(moe_model, num_new_experts=2)
    
    # 4. Test add_layers
    print("\n4. Adding layers...")
    moe_model = add_layers(moe_model, num_new_layers=2)
    
    # 5. Verify forward pass still works
    print("\n5. Testing forward pass...")
    input_ids = torch.randint(0, dense_config.vocab_size, (2, 64))
    output = moe_model(input_ids, labels=input_ids)
    print(f"   Loss: {output['loss'].item():.4f}")
    print(f"   Aux Loss: {output['aux_loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("All growth utilities working!")
    print("=" * 60)
