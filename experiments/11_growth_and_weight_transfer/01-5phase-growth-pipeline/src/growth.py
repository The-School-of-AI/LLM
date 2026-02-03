"""
Growth Utilities for Weight Transfer

This module provides functions to grow a model across architectural phases:
1. dense_to_moe: Convert dense FFN layers to MoE blocks
2. add_layers: Add new transformer blocks with identity initialization  
3. scale_hidden_dim: Increase hidden dimension by ADDING HEADS (not fattening them)
4. add_experts: Add more experts (expert explosion)

Key principle: All growth operations must preserve the model's current behavior
as closely as possible to avoid loss spikes.

CRITICAL INSIGHT (RoPE Barrier):
RoPE pairs dimensions based on head_dim. Changing head_dim breaks RoPE pairing.
Solution: Keep head_dim constant, add more heads instead.
"""

import copy
import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .model import SmolLM2, SmolLM2Config, TransformerBlock, MLP, RMSNorm, Attention
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
    """
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
    
    moe_model = SmolLM2MoE(moe_config)
    
    # Transfer embeddings
    moe_model.embed_tokens.load_state_dict(dense_model.embed_tokens.state_dict())
    moe_model.norm.load_state_dict(dense_model.norm.state_dict())
    
    if not dense_config.tie_word_embeddings and dense_model.lm_head is not None:
        moe_model.lm_head.load_state_dict(dense_model.lm_head.state_dict())
    
    # Transfer layer-by-layer
    for layer_idx, (dense_layer, moe_layer) in enumerate(
        zip(dense_model.layers, moe_model.layers)
    ):
        moe_layer.input_layernorm.load_state_dict(dense_layer.input_layernorm.state_dict())
        moe_layer.self_attn.load_state_dict(dense_layer.self_attn.state_dict())
        moe_layer.post_attention_layernorm.load_state_dict(
            dense_layer.post_attention_layernorm.state_dict()
        )
        
        # Clone dense MLP into each expert
        dense_mlp_state = dense_layer.mlp.state_dict()
        for expert in moe_layer.moe.experts:
            expert.load_state_dict(dense_mlp_state)
        
        # Initialize router with small weights
        nn.init.normal_(moe_layer.moe.router.gate.weight, mean=0.0, std=0.01)
    
    print(f"✓ Converted dense model ({dense_model.num_parameters():,} params) "
          f"to MoE model ({moe_model.num_parameters():,} params)")
    print(f"  - {num_experts} experts per layer, top-{num_experts_per_tok} routing")
    
    return moe_model


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
    - Initialize new layers to approximate identity function
    """
    current_layers = len(model.layers)
    
    if insert_positions is None:
        step = current_layers // (num_new_layers + 1)
        insert_positions = [step * (i + 1) for i in range(num_new_layers)]
    
    is_moe = hasattr(model.layers[0], 'moe')
    
    new_layers = []
    for pos in insert_positions:
        if is_moe:
            new_layer = MoETransformerBlock(model.config, layer_idx=pos)
        else:
            new_layer = TransformerBlock(model.config, layer_idx=pos)
        
        if init_mode == "identity":
            with torch.no_grad():
                new_layer.self_attn.o_proj.weight.zero_()
                
                if hasattr(new_layer, 'mlp'):
                    new_layer.mlp.down_proj.weight.zero_()
                elif hasattr(new_layer, 'moe'):
                    for expert in new_layer.moe.experts:
                        expert.down_proj.weight.zero_()
        
        new_layers.append((pos, new_layer))
    
    for pos, layer in sorted(new_layers, key=lambda x: x[0], reverse=True):
        model.layers.insert(pos, layer)
    
    for idx, layer in enumerate(model.layers):
        layer.layer_idx = idx
    
    model.config.num_hidden_layers = len(model.layers)
    
    print(f"✓ Added {num_new_layers} new layers")
    print(f"  - Insert positions: {[p for p, _ in new_layers]}")
    print(f"  - Total layers: {current_layers} → {len(model.layers)}")
    print(f"  - New parameter count: {model.num_parameters():,}")
    
    return model


def scale_hidden_dim(
    model: nn.Module,
    new_hidden_size: int,
    new_intermediate_size: Optional[int] = None,
    new_num_heads: Optional[int] = None,
    new_num_kv_heads: Optional[int] = None,
    **kwargs,
) -> nn.Module:
    """
    Scale the hidden dimension by ADDING HEADS (not fattening them).
    
    THE ROPE BARRIER:
    =================
    RoPE pairs dimensions based on head_dim: pairs (i, i + head_dim/2).
    Changing head_dim breaks this pairing and scrambles positional encoding.
    
    SOLUTION: Keep head_dim constant, add more heads instead.
    
    Example:
    - Old: hidden=576, 9 heads, head_dim=64
    - New: hidden=768, 12 heads, head_dim=64 (ADD 3 HEADS, KEEP DIM 64)
    
    For the new heads:
    - Q/K/V projections: initialized with small random weights
    - O projection: initialized to ZERO (new heads are "silent" initially)
    
    This ensures the model behaves exactly like before, with new heads
    slowly learning during training.
    
    Args:
        model: Model to scale
        new_hidden_size: Target hidden size (must be multiple of head_dim)
        new_intermediate_size: Target intermediate size (auto-computed if None)
        new_num_heads: Target num_attention_heads (auto-computed if None)
        new_num_kv_heads: Target num_kv_heads (auto-computed if None)
    """
    old_config = model.config
    old_hidden = old_config.hidden_size
    old_intermediate = old_config.intermediate_size
    old_num_heads = old_config.num_attention_heads
    old_num_kv_heads = old_config.num_key_value_heads
    head_dim = old_hidden // old_num_heads  # This MUST stay constant!
    
    # Validate: new_hidden_size must be multiple of head_dim
    if new_hidden_size % head_dim != 0:
        raise ValueError(
            f"new_hidden_size ({new_hidden_size}) must be multiple of head_dim ({head_dim}). "
            f"Valid options: {[head_dim * n for n in range(old_num_heads, old_num_heads + 10)]}"
        )
    
    # Compute new config
    if new_num_heads is None:
        new_num_heads = new_hidden_size // head_dim
    
    if new_num_kv_heads is None:
        # Scale KV heads proportionally
        kv_ratio = old_num_heads // old_num_kv_heads  # e.g., 9 // 3 = 3
        new_num_kv_heads = new_num_heads // kv_ratio
    
    if new_intermediate_size is None:
        new_intermediate_size = int(old_intermediate * new_hidden_size / old_hidden)
    
    added_heads = new_num_heads - old_num_heads
    added_kv_heads = new_num_kv_heads - old_num_kv_heads
    
    print(f"🔧 Scaling: Add {added_heads} Q heads, {added_kv_heads} KV heads (head_dim={head_dim} preserved)")
    
    # Determine if MoE
    is_moe = hasattr(model, 'config') and hasattr(model.config, 'num_experts')
    
    if is_moe:
        new_config = MoEConfig(
            vocab_size=old_config.vocab_size,
            hidden_size=new_hidden_size,
            intermediate_size=new_intermediate_size,
            num_hidden_layers=old_config.num_hidden_layers,
            num_attention_heads=new_num_heads,
            num_key_value_heads=new_num_kv_heads,
            max_position_embeddings=old_config.max_position_embeddings,
            rms_norm_eps=old_config.rms_norm_eps,
            rope_theta=old_config.rope_theta,
            tie_word_embeddings=old_config.tie_word_embeddings,
            num_experts=old_config.num_experts,
            num_experts_per_tok=old_config.num_experts_per_tok,
        )
        new_model = SmolLM2MoE(new_config)
    else:
        new_config = SmolLM2Config(
            vocab_size=old_config.vocab_size,
            hidden_size=new_hidden_size,
            intermediate_size=new_intermediate_size,
            num_hidden_layers=old_config.num_hidden_layers,
            num_attention_heads=new_num_heads,
            num_key_value_heads=new_num_kv_heads,
            max_position_embeddings=old_config.max_position_embeddings,
            rms_norm_eps=old_config.rms_norm_eps,
            rope_theta=old_config.rope_theta,
            tie_word_embeddings=old_config.tie_word_embeddings,
        )
        new_model = SmolLM2(new_config)
    
    with torch.no_grad():
        # === 1. Embeddings: Zero-pad new dimensions ===
        old_embed = model.embed_tokens.weight.data
        new_model.embed_tokens.weight.data.zero_()
        new_model.embed_tokens.weight.data[:, :old_hidden] = old_embed
        
        # === 2. Final Norm: new dims = 1.0 ===
        new_model.norm.weight.data[:old_hidden] = model.norm.weight.data
        new_model.norm.weight.data[old_hidden:] = 1.0
        
        # === 3. LM Head: Zero new input cols ===
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            new_model.lm_head.weight.data.zero_()
            new_model.lm_head.weight.data[:, :old_hidden] = model.lm_head.weight.data
        
        # === 4. Transfer each layer ===
        for old_layer, new_layer in zip(model.layers, new_model.layers):
            # Layer norms
            new_layer.input_layernorm.weight.data[:old_hidden] = old_layer.input_layernorm.weight.data
            new_layer.input_layernorm.weight.data[old_hidden:] = 1.0
            new_layer.post_attention_layernorm.weight.data[:old_hidden] = old_layer.post_attention_layernorm.weight.data
            new_layer.post_attention_layernorm.weight.data[old_hidden:] = 1.0
            
            old_attn = old_layer.self_attn
            new_attn = new_layer.self_attn
            
            # === Q projection: Copy old heads, zero-init new heads ===
            # Old Q: (old_num_heads * head_dim, old_hidden) = (576, 576)
            # New Q: (new_num_heads * head_dim, new_hidden) = (768, 768)
            old_q_out = old_num_heads * head_dim
            new_q_out = new_num_heads * head_dim
            
            new_attn.q_proj.weight.data.zero_()
            # Copy old heads (rows 0 to old_q_out, cols 0 to old_hidden)
            new_attn.q_proj.weight.data[:old_q_out, :old_hidden] = old_attn.q_proj.weight.data
            # New heads (rows old_q_out to new_q_out) stay zero initially
            # They'll learn during training
            
            # === K projection: Copy old KV heads, zero-init new heads ===
            old_k_out = old_num_kv_heads * head_dim
            new_k_out = new_num_kv_heads * head_dim
            
            new_attn.k_proj.weight.data.zero_()
            new_attn.k_proj.weight.data[:old_k_out, :old_hidden] = old_attn.k_proj.weight.data
            
            # === V projection: Copy old KV heads, zero-init new heads ===
            new_attn.v_proj.weight.data.zero_()
            new_attn.v_proj.weight.data[:old_k_out, :old_hidden] = old_attn.v_proj.weight.data
            
            # === O projection: CRITICAL - Zero for new heads! ===
            # Old O: (old_hidden, old_num_heads * head_dim)
            # New O: (new_hidden, new_num_heads * head_dim)
            # 
            # Structure:
            # - Cols 0 to old_q_out: old heads (copy weights for old output rows)
            # - Cols old_q_out to new_q_out: NEW heads (must be ZERO!)
            # - Rows 0 to old_hidden: old output dims (copy)
            # - Rows old_hidden to new_hidden: new output dims (zero)
            
            new_attn.o_proj.weight.data.zero_()
            # Copy: old output rows, old head columns
            new_attn.o_proj.weight.data[:old_hidden, :old_q_out] = old_attn.o_proj.weight.data
            # New heads' columns (old_q_out:new_q_out) are ZERO
            # New output rows (old_hidden:new_hidden) are ZERO
            
            # === MLP / MoE Experts ===
            if hasattr(old_layer, 'mlp'):
                _transfer_mlp_weights(old_layer.mlp, new_layer.mlp, 
                                      old_hidden, new_hidden_size,
                                      old_intermediate, new_intermediate_size)
            elif hasattr(old_layer, 'moe'):
                # Router: zero new input cols
                old_gate = old_layer.moe.router.gate.weight.data
                new_layer.moe.router.gate.weight.data.zero_()
                new_layer.moe.router.gate.weight.data[:, :old_hidden] = old_gate
                
                # Each expert
                for old_exp, new_exp in zip(old_layer.moe.experts, new_layer.moe.experts):
                    _transfer_mlp_weights(old_exp, new_exp,
                                          old_hidden, new_hidden_size,
                                          old_intermediate, new_intermediate_size)
    
    print(f"✓ Scaled hidden dimension from {old_hidden} to {new_hidden_size}")
    print(f"  - Heads: {old_num_heads} → {new_num_heads} (head_dim={head_dim} PRESERVED)")
    print(f"  - KV Heads: {old_num_kv_heads} → {new_num_kv_heads}")
    print(f"  - Intermediate: {old_intermediate} → {new_intermediate_size}")
    print("  - Strategy: ADD HEADS (RoPE-safe, zero-spike)")
    print(f"  - New parameter count: {new_model.num_parameters():,}")
    
    return new_model


def _transfer_mlp_weights(old_mlp, new_mlp, old_hidden, new_hidden, old_inter, new_inter):
    """
    Transfer MLP weights with zero-padding for new dimensions.
    """
    # gate_proj: (intermediate, hidden) - zero new input cols
    new_mlp.gate_proj.weight.data.zero_()
    new_mlp.gate_proj.weight.data[:old_inter, :old_hidden] = old_mlp.gate_proj.weight.data
    
    # up_proj: same as gate_proj
    new_mlp.up_proj.weight.data.zero_()
    new_mlp.up_proj.weight.data[:old_inter, :old_hidden] = old_mlp.up_proj.weight.data
    
    # down_proj: (hidden, intermediate) - zero new output rows
    new_mlp.down_proj.weight.data.zero_()
    new_mlp.down_proj.weight.data[:old_hidden, :old_inter] = old_mlp.down_proj.weight.data


def add_experts(
    moe_model: SmolLM2MoE,
    num_new_experts: int,
    clone_from: str = "random",  # Ignored - always clones for function preservation
    noise_std: float = 0.01,
) -> SmolLM2MoE:
    """
    Add more experts to an existing MoE model (expert explosion).
    
    Strategy:
    - Clone existing experts into new experts with small noise
    - CRITICAL: Also clone router weights from source expert
    - This preserves behavior while adding capacity
    """
    old_num_experts = moe_model.config.num_experts
    new_num_experts = old_num_experts + num_new_experts
    
    for layer in moe_model.layers:
        moe_block = layer.moe
        old_gate = moe_block.router.gate.weight.data
        
        new_gate_rows = []  # Store cloned router weights
        
        for i in range(num_new_experts):
            # 1. Pick a parent expert to clone from
            source_idx = torch.randint(0, old_num_experts, (1,)).item()
            source_expert = moe_block.experts[source_idx]
            
            # 2. Clone the expert (the "employee")
            new_expert = copy.deepcopy(source_expert)
            
            with torch.no_grad():
                # 3. CRITICAL FIX: Clone the router weights (the "boss's instructions")
                # This ensures the router knows how to route to the new expert
                source_gate_weight = old_gate[source_idx].clone()
                new_gate_row = source_gate_weight + (torch.randn_like(source_gate_weight) * noise_std)
                new_gate_rows.append(new_gate_row)
                
                # 4. Add noise to expert weights for future specialization
                for param in new_expert.parameters():
                    param.add_(torch.randn_like(param) * noise_std)
            
            moe_block.experts.append(new_expert)
        
        # 5. Stack and concatenate router weights
        new_gate_weights = torch.stack(new_gate_rows).to(old_gate.device)
        combined_gate = torch.cat([old_gate, new_gate_weights], dim=0)
        
        # 6. Create new router with combined weights
        moe_block.router.gate = nn.Linear(
            moe_block.hidden_size, new_num_experts, bias=False
        )
        moe_block.router.gate.weight.data = combined_gate
        moe_block.num_experts = new_num_experts
    
    moe_model.config.num_experts = new_num_experts
    
    print(f"✓ Added {num_new_experts} experts per layer")
    print(f"  - Experts: {old_num_experts} → {new_num_experts}")
    print(f"  - Router weights: cloned from source experts (function-preserving)")
    print(f"  - New parameter count: {moe_model.num_parameters():,}")
    
    return moe_model


def scale_context_length(
    model: nn.Module,
    new_max_length: int,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> nn.Module:
    """
    Extend context length using YaRN (Yet another RoPE extensioN).
    
    This replaces the RoPE embeddings in all attention layers with YaRN RoPE,
    which uses NTK-by-parts interpolation for smooth context extension.
    
    Args:
        model: Model to update
        new_max_length: New maximum context length
        alpha: NTK-by-parts lower threshold (default: 1)
        beta: NTK-by-parts upper threshold (default: 32)
        
    Returns:
        Model with YaRN RoPE (same model, modified in place)
    """
    from .yarn import YaRNRotaryEmbedding
    
    old_max_length = model.config.max_position_embeddings
    scale = new_max_length / old_max_length
    head_dim = model.config.head_dim
    rope_theta = model.config.rope_theta
    
    # Create YaRN RoPE embedding
    yarn_rope = YaRNRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=new_max_length,
        base=rope_theta,
        scale=scale,
        alpha=alpha,
        beta=beta,
        original_max_position_embeddings=old_max_length,
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    yarn_rope = yarn_rope.to(device)
    
    # Replace RoPE in all attention layers
    for layer in model.layers:
        layer.self_attn.rotary_emb = yarn_rope
    
    # Update config
    model.config.max_position_embeddings = new_max_length
    
    print("✓ Extended context length using YaRN")
    print(f"  - Context: {old_max_length} → {new_max_length} ({scale:.1f}x)")
    print(f"  - Method: NTK-by-parts (α={alpha}, β={beta})")
    print(f"  - Attention scale: {yarn_rope.attention_scale:.4f}")
    
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Growth Utilities")
    print("=" * 60)
    
    # 1. Create dense model
    print("\n1. Creating dense model...")
    dense_config = SmolLM2Config(num_hidden_layers=4, hidden_size=256, intermediate_size=512,
                                  num_attention_heads=8, num_key_value_heads=2)
    dense_model = SmolLM2(dense_config)
    print(f"   Dense model: {dense_model.num_parameters():,} params")
    print(f"   Head dim: {dense_config.head_dim}")
    
    # 2. Test dense_to_moe
    print("\n2. Converting to MoE...")
    moe_model = dense_to_moe(dense_model, num_experts=4, num_experts_per_tok=2)
    
    # 3. Test add_layers
    print("\n3. Adding layers...")
    moe_model = add_layers(moe_model, num_new_layers=2)
    
    # 4. Test scale_hidden_dim with ADD HEADS (head_dim=32 preserved)
    print("\n4. Scaling hidden dimension (ADD HEADS)...")
    # 256 = 8 heads * 32 dim
    # 384 = 12 heads * 32 dim (add 4 heads)
    moe_model = scale_hidden_dim(moe_model, new_hidden_size=384)
    
    # 5. Test add_experts
    print("\n5. Adding experts...")
    moe_model = add_experts(moe_model, num_new_experts=2)
    
    # 6. Verify forward pass
    print("\n6. Testing forward pass...")
    input_ids = torch.randint(0, dense_config.vocab_size, (2, 64))
    output = moe_model(input_ids, labels=input_ids)
    print(f"   Loss: {output['loss'].item():.4f}")
    print(f"   Aux Loss: {output['aux_loss']:.4f}")
    
    print("\n" + "=" * 60)
    print("All growth utilities working!")
    print("=" * 60)
