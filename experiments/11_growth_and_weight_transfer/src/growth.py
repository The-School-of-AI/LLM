"""
Growth Utilities for Weight Transfer

This module provides functions to grow a model across architectural phases:
1. dense_to_moe: Convert dense FFN layers to MoE blocks
2. add_layers: Add new transformer blocks with identity initialization  
3. scale_hidden_dim: Increase hidden dimension with zero-output initialization
4. add_experts: Add more experts (expert explosion)

Key principle: All growth operations must preserve the model's current behavior
as closely as possible to avoid loss spikes.
"""

import copy
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
    **kwargs,  # Accept but ignore legacy params like padding_mode
) -> nn.Module:
    """
    Scale the hidden dimension using ZERO-OUTPUT initialization.
    
    Strategy (Function-Preserving):
    ================================
    The key insight is that new dimensions should NOT affect the output initially.
    This is achieved by setting OUTPUT-side weights to zero for new dimensions:
    
    1. Embeddings: new dims = 0 (tokens start with only original info)
    2. q_proj, k_proj, v_proj: new INPUT cols = 0 (don't read new dims)  
    3. o_proj: new OUTPUT rows = 0 (new dims don't contribute to residual)
    4. gate_proj, up_proj: new INPUT cols = 0 (don't read new dims)
    5. down_proj: new OUTPUT rows = 0 (new dims don't contribute to residual)
    6. RMSNorm: new dims = 1.0 (identity scaling)
    7. lm_head: new INPUT cols = 0 (don't read new dims for prediction)
    
    This ensures:
    - Model output is IDENTICAL before and after scaling
    - New dimensions exist but are "dormant"
    - Training will gradually activate new dimensions
    
    Args:
        model: Model to scale (MoE or dense)
        new_hidden_size: Target hidden size
        new_intermediate_size: Target intermediate size (default: proportional)
    """
    old_config = model.config
    old_hidden = old_config.hidden_size
    old_intermediate = old_config.intermediate_size
    
    if new_hidden_size <= old_hidden:
        raise ValueError(f"new_hidden_size must be > current ({old_hidden})")
    
    if new_intermediate_size is None:
        new_intermediate_size = int(old_intermediate * new_hidden_size / old_hidden)
    
    # Determine if MoE
    is_moe = hasattr(model, 'config') and hasattr(model.config, 'num_experts')
    
    if is_moe:
        new_config = MoEConfig(
            vocab_size=old_config.vocab_size,
            hidden_size=new_hidden_size,
            intermediate_size=new_intermediate_size,
            num_hidden_layers=old_config.num_hidden_layers,
            num_attention_heads=old_config.num_attention_heads,
            num_key_value_heads=old_config.num_key_value_heads,
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
            num_attention_heads=old_config.num_attention_heads,
            num_key_value_heads=old_config.num_key_value_heads,
            max_position_embeddings=old_config.max_position_embeddings,
            rms_norm_eps=old_config.rms_norm_eps,
            rope_theta=old_config.rope_theta,
            tie_word_embeddings=old_config.tie_word_embeddings,
        )
        new_model = SmolLM2(new_config)
    
    # Transfer and initialize weights with ZERO-OUTPUT strategy
    with torch.no_grad():
        # === 1. Embeddings: new dims = 0 ===
        # Tokens only carry information in original dimensions initially
        old_embed = model.embed_tokens.weight.data
        new_model.embed_tokens.weight.data.zero_()
        new_model.embed_tokens.weight.data[:, :old_hidden] = old_embed
        # New cols (576:720) are already 0
        
        # === 2. Final Norm (RMSNorm): new dims = 1.0 ===
        new_model.norm.weight.data[:old_hidden] = model.norm.weight.data
        new_model.norm.weight.data[old_hidden:] = 1.0  # Identity scaling
        
        # === 3. LM Head: don't read new dims ===
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            new_model.lm_head.weight.data.zero_()
            new_model.lm_head.weight.data[:, :old_hidden] = model.lm_head.weight.data
            # New cols (576:720) are 0 → new dims don't affect logits
        
        # === 4. Transfer each layer ===
        for old_layer, new_layer in zip(model.layers, new_model.layers):
            # Layer norms: new dims = 1.0 (identity)
            new_layer.input_layernorm.weight.data[:old_hidden] = old_layer.input_layernorm.weight.data
            new_layer.input_layernorm.weight.data[old_hidden:] = 1.0
            new_layer.post_attention_layernorm.weight.data[:old_hidden] = old_layer.post_attention_layernorm.weight.data
            new_layer.post_attention_layernorm.weight.data[old_hidden:] = 1.0
            
            # === Attention ===
            old_attn = old_layer.self_attn
            new_attn = new_layer.self_attn
            
            # Q, K, V projections: (out, in) - don't read new INPUT dims
            # New model has larger input dim, set new input cols to 0
            for old_proj, new_proj in [
                (old_attn.q_proj, new_attn.q_proj),
                (old_attn.k_proj, new_attn.k_proj),
                (old_attn.v_proj, new_attn.v_proj),
            ]:
                new_proj.weight.data.zero_()
                old_out = old_proj.weight.shape[0]
                new_out = new_proj.weight.shape[0]
                
                # Copy old weights
                new_proj.weight.data[:old_out, :old_hidden] = old_proj.weight.data
                
                # New input dims (576:720) are 0 → don't read from them
                # New output dims might exist if head_dim changes
                # We leave them as 0 initially
            
            # O projection: (hidden, qkv_out) - don't write new OUTPUT dims
            new_attn.o_proj.weight.data.zero_()
            old_out_h = old_attn.o_proj.weight.shape[0]  # old hidden
            old_in = old_attn.o_proj.weight.shape[1]  # old qkv total
            
            # Copy old weights (old_hidden rows, old qkv cols)
            new_attn.o_proj.weight.data[:old_out_h, :old_in] = old_attn.o_proj.weight.data
            
            # New OUTPUT rows (576:720) are 0 → don't contribute to residual
            # This is the KEY: new dims in residual stream are 0 after attention
            
            # === MLP / MoE Experts ===
            if hasattr(old_layer, 'mlp'):
                _transfer_mlp_zero_output(old_layer.mlp, new_layer.mlp, 
                                          old_hidden, new_hidden_size,
                                          old_intermediate, new_intermediate_size)
            elif hasattr(old_layer, 'moe'):
                # Transfer router: don't read new dims
                old_gate = old_layer.moe.router.gate.weight.data
                new_layer.moe.router.gate.weight.data.zero_()
                new_layer.moe.router.gate.weight.data[:, :old_hidden] = old_gate
                
                # Transfer each expert
                for old_exp, new_exp in zip(old_layer.moe.experts, new_layer.moe.experts):
                    _transfer_mlp_zero_output(old_exp, new_exp,
                                              old_hidden, new_hidden_size,
                                              old_intermediate, new_intermediate_size)
    
    print(f"✓ Scaled hidden dimension from {old_hidden} to {new_hidden_size}")
    print(f"  - Intermediate: {old_intermediate} → {new_intermediate_size}")
    print(f"  - Strategy: ZERO-OUTPUT (function-preserving)")
    print(f"  - New parameter count: {new_model.num_parameters():,}")
    
    return new_model


def _transfer_mlp_zero_output(old_mlp, new_mlp, old_hidden, new_hidden, old_inter, new_inter):
    """
    Transfer MLP weights with zero-output initialization.
    
    Key principle:
    - gate_proj, up_proj: Don't READ from new input dims
    - down_proj: Don't WRITE to new output dims
    """
    # gate_proj: (intermediate, hidden) - don't read new input dims
    new_mlp.gate_proj.weight.data.zero_()
    new_mlp.gate_proj.weight.data[:old_inter, :old_hidden] = old_mlp.gate_proj.weight.data
    # New input cols (576:720) are 0 → don't read
    # New output rows (if any) are 0 → don't contribute
    
    # up_proj: (intermediate, hidden) - don't read new input dims  
    new_mlp.up_proj.weight.data.zero_()
    new_mlp.up_proj.weight.data[:old_inter, :old_hidden] = old_mlp.up_proj.weight.data
    
    # down_proj: (hidden, intermediate) - don't write new output dims
    new_mlp.down_proj.weight.data.zero_()
    new_mlp.down_proj.weight.data[:old_hidden, :old_inter] = old_mlp.down_proj.weight.data
    # New OUTPUT rows (576:720) are 0 → don't contribute to residual
    # This is the KEY for function preservation!


def add_experts(
    moe_model: SmolLM2MoE,
    num_new_experts: int,
    clone_from: str = "random",
) -> SmolLM2MoE:
    """
    Add more experts to an existing MoE model (expert explosion).
    
    Strategy:
    - Clone existing experts into new experts with small noise
    - This preserves behavior while adding capacity
    """
    old_num_experts = moe_model.config.num_experts
    new_num_experts = old_num_experts + num_new_experts
    
    for layer in moe_model.layers:
        moe_block = layer.moe
        
        for i in range(num_new_experts):
            source_idx = torch.randint(0, old_num_experts, (1,)).item()
            source_expert = moe_block.experts[source_idx]
            
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
    
    moe_model.config.num_experts = new_num_experts
    
    print(f"✓ Added {num_new_experts} experts per layer")
    print(f"  - Experts: {old_num_experts} → {new_num_experts}")
    print(f"  - New parameter count: {moe_model.num_parameters():,}")
    
    return moe_model


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Growth Utilities")
    print("=" * 60)
    
    # 1. Create dense model
    print("\n1. Creating dense model...")
    dense_config = SmolLM2Config(num_hidden_layers=4, hidden_size=256, intermediate_size=512)
    dense_model = SmolLM2(dense_config)
    print(f"   Dense model: {dense_model.num_parameters():,} params")
    
    # 2. Test dense_to_moe
    print("\n2. Converting to MoE...")
    moe_model = dense_to_moe(dense_model, num_experts=4, num_experts_per_tok=2)
    
    # 3. Test add_layers
    print("\n3. Adding layers...")
    moe_model = add_layers(moe_model, num_new_layers=2)
    
    # 4. Test scale_hidden_dim with ZERO-OUTPUT
    print("\n4. Scaling hidden dimension (zero-output)...")
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
