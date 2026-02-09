"""
Parameter Estimation Tool (Config-Only)
=========================================

Estimates total and active parameters from config WITHOUT loading the model.
Useful for large models like 70B that can't be loaded locally.

Usage:
    python estimate_params.py --config 70b_moe
    python estimate_params.py --config 1b_dense
    python estimate_params.py --all
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from configs import get_config, CONFIGS
from model.config import ModelType


def estimate_parameters(config) -> Dict[str, float]:
    """
    Estimate parameters from config without loading model.
    
    Returns dict with:
        - total_params: Total model parameters
        - active_params: Active parameters per forward pass
        - components: Breakdown by component
    """
    d = config.hidden_size
    L = config.num_layers
    V = config.tokenizer.vocab_size
    
    # Fine-grained segmentation
    fg = config.expert.fine_grained_factor
    intermediate_eff = config.expert.intermediate_size // fg
    
    # Expert counts
    num_routed = config.num_routed_experts * fg  # Effective routed
    num_shared = config.num_shared_experts
    
    # Router config
    rho = config.router.data_sparsity
    eff_top_k = config.router.top_k * fg
    e_k_real = eff_top_k * rho  # Expected active real experts
    
    # Attention config
    num_heads = config.attention.num_attention_heads
    num_kv_heads = config.attention.num_kv_heads
    head_dim = min(config.attention.head_dim, d // num_heads)  # Handle override
    
    components = {}
    
    # ===== Embeddings =====
    embed_params = V * d
    components['embeddings'] = embed_params
    
    # ===== LM Head =====
    lm_head_params = V * d  # Usually tied, but counted separately
    components['lm_head'] = lm_head_params
    
    # ===== Final Norm =====
    final_norm = d
    components['final_norm'] = final_norm
    
    # ===== Per Layer Components =====
    
    # Attention (GQA base)
    q_proj = d * (num_heads * head_dim)
    k_proj = d * (num_kv_heads * head_dim)
    v_proj = d * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * d
    
    # Base GQA attention
    attention_per_layer = q_proj + k_proj + v_proj + o_proj
    
    # Add GSA components only if attention_type is "gsa"
    if config.attention.attention_type == "gsa":
        # GSA gating (v_gate, o_gate)
        kv_dim = num_kv_heads * head_dim
        q_dim = num_heads * head_dim
        v_gate = d * kv_dim + kv_dim  # projection + bias
        o_gate = d * q_dim + q_dim    # projection + bias
        
        # GSA indexer
        indexer_dim = config.attention.gsa_indexer_dim
        indexer_heads = config.attention.gsa_indexer_heads
        indexer_q = d * (indexer_heads * indexer_dim)
        indexer_k = d * (indexer_heads * indexer_dim)
        indexer_head_weight = d * indexer_heads + indexer_heads
        
        attention_per_layer += (
            v_gate + o_gate +
            indexer_q + indexer_k + indexer_head_weight
        )
    
    # MLA (Multi-head Latent Attention) - DeepSeek V2/V3 Style
    # Reference: DeepSeek-V2 Technical Report
    elif config.attention.attention_type == "mla":
        # MLA parameters (replaces standard Q/K/V/O projections)
        c_kv = config.attention.kv_lora_rank       # KV compressed dim (e.g., 512)
        c_q = config.attention.q_lora_rank         # Q compressed dim (e.g., 1536)
        d_h_r = config.attention.qk_rope_head_dim  # RoPE head dim (e.g., 64)
        d_h_c = config.attention.qk_nope_head_dim  # Non-RoPE head dim (e.g., 128)
        d_h_v = config.attention.v_head_dim        # Value head dim (e.g., 128)
        
        # Down projections (compress hidden → latent)
        W_DKV = d * c_kv                           # KV down-projection
        W_DQ = d * c_q if c_q > 0 else 0           # Q down-projection (optional)
        
        # Up projections (decompress latent → full)
        # K: latent → (nope + rope) dims per head
        W_UK = c_kv * (num_heads * d_h_c)          # K non-RoPE up-projection
        W_KR = d * (num_heads * d_h_r)             # K RoPE (from hidden, not latent)
        
        # V: latent → value dims
        W_UV = c_kv * (num_heads * d_h_v)          # V up-projection
        
        # Q: latent → (nope + rope) dims per head
        if c_q > 0:
            W_UQ = c_q * (num_heads * (d_h_c + d_h_r))  # Q up-projection (nope + rope)
        else:
            # No Q compression - standard Q projection
            W_UQ = d * (num_heads * (d_h_c + d_h_r))
        
        # Output projection
        W_O = (num_heads * d_h_v) * d
        
        # Total MLA attention parameters
        attention_per_layer = W_DKV + W_DQ + W_UK + W_KR + W_UV + W_UQ + W_O
        
        # Store MLA breakdown for reporting
        components['mla_W_DKV'] = W_DKV
        components['mla_W_DQ'] = W_DQ
        components['mla_W_UK'] = W_UK
        components['mla_W_KR'] = W_KR
        components['mla_W_UV'] = W_UV
        components['mla_W_UQ'] = W_UQ
        components['mla_W_O'] = W_O
    
    # Linear Attention (Gated DeltaNet) - Qwen3 Next Style
    # Uses recurrent state instead of KV cache
    elif config.attention.attention_type == "linear":
        # Linear attention parameters
        linear_k_heads = config.attention.linear_num_key_heads
        linear_v_heads = config.attention.linear_num_value_heads
        linear_k_dim = config.attention.linear_key_head_dim
        linear_v_dim = config.attention.linear_value_head_dim
        conv_kernel = config.attention.linear_conv_kernel_dim
        
        key_dim = linear_k_heads * linear_k_dim
        value_dim = linear_v_heads * linear_v_dim
        
        # Projections (Qwen3 Next style)
        # QKVZ projection: Q, K, V, and Z (gate) all projected together
        qkvz_proj = d * (key_dim * 2 + value_dim * 2)  # Q, K, V, Z
        # BA projection: beta and alpha for gating
        ba_proj = d * (linear_v_heads * 2)
        # Causal conv1d for temporal modeling
        conv_proj = (key_dim * 2 + value_dim) * conv_kernel
        # Output projection
        out_proj = value_dim * d
        # Norms (gated RMSNorm)
        linear_norms = linear_v_dim + linear_v_heads  # weight + dt_bias + A_log
        
        attention_per_layer = qkvz_proj + ba_proj + conv_proj + out_proj + linear_norms
        
        # Store Linear breakdown for reporting
        components['linear_qkvz_proj'] = qkvz_proj
        components['linear_ba_proj'] = ba_proj
        components['linear_conv_proj'] = conv_proj
        components['linear_out_proj'] = out_proj
        components['linear_key_dim'] = key_dim
        components['linear_value_dim'] = value_dim
    
    # Hybrid Attention (Qwen3 Next Style) - alternating full and linear layers
    elif config.attention.attention_type == "hybrid":
        # Compute full attention params (GQA style)
        q_proj_full = d * (num_heads * head_dim * 2)  # 2x for gating like Qwen3
        k_proj_full = d * (num_kv_heads * head_dim)
        v_proj_full = d * (num_kv_heads * head_dim)
        o_proj_full = (num_heads * head_dim) * d
        qk_norms = 2 * head_dim  # q_norm + k_norm
        full_attention_params = q_proj_full + k_proj_full + v_proj_full + o_proj_full + qk_norms
        
        # Compute linear attention params
        linear_k_heads = config.attention.linear_num_key_heads
        linear_v_heads = config.attention.linear_num_value_heads
        linear_k_dim = config.attention.linear_key_head_dim
        linear_v_dim = config.attention.linear_value_head_dim
        conv_kernel = config.attention.linear_conv_kernel_dim
        
        key_dim = linear_k_heads * linear_k_dim
        value_dim = linear_v_heads * linear_v_dim
        
        qkvz_proj = d * (key_dim * 2 + value_dim * 2)
        ba_proj = d * (linear_v_heads * 2)
        conv_proj = (key_dim * 2 + value_dim) * conv_kernel
        out_proj = value_dim * d
        linear_norms = linear_v_dim + linear_v_heads
        linear_attention_params = qkvz_proj + ba_proj + conv_proj + out_proj + linear_norms
        
        # Count layer types
        layer_types = config.attention.layer_types
        if layer_types is None:
            # Generate default pattern
            interval = config.attention.full_attention_interval
            layer_types = ["full" if (i + 1) % interval == 0 else "linear" for i in range(L)]
        
        num_full_layers = sum(1 for t in layer_types if t == "full")
        num_linear_layers = L - num_full_layers
        
        # Weighted average per layer
        attention_per_layer = (
            num_full_layers * full_attention_params + 
            num_linear_layers * linear_attention_params
        ) / L
        
        # Store hybrid breakdown
        components['hybrid_full_attention_params'] = full_attention_params
        components['hybrid_linear_attention_params'] = linear_attention_params
        components['hybrid_num_full_layers'] = num_full_layers
        components['hybrid_num_linear_layers'] = num_linear_layers
        components['linear_key_dim'] = key_dim
        components['linear_value_dim'] = value_dim
    
    # GSA-MLA (Gated Sparse Latent Attention) - combines MLA compression with GSA sparsity
    elif config.attention.attention_type == "gsa_mla":
        # MLA parameters (same as pure MLA)
        c_kv = config.attention.kv_lora_rank
        c_q = config.attention.q_lora_rank
        d_h_r = config.attention.qk_rope_head_dim
        d_h_c = config.attention.qk_nope_head_dim
        d_h_v = config.attention.v_head_dim
        
        # MLA projections
        W_DKV = d * c_kv
        W_DQ = d * c_q if c_q > 0 else 0
        W_UK = c_kv * (num_heads * d_h_c)
        W_KR = d * (num_heads * d_h_r)
        W_UV = c_kv * (num_heads * d_h_v)
        if c_q > 0:
            W_UQ = c_q * (num_heads * (d_h_c + d_h_r))
        else:
            W_UQ = d * (num_heads * (d_h_c + d_h_r))
        W_O = (num_heads * d_h_v) * d
        
        mla_params = W_DKV + W_DQ + W_UK + W_KR + W_UV + W_UQ + W_O
        
        # GSA gate params (value gate + output gate)
        v_gate = d * (num_heads * d_h_v) + (num_heads * d_h_v)  # weight + bias
        o_gate = d * (num_heads * d_h_v) + (num_heads * d_h_v)  # weight + bias
        
        # GSA indexer params
        indexer_dim = config.attention.gsa_indexer_dim
        indexer_heads = config.attention.gsa_indexer_heads
        indexer_q = d * (indexer_heads * indexer_dim)
        indexer_k = d * (indexer_heads * indexer_dim)
        indexer_head_weight = d * indexer_heads + indexer_heads
        indexer_bias = indexer_heads
        
        gsa_params = v_gate + o_gate + indexer_q + indexer_k + indexer_head_weight + indexer_bias
        
        attention_per_layer = mla_params + gsa_params
        
        # Store breakdown
        components['gsa_mla_mla_params'] = mla_params
        components['gsa_mla_gsa_params'] = gsa_params
        components['gsa_mla_v_gate'] = v_gate
        components['gsa_mla_o_gate'] = o_gate
        components['gsa_mla_indexer'] = indexer_q + indexer_k + indexer_head_weight + indexer_bias
    
    # GSA-MLA Hybrid: alternates between GSA-MLA (full) and DeltaNet (linear)
    elif config.attention.attention_type == "gsa_mla_hybrid":
        # GSA-MLA params for full layers
        c_kv = config.attention.kv_lora_rank
        c_q = config.attention.q_lora_rank
        d_h_r = config.attention.qk_rope_head_dim
        d_h_c = config.attention.qk_nope_head_dim
        d_h_v = config.attention.v_head_dim
        
        W_DKV = d * c_kv
        W_DQ = d * c_q if c_q > 0 else 0
        W_UK = c_kv * (num_heads * d_h_c)
        W_KR = d * (num_heads * d_h_r)
        W_UV = c_kv * (num_heads * d_h_v)
        if c_q > 0:
            W_UQ = c_q * (num_heads * (d_h_c + d_h_r))
        else:
            W_UQ = d * (num_heads * (d_h_c + d_h_r))
        W_O = (num_heads * d_h_v) * d
        mla_params = W_DKV + W_DQ + W_UK + W_KR + W_UV + W_UQ + W_O
        
        # GSA components
        v_gate = d * (num_heads * d_h_v) + (num_heads * d_h_v)
        o_gate = d * (num_heads * d_h_v) + (num_heads * d_h_v)
        indexer_dim = config.attention.gsa_indexer_dim
        indexer_heads = config.attention.gsa_indexer_heads
        indexer_q = d * (indexer_heads * indexer_dim)
        indexer_k = d * (indexer_heads * indexer_dim)
        indexer_head_weight = d * indexer_heads + indexer_heads
        indexer_bias = indexer_heads
        gsa_params = v_gate + o_gate + indexer_q + indexer_k + indexer_head_weight + indexer_bias
        
        gsa_mla_params = mla_params + gsa_params
        
        # Linear attention params (same as linear type)
        linear_k_heads = config.attention.linear_num_key_heads
        linear_v_heads = config.attention.linear_num_value_heads
        linear_k_dim = config.attention.linear_key_head_dim
        linear_v_dim = config.attention.linear_value_head_dim
        conv_kernel = config.attention.linear_conv_kernel_dim
        
        key_dim = linear_k_heads * linear_k_dim
        value_dim = linear_v_heads * linear_v_dim
        
        qkvz_proj = d * (key_dim * 2 + value_dim * 2)
        ba_proj = d * (linear_v_heads * 2)
        conv_proj = (key_dim * 2 + value_dim) * conv_kernel
        out_proj = value_dim * d
        linear_norms = linear_v_dim + linear_v_heads
        linear_attention_params = qkvz_proj + ba_proj + conv_proj + out_proj + linear_norms
        
        # Count layer types
        layer_types = config.attention.layer_types
        if layer_types is None:
            interval = config.attention.full_attention_interval
            layer_types = ["full" if (i + 1) % interval == 0 else "linear" for i in range(L)]
        
        num_full_layers = sum(1 for t in layer_types if t == "full")
        num_linear_layers = L - num_full_layers
        
        # Weighted average
        attention_per_layer = (
            num_full_layers * gsa_mla_params +
            num_linear_layers * linear_attention_params
        ) / L
        
        # Store breakdown
        components['gsa_mla_hybrid_gsa_mla_params'] = gsa_mla_params
        components['gsa_mla_hybrid_linear_params'] = linear_attention_params
        components['gsa_mla_hybrid_num_full_layers'] = num_full_layers
        components['gsa_mla_hybrid_num_linear_layers'] = num_linear_layers
        components['linear_key_dim'] = key_dim
        components['linear_value_dim'] = value_dim
    
    # GSA Hybrid: alternates between GSA (full) and DeltaNet (linear)
    elif config.attention.attention_type == "gsa_hybrid":
        # GSA params for full layers (same as pure GSA)
        kv_dim = num_kv_heads * head_dim
        q_dim = num_heads * head_dim
        
        # Base GQA projections
        q_proj = d * q_dim
        k_proj = d * kv_dim
        v_proj = d * kv_dim
        o_proj = q_dim * d
        
        # GSA gates
        v_gate = d * kv_dim + kv_dim
        o_gate = d * q_dim + q_dim
        
        # GSA indexer
        indexer_dim = config.attention.gsa_indexer_dim
        indexer_heads = config.attention.gsa_indexer_heads
        indexer_q = d * (indexer_heads * indexer_dim)
        indexer_k = d * (indexer_heads * indexer_dim)
        indexer_head_weight = d * indexer_heads + indexer_heads
        
        gsa_params = (q_proj + k_proj + v_proj + o_proj + 
                      v_gate + o_gate + 
                      indexer_q + indexer_k + indexer_head_weight)
        
        # Linear attention params (same as linear type)
        linear_k_heads = config.attention.linear_num_key_heads
        linear_v_heads = config.attention.linear_num_value_heads
        linear_k_dim = config.attention.linear_key_head_dim
        linear_v_dim = config.attention.linear_value_head_dim
        conv_kernel = config.attention.linear_conv_kernel_dim
        
        key_dim = linear_k_heads * linear_k_dim
        value_dim = linear_v_heads * linear_v_dim
        
        qkvz_proj = d * (key_dim * 2 + value_dim * 2)
        ba_proj = d * (linear_v_heads * 2)
        conv_proj = (key_dim * 2 + value_dim) * conv_kernel
        out_proj = value_dim * d
        linear_norms = linear_v_dim + linear_v_heads
        linear_attention_params = qkvz_proj + ba_proj + conv_proj + out_proj + linear_norms
        
        # Count layer types
        layer_types = config.attention.layer_types
        if layer_types is None:
            interval = config.attention.full_attention_interval
            layer_types = ["full" if (i + 1) % interval == 0 else "linear" for i in range(L)]
        
        num_full_layers = sum(1 for t in layer_types if t == "full")
        num_linear_layers = L - num_full_layers
        
        # Weighted average
        attention_per_layer = (
            num_full_layers * gsa_params +
            num_linear_layers * linear_attention_params
        ) / L
        
        # Store breakdown
        components['gsa_hybrid_gsa_params'] = gsa_params
        components['gsa_hybrid_linear_params'] = linear_attention_params
        components['gsa_hybrid_num_full_layers'] = num_full_layers
        components['gsa_hybrid_num_linear_layers'] = num_linear_layers
        components['linear_key_dim'] = key_dim
        components['linear_value_dim'] = value_dim
    
    # LayerNorms (2 per layer: input and post-attention)
    norms_per_layer = 2 * d
    
    # Check if this is a Dense or MoE model
    is_moe = config.model_type == ModelType.MOE
    
    if is_moe:
        # ===== MoE Model: Expert FFN =====
        # Expert FFN (SwiGLU: w1, w2, w3)
        # Each expert: 3 × hidden × intermediate_effective
        params_per_expert = 3 * d * intermediate_eff
        
        # Routed experts total
        routed_experts_per_layer = num_routed * params_per_expert
        
        # Shared experts (ALSO use fine-grained intermediate, same as routed)
        # See expert.py line 360: intermediate_size=effective_intermediate
        params_per_shared = 3 * d * intermediate_eff  # Same as routed!
        shared_experts_per_layer = num_shared * params_per_shared
        
        # Router (routes over effective experts + null)
        # Linear layer: hidden -> (num_routed + num_null) with bias
        num_null = config.num_null_experts
        router_params = d * (num_routed + num_null) + (num_routed + num_null)
        
        # Total per layer (MoE)
        per_layer_total = (
            attention_per_layer +
            norms_per_layer +
            routed_experts_per_layer +
            shared_experts_per_layer +
            router_params
        )
        
        components['routed_experts_per_layer'] = routed_experts_per_layer
        components['shared_experts_per_layer'] = shared_experts_per_layer
        components['router_per_layer'] = router_params
        
        # ===== Total Parameters (MoE) =====
        total_params = (
            embed_params +
            lm_head_params +
            final_norm +
            L * per_layer_total
        )
        
        # ===== Active Parameters (per forward pass) =====
        # Active routed = E[K_real] = effective_top_k × ρ
        # This is the expected number of REAL experts per token (excluding null)
        # With ρ=0.5: half of the selected slots go to null experts
        e_k_real = eff_top_k * rho  # e.g., 16 × 0.5 = 8
        active_routed = e_k_real * params_per_expert
        active_shared = num_shared * params_per_shared
        
        active_per_layer = (
            attention_per_layer +
            norms_per_layer +
            active_routed +
            active_shared +
            router_params
        )
        
        active_params = (
            embed_params +
            lm_head_params +
            final_norm +
            L * active_per_layer
        )
        
        components['active_routed_per_layer'] = active_routed
        components['active_per_layer'] = active_per_layer
        
        # ===== Null Expert Metrics =====
        null_copies = int(round(num_routed * (1 - rho) / rho)) if rho < 1.0 else 0
        components['effective_experts'] = num_routed
        components['null_copies'] = null_copies
        components['e_k_real'] = e_k_real
        components['total_active_experts'] = e_k_real + num_shared
        
    else:
        # ===== Dense Model: Standard FFN =====
        # Dense FFN uses full intermediate size (no segmentation)
        dense_ffn_params = 3 * d * config.expert.intermediate_size
        components['ffn_per_layer'] = dense_ffn_params
        
        # Total per layer (Dense)
        per_layer_total = (
            attention_per_layer +
            norms_per_layer +
            dense_ffn_params
        )
        
        # No router, no experts
        components['routed_experts_per_layer'] = 0
        components['shared_experts_per_layer'] = 0
        components['router_per_layer'] = 0
        
        # ===== Total Parameters (Dense) =====
        total_params = (
            embed_params +
            lm_head_params +
            final_norm +
            L * per_layer_total
        )
        
        # For dense models, all params are active
        active_params = total_params
        components['active_per_layer'] = per_layer_total
        
        # No MoE-specific metrics for dense models
        components['effective_experts'] = 0
        components['null_copies'] = 0
        components['e_k_real'] = 0
        components['total_active_experts'] = 0
    
    components['attention_per_layer'] = attention_per_layer
    components['norms_per_layer'] = norms_per_layer
    components['per_layer_total'] = per_layer_total
    components['total'] = total_params
    components['active_total'] = active_params
    
    return {
        'total_params': total_params,
        'active_params': active_params,
        'ratio': active_params / total_params if total_params > 0 else 0,
        'components': components
    }


def format_params(n: float) -> str:
    """Format parameter count."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    elif n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(int(n))


def print_estimate(config, name: str = "Config"):
    """Print detailed parameter estimate."""
    est = estimate_parameters(config)
    c = est['components']
    
    print(f"\n{'=' * 70}")
    print(f"PARAMETER ESTIMATE: {name.upper()}")
    print(f"{'=' * 70}")
    
    # Config summary
    print(f"\n📋 Configuration:")
    print(f"   Model Type:        {config.model_type.value.upper()}")
    print(f"   Hidden Size:       {config.hidden_size}")
    print(f"   Num Layers:        {config.num_layers}")
    print(f"   Vocab Size:        {config.tokenizer.vocab_size:,}")
    print(f"   Attention Type:    {config.attention.attention_type.upper()}")
    
    is_moe = config.model_type == ModelType.MOE
    
    if is_moe:
        # MoE config
        fg = config.expert.fine_grained_factor
        print(f"   Fine-Grained:      {fg}×")
        
        print(f"\n🔀 MoE Configuration:")
        print(f"   Base Routed:       {config.num_routed_experts}")
        print(f"   Effective (N):     {c['effective_experts']}")
        print(f"   Null Copies (M):   {c['null_copies']}")
        print(f"   Shared Experts:    {config.num_shared_experts}")
        print(f"   ρ (data sparsity): {config.router.data_sparsity}")
        print(f"   E[K_real]:         {c['e_k_real']:.1f}")
        print(f"   Total Active/tok:  {c['total_active_experts']:.1f}")
    else:
        # Dense config
        print(f"   Intermediate Size: {config.expert.intermediate_size}")
    
    # Parameter counts
    print(f"\n📊 Parameter Counts:")
    print(f"   {'Component':<30} {'Params':>15}")
    print(f"   {'-' * 45}")
    print(f"   {'Embeddings':<30} {format_params(c['embeddings']):>15}")
    print(f"   {'LM Head':<30} {format_params(c['lm_head']):>15}")
    print(f"   {'Final Norm':<30} {format_params(c['final_norm']):>15}")
    print(f"   ")
    print(f"   Per Layer:")
    print(f"   {'  Attention':<30} {format_params(c['attention_per_layer']):>15}")
    
    # Show MLA breakdown if using MLA
    if config.attention.attention_type == "mla":
        print(f"      MLA Breakdown:")
        print(f"      {'    W_DKV (KV downproj)':<28} {format_params(c.get('mla_W_DKV', 0)):>13}")
        print(f"      {'    W_DQ (Q downproj)':<28} {format_params(c.get('mla_W_DQ', 0)):>13}")
        print(f"      {'    W_UK (K upproj)':<28} {format_params(c.get('mla_W_UK', 0)):>13}")
        print(f"      {'    W_KR (K RoPE)':<28} {format_params(c.get('mla_W_KR', 0)):>13}")
        print(f"      {'    W_UV (V upproj)':<28} {format_params(c.get('mla_W_UV', 0)):>13}")
        print(f"      {'    W_UQ (Q upproj)':<28} {format_params(c.get('mla_W_UQ', 0)):>13}")
        print(f"      {'    W_O (output)':<28} {format_params(c.get('mla_W_O', 0)):>13}")
    
    # Show Linear attention breakdown
    elif config.attention.attention_type == "linear":
        print(f"      Linear (Gated DeltaNet) Breakdown:")
        print(f"      {'    key_dim':<28} {c.get('linear_key_dim', 0):>13}")
        print(f"      {'    value_dim':<28} {c.get('linear_value_dim', 0):>13}")
        print(f"      {'    QKVZ proj':<28} {format_params(c.get('linear_qkvz_proj', 0)):>13}")
        print(f"      {'    BA proj':<28} {format_params(c.get('linear_ba_proj', 0)):>13}")
        print(f"      {'    Conv1d':<28} {format_params(c.get('linear_conv_proj', 0)):>13}")
        print(f"      {'    Out proj':<28} {format_params(c.get('linear_out_proj', 0)):>13}")
    
    # Show Hybrid attention breakdown
    elif config.attention.attention_type == "hybrid":
        num_full = c.get('hybrid_num_full_layers', 0)
        num_linear = c.get('hybrid_num_linear_layers', 0)
        print(f"      Hybrid Breakdown ({num_linear} linear : {num_full} full):")
        print(f"      {'    Full attn/layer':<28} {format_params(c.get('hybrid_full_attention_params', 0)):>13}")
        print(f"      {'    Linear attn/layer':<28} {format_params(c.get('hybrid_linear_attention_params', 0)):>13}")
        print(f"      {'    key_dim':<28} {c.get('linear_key_dim', 0):>13}")
        print(f"      {'    value_dim':<28} {c.get('linear_value_dim', 0):>13}")
    
    # Show GSA-MLA breakdown
    elif config.attention.attention_type == "gsa_mla":
        print(f"      GSA-MLA Breakdown:")
        print(f"      {'    MLA params':<28} {format_params(c.get('gsa_mla_mla_params', 0)):>13}")
        print(f"      {'    GSA params':<28} {format_params(c.get('gsa_mla_gsa_params', 0)):>13}")
        print(f"      {'      V gate':<28} {format_params(c.get('gsa_mla_v_gate', 0)):>13}")
        print(f"      {'      O gate':<28} {format_params(c.get('gsa_mla_o_gate', 0)):>13}")
        print(f"      {'      Indexer':<28} {format_params(c.get('gsa_mla_indexer', 0)):>13}")
    
    # Show GSA-MLA Hybrid breakdown
    elif config.attention.attention_type == "gsa_mla_hybrid":
        num_full = c.get('gsa_mla_hybrid_num_full_layers', 0)
        num_linear = c.get('gsa_mla_hybrid_num_linear_layers', 0)
        print(f"      GSA-MLA Hybrid ({num_linear} linear : {num_full} GSA-MLA):")
        print(f"      {'    GSA-MLA attn/layer':<28} {format_params(c.get('gsa_mla_hybrid_gsa_mla_params', 0)):>13}")
        print(f"      {'    Linear attn/layer':<28} {format_params(c.get('gsa_mla_hybrid_linear_params', 0)):>13}")
        print(f"      {'    key_dim':<28} {c.get('linear_key_dim', 0):>13}")
        print(f"      {'    value_dim':<28} {c.get('linear_value_dim', 0):>13}")
    
    # Show GSA Hybrid breakdown
    elif config.attention.attention_type == "gsa_hybrid":
        num_full = c.get('gsa_hybrid_num_full_layers', 0)
        num_linear = c.get('gsa_hybrid_num_linear_layers', 0)
        print(f"      GSA Hybrid ({num_linear} linear : {num_full} GSA):")
        print(f"      {'    GSA attn/layer':<28} {format_params(c.get('gsa_hybrid_gsa_params', 0)):>13}")
        print(f"      {'    Linear attn/layer':<28} {format_params(c.get('gsa_hybrid_linear_params', 0)):>13}")
        print(f"      {'    key_dim':<28} {c.get('linear_key_dim', 0):>13}")
        print(f"      {'    value_dim':<28} {c.get('linear_value_dim', 0):>13}")
    
    print(f"   {'  Norms':<30} {format_params(c['norms_per_layer']):>15}")
    
    if is_moe:
        print(f"   {'  Routed Experts':<30} {format_params(c['routed_experts_per_layer']):>15}")
        print(f"   {'  Shared Experts':<30} {format_params(c['shared_experts_per_layer']):>15}")
        print(f"   {'  Router':<30} {format_params(c['router_per_layer']):>15}")
    else:
        print(f"   {'  FFN':<30} {format_params(c.get('ffn_per_layer', 0)):>15}")
    
    print(f"   {'  ─────────────────':<30}")
    print(f"   {'  Layer Total':<30} {format_params(c['per_layer_total']):>15}")
    print(f"   × {config.num_layers} layers")
    
    print(f"\n⚡ Summary:")
    print(f"   {'TOTAL PARAMETERS:':<25} {format_params(est['total_params']):>20}")
    print(f"   {'ACTIVE PARAMETERS:':<25} {format_params(est['active_params']):>20}")
    print(f"   {'ACTIVATION RATIO:':<25} {est['ratio']*100:>19.1f}%")
    
    print(f"\n💾 Memory Estimates:")
    print(f"   Total (FP16):      {est['total_params'] * 2 / 1e9:.2f} GB")
    print(f"   Active (FP16):     {est['active_params'] * 2 / 1e9:.2f} GB")
    
    return est


def main():
    parser = argparse.ArgumentParser(description="Estimate parameters from config without loading model")
    parser.add_argument(
        "--config", "-c",
        type=str,
        choices=list(CONFIGS.keys()),
        help="Config name (e.g., 1b_dense, 3b_moe, 8b_moe, 70b_moe)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Estimate all available configs"
    )
    args = parser.parse_args()
    
    configs_to_load = []
    if args.all:
        configs_to_load = list(CONFIGS.keys())  # All available configs
    elif args.config:
        configs_to_load = [args.config]
    else:
        configs_to_load = ["70b_moe"]  # Default to 70B
    
    # Load and estimate
    results = {}
    for cfg_name in configs_to_load:
        try:
            config = get_config(cfg_name)
            est = print_estimate(config, cfg_name)
            results[cfg_name] = est
        except Exception as e:
            print(f"Error loading {cfg_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison table
    if len(results) > 1:
        print(f"\n\n{'=' * 85}")
        print("COMPARISON TABLE")
        print(f"{'=' * 85}")
        print(f"{'Config':<12} {'Total':>12} {'Active':>12} {'Ratio':>10} {'Eff N':>10} {'Null M':>10}")
        print(f"{'-' * 85}")
        for name, est in results.items():
            c = est['components']
            eff_n = c.get('effective_experts', 0)
            null_m = c.get('null_copies', 0)
            print(f"{name:<12} {format_params(est['total_params']):>12} {format_params(est['active_params']):>12} {est['ratio']*100:>9.1f}% {eff_n:>10} {null_m:>10}")
        print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
