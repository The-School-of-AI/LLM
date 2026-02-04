"""
Parameter Estimation Tool (Config-Only)
=========================================

Estimates total and active parameters from config WITHOUT loading the model.
Useful for large models like 70B that can't be loaded locally.

Usage:
    python estimate_params.py --config 70b_moe
    python estimate_params.py --all
"""

import argparse
from typing import Dict, Tuple


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
    
    # Attention (GQA)
    q_proj = d * (num_heads * head_dim)
    k_proj = d * (num_kv_heads * head_dim)
    v_proj = d * (num_kv_heads * head_dim)
    o_proj = (num_heads * head_dim) * d
    
    # GSA gating (v_gate, o_gate)
    v_gate = d * (num_kv_heads * head_dim) + (num_kv_heads * head_dim)
    o_gate = d * (num_heads * head_dim) + (num_heads * head_dim)
    
    # GSA indexer
    indexer_dim = config.attention.gsa_indexer_dim
    indexer_heads = config.attention.gsa_indexer_heads
    indexer_q = d * (indexer_heads * indexer_dim)
    indexer_k = d * (indexer_heads * indexer_dim)
    indexer_head_weight = d * indexer_heads + indexer_heads
    
    attention_per_layer = (
        q_proj + k_proj + v_proj + o_proj +
        v_gate + o_gate +
        indexer_q + indexer_k + indexer_head_weight
    )
    
    # LayerNorms (2 per layer: input and post-attention)
    norms_per_layer = 2 * d
    
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
    num_null = 1  # Fixed single null expert
    router_params = d * (num_routed + num_null) + (num_routed + num_null)
    
    # Total per layer
    per_layer_total = (
        attention_per_layer +
        norms_per_layer +
        routed_experts_per_layer +
        shared_experts_per_layer +
        router_params
    )
    
    components['attention_per_layer'] = attention_per_layer
    components['norms_per_layer'] = norms_per_layer
    components['routed_experts_per_layer'] = routed_experts_per_layer
    components['shared_experts_per_layer'] = shared_experts_per_layer
    components['router_per_layer'] = router_params
    components['per_layer_total'] = per_layer_total
    
    # ===== Total Parameters =====
    total_params = (
        embed_params +
        lm_head_params +
        final_norm +
        L * per_layer_total
    )
    components['total'] = total_params
    
    # ===== Active Parameters (per forward pass) =====
    # Active routed = base top_k experts (matches load_model_summary.py line 139)
    # load_model_summary.py uses config.router.top_k, not effective_top_k
    base_top_k = config.router.top_k
    active_routed = base_top_k * params_per_expert
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
    components['active_total'] = active_params
    
    # ===== Null Expert Metrics =====
    null_copies = int(round(num_routed * (1 - rho) / rho)) if rho < 1.0 else 0
    components['effective_experts'] = num_routed
    components['null_copies'] = null_copies
    components['e_k_real'] = e_k_real
    components['total_active_experts'] = e_k_real + num_shared
    
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
    fg = config.expert.fine_grained_factor
    print(f"\n📋 Configuration:")
    print(f"   Hidden Size:       {config.hidden_size}")
    print(f"   Num Layers:        {config.num_layers}")
    print(f"   Vocab Size:        {config.tokenizer.vocab_size:,}")
    print(f"   Fine-Grained:      {fg}×")
    
    # MoE config
    print(f"\n🔀 MoE Configuration:")
    print(f"   Base Routed:       {config.num_routed_experts}")
    print(f"   Effective (N):     {c['effective_experts']}")
    print(f"   Null Copies (M):   {c['null_copies']}")
    print(f"   Shared Experts:    {config.num_shared_experts}")
    print(f"   ρ (data sparsity): {config.router.data_sparsity}")
    print(f"   E[K_real]:         {c['e_k_real']:.1f}")
    print(f"   Total Active/tok:  {c['total_active_experts']:.1f}")
    
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
    print(f"   {'  Norms':<30} {format_params(c['norms_per_layer']):>15}")
    print(f"   {'  Routed Experts':<30} {format_params(c['routed_experts_per_layer']):>15}")
    print(f"   {'  Shared Experts':<30} {format_params(c['shared_experts_per_layer']):>15}")
    print(f"   {'  Router':<30} {format_params(c['router_per_layer']):>15}")
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
    parser.add_argument("--config", type=str, help="Config name (e.g., 3b_moe, 8b_moe, 70b_moe)")
    parser.add_argument("--all", action="store_true", help="Estimate all MoE configs")
    args = parser.parse_args()
    
    configs_to_load = []
    if args.all:
        configs_to_load = ["3b_moe", "8b_moe", "70b_moe"]
    elif args.config:
        configs_to_load = [args.config]
    else:
        configs_to_load = ["70b_moe"]  # Default to 70B
    
    # Load and estimate
    results = {}
    for cfg_name in configs_to_load:
        try:
            if cfg_name == "3b_moe":
                from configs.config_3b_moe import get_config
            elif cfg_name == "8b_moe":
                from configs.config_8b_moe import get_config
            elif cfg_name == "70b_moe":
                from configs.config_70b_moe import get_config
            else:
                print(f"Unknown config: {cfg_name}")
                continue
            
            config = get_config()
            est = print_estimate(config, cfg_name)
            results[cfg_name] = est
        except Exception as e:
            print(f"Error loading {cfg_name}: {e}")
    
    # Comparison table
    if len(results) > 1:
        print(f"\n\n{'=' * 85}")
        print("COMPARISON TABLE")
        print(f"{'=' * 85}")
        print(f"{'Config':<12} {'Total':>12} {'Active':>12} {'Ratio':>10} {'Eff N':>10} {'Null M':>10}")
        print(f"{'-' * 85}")
        for name, est in results.items():
            c = est['components']
            print(f"{name:<12} {format_params(est['total_params']):>12} {format_params(est['active_params']):>12} {est['ratio']*100:>9.1f}% {c['effective_experts']:>10} {c['null_copies']:>10}")
        print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
