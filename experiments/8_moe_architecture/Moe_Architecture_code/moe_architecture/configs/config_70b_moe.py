"""
Stage 4: 70B MoE Model Configuration
====================================

Expert explosion stage: SAME model structure as 8B, only explode experts.
This massively increases capacity while preserving compute efficiency.

Architecture (DeepSeek-faithful + Null Experts paper):
- SAME hidden_size, num_layers as 8B (structure preserved)
- Expand experts: 26 → 280 effective (70 base × 4 fine-grained)
- Reduce shared experts: 2 → 1 (paper: decays at scale)

Key Transition (8B → 70B):
- Same intermediate_size (4096), same num_layers (32)
- Expert explosion: 26 base → 70 base (×4 = 280 effective)
- Same ρ=0.5, E[K_real]=8
- Null copies scaled: M = 280 (from formula)

Paper Parameters:
- Segments (m) = 4
- Real routed experts (N) = 256
- Shared experts (Ks) = 1
- k_max = 16 (effective), ρ = 0.5
- E[K_real] = 8, Total active ≈ 9

Active Parameter Target: ~2.4B
"""

from model.config import (
    MoEModelConfig,
    ModelType,
    RouterType,
    RouterConfig,
    ExpertConfig,
    AttentionConfig,
    TokenizerConfig,
    ComputeBudget,
    TelemetryConfig,
)


def get_config() -> MoEModelConfig:
    """Get 70B MoE model configuration (expert explosion from 8B)."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_70b_moe256",
        model_type=ModelType.MOE,
        stage=4,
        
        # Core dimensions (SAME as 8B! - structure preserved)
        hidden_size=2560,                # SAME as 8B
        num_layers=32,                   # SAME as 8B 
        
        # MoE Configuration (EXPLODED experts for 70B)
        # Base 70 experts × 4 fine-grained factor = 280 effective routed experts
        # Total params: 280 experts × 3 × 6144 × 64 × 32 layers ≈ 70B
        num_routed_experts=70,          # EXPLODED Total:280
        num_shared_experts=1,            # REDUCED (was 2) - paper: decays at scale
        num_null_experts=1,              # Single null (M=N=280 copies in router)
        moe_layer_frequency=1,           # MoE on ALL layers
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=49152,            # Standardized
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
            junk_token_ids=[0],
            special_token_range=(0, 100),
            punctuation_range=(100, 200),
            common_word_range=(300, 1000),
        ),
        
        # Null Expert Router Configuration (arXiv:2601.15370v1)
        # Paper formula: M = N × (1-ρ)/ρ, E[K_real] = k_max × ρ
        # With N=280, ρ=0.5, k_max=16: M=280 null copies, E[K_real]=16
        router=RouterConfig(
            router_type=RouterType.NULL_EXPERT,
            top_k=4,                     # Same k_max base (×4 = 16 effective)
            data_sparsity=0.5,           # ρ = 0.5 (paper stable region)
            null_copies=0,               # Auto-derive: M = 280 × (1-0.5)/0.5 = 280
            use_aux_loss=True,
            aux_loss_weight=0.02,
            router_z_loss_weight=0.001,
        ),
        
        # Expert Configuration (fine-grained, sized for 2.4B active target)
        # Active = 9 experts/token × 3 × 6144 × 64 × 28 ≈ 2.4B
        # Total = 280 experts × 3 × 6144 × 64 × 28 ≈ 70B  
        expert=ExpertConfig(
            intermediate_size=4096,       
            fine_grained_factor=4,       # DeepSeek-MoE style
            use_dual_gating=False,
            gate_bias_init=0.0,
            expert_init_std=0.02,
            noise_std_for_expansion=1e-3, # More noise for expert divergence
        ),
        
        # Attention Configuration (SAME as 8B)
        attention=AttentionConfig(
            attention_type="gsa",
            num_attention_heads=16,      # Same as 8B
            num_kv_heads=8,
            head_dim=160,
            rope_theta=10000.0,
            attention_dropout=0.0,
            gsa_indexer_dim=64,
            gsa_indexer_heads=4,
            gsa_k_base=2048,
            gsa_k_min=256,
            gsa_k_max=4096,
        ),
        
        # Compute Budget
        compute_budget=ComputeBudget(
            max_params_total=int(75e9),     # 75B ceiling
            max_params_active=int(2.5e9),   # ~2.4B active target
            target_tokens=int(2e12),
            max_sequence_length=4096,
        ),
        
        # Telemetry (stricter for 256 experts)
        telemetry=TelemetryConfig(
            log_every_n_steps=100,
            dead_expert_threshold=0.005,    # Stricter (<0.5% = dead)
            overload_expert_threshold=2.5,
            min_router_entropy=0.75,        # Higher entropy needed
            max_gini_coefficient=0.4,       # Stricter balance
            junk_null_rate_alert_low=0.5,
            junk_null_rate_alert_high=0.9,
            signal_null_rate_alert=0.15,
            enable_auto_correction=True,
            correction_strength=0.05,       # Gentler at scale
        ),
        
        # Training
        max_position_embeddings=4096,
        hidden_dropout=0.0,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        torch_dtype="bfloat16",
    )


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    print(f"\nExpert Explosion from 8B:")
    print(f"  Hidden: 5120 → {config.hidden_size} (SAME!)")
    print(f"  Layers: 24 → {config.num_layers} (SAME!)")
    print(f"  Experts: 32 → {config.effective_num_routed_experts} (8× explosion)")
    print(f"  Null copies: 32 → M derived from N=256")
    print(f"  E[K_real]: 8 (preserved)")
