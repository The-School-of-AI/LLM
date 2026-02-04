"""
Stage 4: 70B MoE Model Configuration
====================================

Expert explosion stage: SAME model structure as 8B, only explode experts.
This massively increases capacity while preserving compute efficiency.

Architecture (DeepSeek-faithful + Null Experts paper):
- SAME hidden_size, num_layers as 8B (structure preserved)
- Expand experts: 32 → 256 effective (64 base × 4 fine-grained)
- Reduce shared experts: 2 → 1 (paper: decays at scale)

Key Transition (8B → 70B):
- Same hidden_size (5120), same num_layers (24)
- Expert explosion: 8 base → 64 base (×4 = 256 effective)
- Same ρ=0.5, E[K_real]=8
- Null copies scaled: M = 256 (from formula)

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
        hidden_size=3072,                # SAME as 8B
        num_layers=56,                   # SAME as 8B # 24
        
        # MoE Configuration (EXPLODED experts for 70B)
        # Base 512 experts × 4 fine-grained factor = 2048 effective routed experts
        # Total params: 2048 experts × 3 × 6144 × 64 × 28 layers ≈ 70B
        num_routed_experts=256,          # EXPLODED (was 8) - 64× expansion
        num_shared_experts=1,            # REDUCED (was 2) - paper: decays at scale
        num_null_experts=1,              # Single null (M=2048 copies in router)
        moe_layer_frequency=1,           # MoE on ALL layers
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=128000,            # Standardized
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
        # With N=2048, ρ=0.5, k_max=16: M=2048 null copies, E[K_real]=8
        router=RouterConfig(
            router_type=RouterType.NULL_EXPERT,
            top_k=4,                     # Same k_max base (×4 = 16 effective)
            data_sparsity=0.5,           # ρ = 0.5 (paper stable region)
            null_copies=0,               # Auto-derive: M = 2048 × (1-0.5)/0.5 = 2048
            use_aux_loss=True,
            aux_loss_weight=0.02,
            router_z_loss_weight=0.001,
        ),
        
        # Expert Configuration (fine-grained, sized for 2.4B active target)
        # Active = 9 experts/token × 3 × 6144 × 64 × 28 ≈ 2.4B
        # Total = 2048 experts × 3 × 6144 × 64 × 28 ≈ 70B  
        # (8/3)*8192 ~ 22016 -> 22016/64 = 344
        expert=ExpertConfig(
            intermediate_size=512,       # Base size, effective = 64 (small for low active)
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
            num_kv_heads=4,
            head_dim=512,
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


# Configuration instance
CONFIG = get_config()


"""
70B MoE Expert Explosion Strategy:
==================================

Key Insight: Keep model structure, only increase experts.
This preserves compute per token while massively scaling capacity.

From 8B:            To 70B:
- hidden=5120   →   hidden=5120 (SAME!)
- layers=24     →   layers=24 (SAME!)
- experts=32    →   experts=256 (8× expansion)
- shared=2      →   shared=1 (paper: decay at scale)
- ρ=0.5, M=32   →   ρ=0.5, M=256 (scaled with N)
- E[K_real]=8   →   E[K_real]=8 (SAME!)

Parameter Calculation:
- Embeddings: 32K × 5120 = 163M
- Per expert (fine-grained): 3 × 5120 × 320 = 4.9M
- Routed experts per layer: 256 × 4.9M = 1.25B
- Shared expert per layer: 1 × 4.9M (×4 for full) = 19.6M
- Attention per layer: ~80M
- Per layer total: ~1.35B
- 24 layers: 24 × 1.35B = 32.4B
- Plus embeddings, router: ~35B

Note: Adjust intermediate_size to hit ~70B total.
Current config may need tuning.

Active Parameters (~2.4B target):
- E[K_real] = 8 routed + 1 shared = 9 active experts
- Per expert active: 4.9M
- Per layer active: 9 × 4.9M + 80M (attention) = 124M
- Total active: 24 × 124M + embeddings ≈ 3B

Tune intermediate_size down to hit 2.4B active.
"""


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    print(f"\nExpert Explosion from 8B:")
    print(f"  Hidden: 5120 → {config.hidden_size} (SAME!)")
    print(f"  Layers: 24 → {config.num_layers} (SAME!)")
    print(f"  Experts: 32 → {config.effective_num_routed_experts} (8× explosion)")
    print(f"  Null copies: 32 → M derived from N=256")
    print(f"  E[K_real]: 8 (preserved)")
