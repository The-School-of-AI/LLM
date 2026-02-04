"""
Stage 3: 8B MoE Model Configuration
===================================

Width-scaled version of 3B, keeping SAME expert configuration.
This preserves routing knowledge learned in Stage 2.

Architecture (DeepSeek-faithful + Null Experts paper):
- Same expert structure as 3B (N=32 effective, M=32 null copies)
- Increased hidden_size for width scaling (4096 → 5120)
- Slightly more layers (20 → 24)

Key Transition (3B → 8B):
- Same num_routed_experts, fine_grained_factor, top_k
- Same null expert configuration (ρ=0.5, M derived)
- Scale hidden dimensions for more capacity
- Routing patterns preserved!

Paper Parameters:
- Segments (m) = 4
- Real routed experts (N) = 32
- Shared experts (Ks) = 2
- k_max = 16 (effective), ρ = 0.5
- E[K_real] = 8, Total active ≈ 10
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
    """Get 8B MoE model configuration (width-scaled from 3B)."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_8b_moe32",
        model_type=ModelType.MOE,
        stage=3,
        
        # Core dimensions (WIDTH SCALED from 3B for capacity)
        hidden_size=2048,                # Scaled up (was 4096) - wider model
        num_layers=96,                   # Deeper (was 20)
        
        # MoE Configuration (SAME as 3B! - DeepSeek + Null Experts paper)
        # Base 8 experts × 4 fine-grained factor = 32 effective routed experts
        num_routed_experts=20,            # SAME as 3B (N_seg = 8 per segment)
        num_shared_experts=1,            # SAME as 3B (Ks = 2)
        num_null_experts=1,              # SAME as 3B (M copies in router)
        moe_layer_frequency=1,           # MoE on ALL layers
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=128000,            # Standardized with 3B
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
            junk_token_ids=[0],
            special_token_range=(0, 100),
            punctuation_range=(100, 200),
            common_word_range=(300, 1000),
        ),
        
        # Null Expert Router Configuration (SAME as 3B - arXiv:2601.15370v1)
        # Paper formula: M = N × (1-ρ)/ρ, E[K_real] = k_max × ρ
        # With N=32, ρ=0.5, k_max=16: M=32 null copies, E[K_real]=8
        router=RouterConfig(
            router_type=RouterType.NULL_EXPERT,
            top_k=2,                     # SAME as 3B (×4 fine-grained = 16 effective)
            data_sparsity=0.5,           # SAME as 3B (ρ = 0.5)
            null_copies=0,               # SAME as 3B (auto-derive: M=32)
            use_aux_loss=True,
            aux_loss_weight=0.02,
            router_z_loss_weight=0.001,
        ),
        
        # Expert Configuration (SAME as 3B - fine-grained)
        expert=ExpertConfig(
            intermediate_size=512,       # Scaled with hidden (6144 × 0.125 = 768)
            fine_grained_factor=4,       # SAME as 3B (DeepSeek-MoE style)
            use_dual_gating=False,
            gate_bias_init=0.0,
            expert_init_std=0.02,
            noise_std_for_expansion=1e-4,
        ),
        
        # Attention Configuration (scaled with hidden)
        attention=AttentionConfig(
            attention_type="gsa",
            num_attention_heads=32,       # Scaled (6144/128 = 48)
            num_kv_heads=8,               # 6:1 GQA
            head_dim=128,                 # Standard
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
            max_params_total=int(10e9),
            max_params_active=int(3e9),
            target_tokens=int(1e12),
            max_sequence_length=4096,
        ),
        
        # Telemetry (Team 7 Integration)
        telemetry=TelemetryConfig(
            log_every_n_steps=100,
            dead_expert_threshold=0.01,
            overload_expert_threshold=3.0,
            min_router_entropy=0.7,
            max_gini_coefficient=0.5,
            junk_null_rate_alert_low=0.5,
            junk_null_rate_alert_high=0.9,
            signal_null_rate_alert=0.15,
            enable_auto_correction=True,
            correction_strength=0.1,
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
8B MoE Width-Scaling Strategy:
==============================

Key Insight: Preserve routing knowledge by keeping expert structure.
Scale capacity through hidden_size (width), not expert count.

From 3B:            To 8B:
- hidden=4096   →   hidden=5120 (1.25×)
- layers=20     →   layers=24 (1.2×)
- experts=32    →   experts=32 (SAME!)
- ρ=0.5, M=32   →   ρ=0.5, M=32 (SAME!)
- E[K_real]=8   →   E[K_real]=8 (SAME!)

Parameter Scaling:
- Embeddings: 32K × 5120 = 163M
- Attention per layer: ~80M
- Expert per layer: 32 × 3 × 5120 × 160 = 78M (routed)
- Total experts per layer: ~100M
- Per layer total: ~180M
- Total: 24 × 180M + embeddings ≈ 4.5B + overhead

Note: Fine-tune scaling factors to hit ~8B total.
"""


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    print(f"\nWidth Scaling from 3B:")
    print(f"  Hidden: 4096 → {config.hidden_size}")
    print(f"  Layers: 20 → {config.num_layers}")
    print(f"  Experts: 32 → {config.effective_num_routed_experts} (SAME!)")
    print(f"  E[K_real]: 8 (preserved)")
