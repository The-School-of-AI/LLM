"""
Stage 3: 8B MoE-8 Model Configuration
=====================================

Scale dimensions while keeping the SAME 8 experts.
This preserves routing knowledge learned in Stage 2.

Architecture:
- 48 layers (2× more)
- 4096 hidden dimension (2× larger)
- SAME 8 routed experts (but bigger)
- SAME 2 shared + 1 null
- Top-2 routing (preserved from 3B)

Key Transition (3B → 8B):
- Interpolate expert weights from 2048→4096 hidden
- Scale intermediate from 5504→11008
- Double the number of layers (24→48)
- Router input dimension scales automatically
- Routing PATTERNS are preserved!

Mathematical Rationale:
- Same 8 experts: Preserve learned specializations
- 2× hidden: More capacity per expert
- 2× layers: More processing depth
- Same Top-2: Same routing pattern works

This is DIMENSION SCALING, not EXPERT EXPANSION.
Expert expansion happens in Stage 4 (8→64).
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
    """Get 8B MoE-8 model configuration."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_8b_moe8",
        model_type=ModelType.MOE,
        stage=3,
        
        # Core dimensions (SCALED 2× from 3B)
        hidden_size=4096,                # 2× (was 2048)
        num_layers=40,                   # 2× (was 24)
        
        # MoE Configuration (SAME expert count as 3B!)
        num_routed_experts=8,            # SAME as 3B
        num_shared_experts=2,            # SAME as 3B
        num_null_experts=1,              # SAME as 3B
        moe_layer_frequency=1,           # MoE on ALL layers
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=128000,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
            junk_token_ids=[0],
            special_token_range=(0, 100),
            punctuation_range=(100, 200),
            common_word_range=(300, 1000),
        ),
        
        # Null Expert Router Configuration (data sparsity)
        router=RouterConfig(
            router_type=RouterType.NULL_EXPERT,
            top_k=2,
            data_sparsity=0.8,
            null_copies=0,
            use_aux_loss=True,
            aux_loss_weight=0.01,
        ),
        
        # Expert Configuration (SCALED 2×)
        expert=ExpertConfig(
            intermediate_size=2048,       # Reduced to 2048 (Fine-Grained)
            use_dual_gating=False,        # Disabled for efficiency
            gate_bias_init=0.0,
            expert_init_std=0.02,
            noise_std_for_expansion=1e-4,
        ),
        
        # Attention Configuration (SCALED)
        attention=AttentionConfig(
            attention_type="gsa",
            num_attention_heads=32,       # 2× (was 16)
            num_kv_heads=8,               # 2× (was 4) - maintain 4:1 GQA
            head_dim=128,                 # SAME (4096/32 = 128)
            rope_theta=10000.0,
            attention_dropout=0.0,
            # GSA defaults (Table 1)
            gsa_indexer_dim=64,
            gsa_indexer_heads=4,
            gsa_k_base=2048,
            gsa_k_min=256,
            gsa_k_max=4096,
        ),
        
        # Compute Budget
        compute_budget=ComputeBudget(
            max_params_total=int(10e9),     # 10B ceiling
            max_params_active=int(4e9),     # ~4B active
            target_tokens=int(1e12),        # 1T tokens
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


# Parameter breakdown
"""
8B MoE-8 Parameter Breakdown:
=============================

DIMENSION SCALING from 3B:
  - hidden_size: 2048 → 4096 (2×)
  - intermediate_size: 5504 → 11008 (2×)
  - num_layers: 24 → 48 (2×)
  - attention_heads: 16 → 32 (2×)
  - kv_heads: 4 → 8 (2×)

SAME expert structure:
  - 8 routed experts (but bigger)
  - 2 shared experts (but bigger)
  - 1 null expert
  - Top-2 routing

Per Expert (scaled):
  - W1: 4096 × 11008 = 45.1M
  - W2: 11008 × 4096 = 45.1M
  - W3: 4096 × 11008 = 45.1M
  - Per expert: ~135M (was ~34M, 4× larger)

Per MoE Layer:
  - Routed: 8 × 135M = 1.08B
  - Shared: 2 × 135M = 0.27B
  - Attention: ~67M (scaled)
  - Router: ~0.5M
  - Per layer: ~1.4B

But wait - 48 layers × 1.4B = 67B (too high!)

Resolution: MoE on every 2nd layer
  - moe_layer_frequency=2
  - 24 MoE layers, 24 dense layers
  - MoE: 24 × 1.4B = 33.6B (still too high)

Resolution 2: For 8B target, need smaller config
Let's recalculate for realistic 8B:
  - Use intermediate_size = 8192 (smaller ratio)
  - Or fewer layers
  - Or moe_layer_frequency=4

REALISTIC 8B CONFIG:
  - hidden=4096, intermediate=8192, layers=32
  - Expert: 3 × 4096 × 8192 = 100M
  - MoE layer (all): 8×100 + 2×100 + attention = 1.05B
  - With moe_freq=2: 16 MoE + 16 dense
  - Total ≈ 8B

NOTE: The exact numbers need tuning to fit 8B budget.
This config serves as the template.

Active Parameters:
  - 2 shared + 2 routed = 4 experts active
  - ~3B active params per forward pass
"""


# Scaling procedure from 3B MoE
"""
SCALING: 3B MoE-8 → 8B MoE-8
============================

Key insight: Keep SAME experts, just make them BIGGER.
Routing knowledge is preserved!

Step 1: Load 3B MoE Checkpoint
    moe_3b = load_checkpoint("3b_moe8.pt")

Step 2: Interpolate Expert Weights (2048→4096)
    for expert_idx in range(8):
        # For each weight matrix, interpolate dimensions
        old_w1 = moe_3b.experts[expert_idx].w1.data  # [2048, 5504]
        new_w1 = interpolate_weights(old_w1, [4096, 11008])
        
        # Or use linear projection expansion:
        # new_w = expansion_proj @ old_w @ expansion_proj.T
        
Step 3: Scale Router Input Projection
    # Router now takes 4096-dim input instead of 2048
    old_router_proj = moe_3b.router.query_proj  # [2048, heads×dim]
    new_router_proj = interpolate_weights(old_router_proj, [4096, heads×dim×2])

Step 4: Add New Layers
    # Original: 24 layers
    # New: 48 layers
    # Option A: Duplicate each layer
    # Option B: Initialize new layers fresh, interleave

Step 5: Scale Attention Projections
    # Similar interpolation for Q, K, V, O projections

GRADIENT PRESERVATION:
    Weight interpolation maintains gradient flow characteristics.
    The scaled model behaves similarly to the original at init.
    
ROUTING PRESERVATION:
    Expert "keys" in router are scaled proportionally.
    Relative routing decisions remain similar.
    Expert specializations are maintained.
"""


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    print(f"\nScaling from 3B:")
    print(f"  Hidden: 2048 → {config.hidden_size} (2×)")
    print(f"  Intermediate: 5504 → {config.expert.intermediate_size} (2×)")
    print(f"  Layers: 24 → {config.num_layers} (2×)")
    print(f"  Experts: 8 → {config.num_routed_experts} (SAME)")
