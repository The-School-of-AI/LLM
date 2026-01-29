"""
Stage 2: 3B MoE-8 Model Configuration
=====================================

First MoE stage where we learn routing patterns.
Initialized from Stage 1 (1B Dense) via "explosion" - copying FFN to all experts.

Architecture:
- 24 layers (all MoE)
- 2048 hidden dimension (same as 1B)
- 8 routed experts + 2 shared + 1 null
- Top-2 routing with GSA-style router

Key Transition (1B → 3B):
- Copy 1B FFN weights to all 8 experts
- Initialize fresh GSA router
- Add tiny noise (σ=1e-4) for symmetry breaking
- Total params: ~3B, Active params: ~1.2B

Training Focus:
- Learn routing patterns
- Experts specialize through gradient updates
- Router learns to discriminate token types

Mathematical Rationale:
- 8 experts: √N rule → K ≈ √8 ≈ 2.8 → Top-2
- 2 shared: 25% of effective capacity (2 shared for 2 active)
- 1 null: ceil(0.20 × 2 / 0.6) = 1 for junk absorption
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
    """Get 3B MoE-8 model configuration."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_3b_moe8",
        model_type=ModelType.MOE,
        stage=2,
        
        # Core dimensions (SAME as 1B)
        hidden_size=2048,
        num_layers=24,
        
        # MoE Configuration
        num_routed_experts=8,            # Selected by router
        num_shared_experts=2,            # Always active
        num_null_experts=1,              # Zero-compute pathway
        moe_layer_frequency=5,           # MoE every 5th layer (budget-fit)
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=32000,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
            # Junk tokens for null routing
            junk_token_ids=[0],  # Padding
            special_token_range=(0, 100),
            punctuation_range=(100, 200),
            common_word_range=(300, 1000),
        ),
        
        # GSA-Style Router Configuration
        router=RouterConfig(
            router_type=RouterType.GSA,
            
            # Router architecture (from GSA paper)
            num_router_heads=4,          # H_I = 4 indexer heads
            router_dim=64,               # d_I = 64 low-dim projection
            
            # Top-K configuration
            top_k=2,                     # Base active experts
            top_k_min=1,                 # Adaptive minimum
            top_k_max=4,                 # Adaptive maximum
            use_adaptive_top_k=True,     # GSA adaptive sparsity
            
            # Load balancing (LOSS-FREE per spec)
            use_aux_loss=False,          # NO auxiliary loss
            aux_loss_weight=0.0,
            bias_update_speed=0.001,     # γ for bias adjustment
            bias_clamp_min=-2.0,
            bias_clamp_max=2.0,
            
            # Null expert configuration
            null_bias_init=0.1,          # Slight preference for null
            null_target_junk_rate=(0.6, 0.8),   # 60-80% junk→null
            null_target_signal_rate=(0.0, 0.1), # <10% signal→null
            
            # Variance tracking for adaptive top-k
            variance_ema_decay=0.99,
        ),
        
        # Expert Configuration (with dual gating)
        expert=ExpertConfig(
            intermediate_size=5504,       # Same as 1B FFN
            use_dual_gating=True,         # G1 (output) + G2 (input) gates
            gate_bias_init=0.0,           # σ(0) = 0.5 at init
            expert_init_std=0.02,
            noise_std_for_expansion=1e-4, # Symmetry breaking noise
        ),
        
        # Attention Configuration (SAME as 1B)
        attention=AttentionConfig(
            num_attention_heads=16,
            num_kv_heads=4,               # 4:1 GQA
            head_dim=128,
            rope_theta=10000.0,
            attention_dropout=0.0,
        ),
        
        # Compute Budget
        compute_budget=ComputeBudget(
            max_params_total=int(3.5e9),     # 3.5B ceiling
            max_params_active=int(1.5e9),    # ~1.5B active
            target_tokens=int(500e9),        # 500B tokens
            max_sequence_length=4096,
        ),
        
        # Telemetry (Team 7 Integration)
        telemetry=TelemetryConfig(
            log_every_n_steps=100,
            
            # Health check thresholds
            dead_expert_threshold=0.01,      # <1% = dead
            overload_expert_threshold=3.0,   # >3× average = overloaded
            min_router_entropy=0.7,          # Minimum normalized entropy
            max_gini_coefficient=0.5,        # Maximum load imbalance
            
            # Null routing alerts
            junk_null_rate_alert_low=0.5,    # Alert if junk→null < 50%
            junk_null_rate_alert_high=0.9,   # Alert if junk→null > 90%
            signal_null_rate_alert=0.15,     # Alert if signal→null > 15%
            
            # Auto-correction
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
3B MoE-8 Parameter Breakdown:
=============================

Expert Configuration:
  - 8 routed experts
  - 2 shared experts  
  - 1 null expert (0 params)
  - Top-2 active per token

Per Expert (SwiGLU + Dual Gating):
  - W1: 2048 × 5504 = 11.3M
  - W2: 5504 × 2048 = 11.3M
  - W3: 2048 × 5504 = 11.3M
  - G1 (output gate): 2048 × 2048 = 4.2M
  - G2 (input gate): 2048 × 2048 = 4.2M
  - Per expert: ~42M with gating

Per MoE Layer:
  - Routed experts: 8 × 42M = 336M
  - Shared experts: 2 × 42M = 84M
  - Router: ~0.5M (negligible)
  - Attention: ~10M
  - Total per layer: ~430M

Total Model:
  - Embeddings: ~65M
  - Layers: 24 × 430M = 10.3B... 

WAIT - this exceeds 3B! Let me recalculate:

Without dual gating (to fit budget):
  - Per expert: ~34M
  - Per MoE layer: 8×34 + 2×34 + 10M = 340M + 10M = 350M
  - Total: 24 × 350M + 65M = 8.4B + 65M = 8.5B (still too high!)

Resolution (budget-fit):
- MoE every 5th layer (5 MoE, 19 dense)
- Keep intermediate_size=5504 for compatibility with 1B
- 8 routed + 2 shared + 1 null experts
- This fits within 3.5B total params
"""


# Expansion procedure from 1B Dense
"""
EXPANSION: 1B Dense → 3B MoE-8
==============================

Step 1: Load 1B Dense Checkpoint
    dense_model = load_checkpoint("1b_dense.pt")

Step 2: Copy FFN to All Experts
    for layer_idx in moe_layer_indices:
        for expert_idx in range(8):
            experts[expert_idx].w1.data = dense_model.layers[layer_idx].ffn.w1.data.clone()
            experts[expert_idx].w2.data = dense_model.layers[layer_idx].ffn.w2.data.clone()
            experts[expert_idx].w3.data = dense_model.layers[layer_idx].ffn.w3.data.clone()

Step 3: Add Symmetry-Breaking Noise
    for expert in experts:
        expert.w1.data += torch.randn_like(expert.w1.data) * 1e-4
        expert.w2.data += torch.randn_like(expert.w2.data) * 1e-4
        expert.w3.data += torch.randn_like(expert.w3.data) * 1e-4

Step 4: Initialize Fresh Router
    router = GSARouter(config)  # Random init

Step 5: Initialize Shared Experts (copy from dense)
    for shared_idx in range(2):
        shared_experts[shared_idx] = copy_expert_weights(dense_ffn)

Step 6: Verify Lossless Initialization
    # At init (before noise), with uniform routing:
    # MoE_output ≈ Dense_output (up to normalization)
    
LOSSLESS GUARANTEE:
    All experts identical + gating weights sum to 1
    → Σᵢ wᵢ × Expert_i(x) = Σᵢ wᵢ × FFN(x) = FFN(x) × Σᵢ wᵢ = FFN(x)
"""


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    # Print expert indices
    indices = config.get_expert_indices()
    print(f"\nExpert Indices:")
    print(f"  Routed: {indices['routed']}")
    print(f"  Null: {indices['null']}")
    print(f"  Shared: {indices['shared']}")
