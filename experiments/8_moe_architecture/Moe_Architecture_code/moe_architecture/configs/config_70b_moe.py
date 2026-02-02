"""
Stage 4: 70B MoE-64 Model Configuration
=======================================

Expert expansion stage: 8 experts → 64 experts.
Each of the 8 parent experts becomes 8 children (8×8=64).

Architecture:
- 80 layers
- 4096 hidden dimension (same as 8B)
- 64 routed experts (expanded from 8!)
- 4 shared + 2 null
- Top-4 routing (increased from Top-2)

Key Transition (8B → 70B):
- Each of 8 experts copied to 8 children
- Fresh GSA router for 64 experts
- Noise added for children divergence
- Top-K increased: 2→4 (because √64=8, use 0.5×8=4)

Mathematical Rationale:
- 64 experts: 8 parents × 8 children
- Top-4: K = 0.5 × √64 = 4
- 4 shared: 6.25% of routed (appropriate for large MoE)
- 2 null: ceil(0.20 × 4 / 0.6) = 2

This is EXPERT EXPANSION, not dimension scaling.
We increase the number of experts for fine-grained specialization.
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
    """Get 70B MoE-64 model configuration."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_70b_moe64",
        model_type=ModelType.MOE,
        stage=4,
        
        # Core dimensions (same as 8B for hidden, more layers)
        hidden_size=2048,                # SAME as 8B
        num_layers=40,                   # FIXED depth (same as 8B)
        
        # MoE Configuration (EXPANDED experts!)
        num_routed_experts=512,           # 8× expansion (was 8)
        num_shared_experts=4,            # 2× expansion (was 2)
        num_null_experts=1,              # Single null expert for null-copy routing
        moe_layer_frequency=1,           # MoE on ALL layers
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=32000,
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
            top_k=4,
            data_sparsity=0.8,
            null_copies=256,
            use_aux_loss=True,
            aux_loss_weight=0.01,
        ),
        
        # Expert Configuration
        expert=ExpertConfig(
            intermediate_size=512,       # Reduced to 2048 (Fine-Grained)
            use_dual_gating=False,         # G1+G2 for collapse prevention
            gate_bias_init=0.0,
            expert_init_std=0.02,
            noise_std_for_expansion=1e-3, # Slightly more noise for divergence
        ),
        
        # Attention Configuration (same as 8B)
        attention=AttentionConfig(
            attention_type="gsa",
            num_attention_heads=32,
            num_kv_heads=8,               # 4:1 GQA
            head_dim=128,
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
            max_params_total=int(80e9),     # 80B ceiling
            max_params_active=int(15e9),    # ~15B active
            target_tokens=int(2e12),        # 2T tokens
            max_sequence_length=4096,
        ),
        
        # Telemetry (Team 7 Integration)
        telemetry=TelemetryConfig(
            log_every_n_steps=100,
            
            # Health checks (stricter for 64 experts)
            dead_expert_threshold=0.005,    # <0.5% = dead (stricter)
            overload_expert_threshold=2.5,  # >2.5× average = overloaded
            min_router_entropy=0.75,        # Higher entropy needed
            max_gini_coefficient=0.4,       # Stricter balance
            
            # Null routing alerts
            junk_null_rate_alert_low=0.5,
            junk_null_rate_alert_high=0.9,
            signal_null_rate_alert=0.15,
            
            # Auto-correction
            enable_auto_correction=True,
            correction_strength=0.05,       # Gentler correction at scale
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
70B MoE-64 Parameter Breakdown:
===============================

EXPERT EXPANSION from 8B:
  - Routed experts: 8 → 64 (8× expansion)
  - Shared experts: 2 → 4 (2× expansion)
  - Null experts: 1 → 2
  - Top-K: 2 → 4
  - Layers: 48 → 80

Expert Configuration:
  - 64 routed experts
  - 4 shared experts
  - 2 null experts (0 params)
  - Top-4 active per token

Per Expert:
  - W1: 4096 × 11008 = 45.1M
  - W2: 11008 × 4096 = 45.1M
  - W3: 4096 × 11008 = 45.1M
  - Dual gating: ~33M
  - Per expert: ~168M (with gating)
  - Or ~135M without gating

Without gating (to fit budget):
  - Per expert: ~135M

Per MoE Layer:
  - Routed: 64 × 135M = 8.64B
  - Shared: 4 × 135M = 0.54B
  - Attention: ~67M
  - Router: ~8M (larger for 64 experts)
  - Per layer: ~9.3B

Total (80 layers): 80 × 9.3B = 744B (way too high!)

REALISTIC CONFIGURATION for ~70B:
Need to use smaller experts or MoE on fewer layers.

Option 1: MoE every 4th layer (20 MoE, 60 dense)
  - MoE: 20 × 9.3B = 186B (still too high)

Option 2: Smaller expert intermediate
  intermediate = 4096 (1:1 ratio)
  Per expert: 3 × 4096 × 4096 = 50.3M
  Per MoE layer: 64×50 + 4×50 + 67M = 3.47B
  Total (80 layers MoE): 80 × 3.47B = 278B (too high)

Option 3: Even smaller or sparse experts
  intermediate = 2048 (0.5:1 ratio)
  Per expert: 3 × 4096 × 2048 = 25.2M
  Per MoE layer: 64×25 + 4×25 + 67M = 1.77B
  MoE every 2nd layer: 40 × 1.77B + 40 × 0.54B = 70.8B + 21.6B = 92B
  
  Still high. Need moe_freq=4:
  20 MoE × 1.77B + 60 dense × 0.54B = 35.4B + 32.4B = 67.8B ✓

FINAL REALISTIC CONFIG:
  - moe_layer_frequency=4 (20 MoE layers)
  - intermediate_size=512 (or fine-grained experts)
  - 64 routed + 4 shared experts
  - Top-4 routing
  - ~70B total parameters
  - ~12B active parameters

Active Parameters (per forward):
  - 4 shared + 4 routed = 8 experts active
  - Per MoE layer: 8 × 25M = 0.2B
  - 20 MoE layers × 0.2B + 60 dense × 0.54B + attention + embeddings
  - ≈ 4B + 32B + attention ≈ 12B active
"""


# Expert hierarchy for 8→64 expansion
"""
EXPERT HIERARCHY (8 Parents → 64 Children):
===========================================

Parent 0 (Code Expert):
  ├── Child 0.0: Python code
  ├── Child 0.1: JavaScript code
  ├── Child 0.2: C/C++ code
  ├── Child 0.3: Java code
  ├── Child 0.4: SQL queries
  ├── Child 0.5: Shell scripts
  ├── Child 0.6: HTML/CSS
  └── Child 0.7: Other code

Parent 1 (Math Expert):
  ├── Child 1.0: Arithmetic
  ├── Child 1.1: Algebra
  ├── Child 1.2: Calculus
  ├── Child 1.3: Statistics
  ├── Child 1.4: Linear algebra
  ├── Child 1.5: Number theory
  ├── Child 1.6: Geometry
  └── Child 1.7: Applied math

... (6 more parent groups)

Total: 8 parents × 8 children = 64 experts
"""


# Expansion procedure from 8B MoE
"""
EXPANSION: 8B MoE-8 → 70B MoE-64
================================

Step 1: Load 8B MoE Checkpoint
    moe_8b = load_checkpoint("8b_moe8.pt")

Step 2: Expand Each Expert to 8 Children
    for parent_idx in range(8):
        parent_expert = moe_8b.experts[parent_idx]
        
        for child_idx in range(8):
            global_idx = parent_idx * 8 + child_idx
            
            # Copy parent weights
            new_experts[global_idx].w1.data = parent_expert.w1.data.clone()
            new_experts[global_idx].w2.data = parent_expert.w2.data.clone()
            new_experts[global_idx].w3.data = parent_expert.w3.data.clone()
            
            # Add larger noise for child divergence
            noise_std = 1e-3
            new_experts[global_idx].w1.data += torch.randn_like(...) * noise_std
            new_experts[global_idx].w2.data += torch.randn_like(...) * noise_std
            new_experts[global_idx].w3.data += torch.randn_like(...) * noise_std

Step 3: Initialize New Router for 64 Experts
    # Option A: Fresh random init
    new_router = GSARouter(config_64_experts)
    
    # Option B: Hierarchical warm-start (recommended)
    for parent_idx in range(8):
        parent_key = moe_8b.router.expert_keys[parent_idx]
        
        for child_idx in range(8):
            global_idx = parent_idx * 8 + child_idx
            # Child key = parent key + small offset
            new_router.expert_keys[global_idx] = parent_key + randn() * 0.1

Step 4: Expand Shared Experts (2 → 4)
    # Copy existing 2, initialize 2 new from average
    shared_avg = (moe_8b.shared[0] + moe_8b.shared[1]) / 2
    new_shared = [moe_8b.shared[0], moe_8b.shared[1], shared_avg.clone(), shared_avg.clone()]
    # Add noise to new ones
    new_shared[2].add_(randn() * 1e-4)
    new_shared[3].add_(randn() * 1e-4)

Step 5: Add Null Expert
    # Now have 2 null experts instead of 1
    # Second null initialized fresh

Step 6: Add More Layers (48 → 80)
    # Initialize new layers with MoE structure
    # Can use layer duplication or fresh init

HIERARCHICAL ROUTING PRESERVATION:
    With hierarchical key init, tokens that routed to Parent_i
    will initially prefer children {i.0, i.1, ..., i.7}.
    
    Over training, children specialize into sub-categories
    while maintaining parent's general domain.
    
    Example: Token "def" routes to Code Parent
    → Initially may route to any Code Child
    → After training: routes specifically to Python Child
"""


if __name__ == "__main__":
    config = get_config()
    print(config.summary())
    
    print(f"\nExpansion from 8B:")
    print(f"  Experts: 8 → {config.num_routed_experts} (8× expansion)")
    print(f"  Shared: 2 → {config.num_shared_experts} (2× expansion)")
    print(f"  Top-K: 2 → {config.router.top_k} (2× expansion)")
    print(f"  Null: 1 → {config.num_null_experts}")
    print(f"\nExpert Hierarchy:")
    print(f"  8 parent groups × 8 children = 64 total")
    print(f"  Active ratio: {config.active_expert_ratio:.1%}")
