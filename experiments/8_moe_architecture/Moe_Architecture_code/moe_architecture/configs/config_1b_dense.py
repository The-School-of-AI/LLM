"""
Stage 1: 1B Dense Model Configuration
=====================================

Foundation model that establishes base capabilities.
This model's FFN weights become the template for all experts in Stage 2.

Architecture:
- Pure dense transformer (no MoE)
- 24 layers
- 2048 hidden dimension
- SwiGLU FFN with 5504 intermediate (8/3 * 2048 - Swiglu hidden layer expansion rule)

Training Target:
- ~100B tokens for foundation
- Checkpoint used for Stage 2 initialization
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
    """Get 1B Dense model configuration."""
    
    return MoEModelConfig(
        # Model identification
        model_name="team8_1b_dense",
        model_type=ModelType.DENSE,  # Pure dense, no MoE
        stage=1,
        
        # Core dimensions
        hidden_size=2048,
        num_layers=16,
        
        # No MoE for dense model
        num_routed_experts=0,
        num_shared_experts=0,
        num_null_experts=0,
        moe_layer_frequency=1,  # Irrelevant for dense
        
        # Tokenizer (Team 6 specification)
        tokenizer=TokenizerConfig(
            vocab_size=128000,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            unk_token_id=3,
        ),
        
        # Router (disabled for dense)
        router=RouterConfig(
            router_type=RouterType.NONE,
            top_k=0,
        ),
        
        # FFN configuration (becomes expert template)
        expert=ExpertConfig(
            intermediate_size=4096,      # ≈ 2.7 × hidden_size
            use_dual_gating=False,       # No gating in dense model
            expert_init_std=0.02,
        ),
        
        # Attention configuration (GQA)
        attention=AttentionConfig(
            attention_type="gsa",
            num_attention_heads=16,      # Query heads
            num_kv_heads=4,              # 4:1 GQA ratio
            head_dim=128,                # 2048 / 16
            rope_theta=10000.0,
            attention_dropout=0.0,
            # GSA defaults (Table 1)
            gsa_indexer_dim=64,
            gsa_indexer_heads=4,
            gsa_k_base=2048,
            gsa_k_min=256,
            gsa_k_max=4096,
        ),
        
        # Compute budget
        compute_budget=ComputeBudget(
            max_params_total=int(1.5e9),     # 1.5B ceiling
            max_params_active=int(1.5e9),    # Same (dense)
            target_tokens=int(100e9),        # 100B tokens
            max_sequence_length=4096,
        ),
        
        # Telemetry (minimal for dense)
        telemetry=TelemetryConfig(
            log_every_n_steps=100,
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



if __name__ == "__main__":
    config = get_config()
    print(config.summary())
