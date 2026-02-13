"""
Mini configuration for dry-run testing of the full 70B architecture.

Shrinks the model from ~70B to ~15-25M parameters while preserving every
architectural component: DeltaNet, GSA, MoE with null experts, mHC,
MTP, memory stream recurrence, reversible midpoint integration.

Usage:
    from config_mini import apply_mini_config
    from recurrence_model_70b import ModelConfig, Model70B

    config = ModelConfig()
    apply_mini_config(config)
    model = Model70B(config, embedding_type="standard")
"""

def apply_mini_config(config):
    """
    Mutate a ModelConfig in-place to create a miniaturized version
    that exercises every architectural path.

    ~15-25M params. Runs in seconds on CPU, <1GB on GPU.
    """
    # Core dimensions
    config.vocab_size = 4096
    config.hidden_size = 256
    config.num_layers = 4  # 3 DeltaNet + 1 GSA (preserves 75/25 split)

    # Attention mix (auto-derived from num_layers by layer pattern: every 4th is GSA)
    config.num_deltanet_layers = 3
    config.num_gsa_layers = 1

    # DeltaNet
    config.delta_v_heads = 4       # hidden_size / delta_head_dim = 256 / 64
    config.delta_head_dim = 64
    config.delta_gate_dim = 24     # ~9.4% of hidden_size

    # GSA
    config.gsa_num_heads = 4       # hidden_size / gsa_head_dim = 256 / 64
    config.gsa_head_dim = 64
    config.gsa_k_base = 32
    config.gsa_k_min = 4
    config.gsa_k_max = 64
    config.gsa_indexer_heads = 2   # must divide gsa_num_heads

    # MoE — small but still exercises routing, null experts, aux loss
    config.num_real_experts = 4
    config.num_null_experts = 4    # rho=0.5 preserved
    config.total_expert_slots = 8
    config.top_k = 2
    config.expert_intermediate_size = 128
    config.shared_expert_intermediate_size = 256
    config.data_sparsity = 0.5

    # MTP — keep enabled
    config.enable_mtp = True
    config.mtp_num_predictions = 2

    # mHC — reduced streams
    config.n_streams = 2
    config.sinkhorn_iters = 3

    # Context — small for fast iteration
    config.max_seq_len = 512
    config.rope_base = 10000
    config.rope_original_max_position = 512  # no YARN scaling needed at this size
    config.rope_scaling_factor = 1.0

    # Training
    config.dropout = 0.0

    return config
