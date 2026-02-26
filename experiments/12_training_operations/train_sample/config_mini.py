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

    High-util dry-run profile for single-GPU testing (A10 24GB class).
    Preserves all architectural paths while increasing throughput pressure.
    """
    # Core dimensions
    config.vocab_size = 8192
    config.hidden_size = 512
    config.num_layers = 12  # 9 DeltaNet + 3 GSA (preserves 75/25 split)

    # Attention mix (auto-derived from num_layers by layer pattern: every 4th is GSA)
    config.num_deltanet_layers = 9
    config.num_gsa_layers = 3

    # DeltaNet
    config.delta_v_heads = 8  # hidden_size / delta_head_dim = 512 / 64
    config.delta_head_dim = 64
    config.delta_gate_dim = 64

    # GSA
    config.gsa_num_heads = 8  # hidden_size / gsa_head_dim = 512 / 64
    config.gsa_head_dim = 64
    config.gsa_k_base = 64
    config.gsa_k_min = 8
    config.gsa_k_max = 128
    config.gsa_indexer_heads = 4  # must divide gsa_num_heads

    # MoE — small but still exercises routing, null experts, aux loss
    config.num_real_experts = 8
    config.num_null_experts = 8  # rho=0.5 preserved
    config.total_expert_slots = 16
    config.top_k = 2
    config.expert_intermediate_size = 768
    config.shared_expert_intermediate_size = 1024
    config.data_sparsity = 0.5

    # MTP — keep enabled
    config.enable_mtp = True
    config.mtp_num_predictions = 2

    # mHC — reduced streams
    config.n_streams = 4
    config.sinkhorn_iters = 3

    # Context — small for fast iteration
    config.max_seq_len = 1024
    config.rope_base = 10000
    config.rope_original_max_position = 1024
    config.rope_scaling_factor = 1.0

    # Training
    config.dropout = 0.0

    return config
