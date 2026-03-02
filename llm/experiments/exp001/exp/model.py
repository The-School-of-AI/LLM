from transformers.tokenization_utils_tokenizers import TokenizersBackend

from llm.models import KroneckerConfig, KroneckerEmbeddings, Model1B, ModelConfig


def build_kronecker_vocab(
    tokenizer: TokenizersBackend,
) -> tuple[list[str], KroneckerEmbeddings]:
    # Use len(tokenizer) to include special tokens (pad, eos, etc.)
    vocab_size = len(tokenizer)
    bpe_vocab = []
    for i in range(vocab_size):
        try:
            token = tokenizer.decode([i])
            bpe_vocab.append(token if token else f"<unk_{i}>")
        except Exception:
            bpe_vocab.append(f"<unk_{i}>")

    # Create Kronecker codec
    pf_config = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True,
    )

    return bpe_vocab, KroneckerEmbeddings(pf_config)


def build_model(
    embedding_type: str,
    bpe_vocab: list[str] | None,
    k_embed: KroneckerEmbeddings | None,
) -> Model1B:
    config = ModelConfig(
        vocab_size=131072,
        hidden_size=512,  # Down from 4096 — biggest VRAM saving
        num_layers=2,
        # Attention Mix (1 of each)
        num_deltanet_layers=1,
        num_gsa_layers=1,
        # DeltaNet — heads must satisfy: num_heads * head_dim == hidden_size
        delta_v_heads=4,  # 512 / 128 = 4
        delta_head_dim=128,
        delta_gate_dim=64,  # ~12% of hidden_size
        # GSA — head_dim = hidden_size / num_heads
        gsa_num_heads=4,  # 512 / 128 = 4
        gsa_head_dim=128,  # Down from 256
        gsa_k_base=32,
        gsa_k_min=4,
        gsa_k_max=64,
        gsa_indexer_heads=2,
        # MoE (dense, no change needed)
        num_real_experts=0,
        num_null_experts=0,
        total_expert_slots=0,
        top_k=0,
        expert_intermediate_size=512,
        shared_expert_intermediate_size=512,  # Down from 2048
        data_sparsity=0.0,
        # MTP
        enable_mtp=False,
        mtp_num_predictions=1,
        # mHC — n_streams=2 instead of 4 cuts stream memory in half
        n_streams=2,
        sinkhorn_iters=5,  # 20 is overkill for a forward pass check
        # Context — keep short for the test
        max_seq_len=4096,
        rope_base=10000,
        rope_original_max_position=4096,
        rope_scaling_factor=1.0,
        # Training
        dropout=0.0,
        require_fused_deltanet_kernel=True,
        require_fused_gsa_kernel=True,
    )

    return Model1B(
        config=config,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=k_embed,
    )
