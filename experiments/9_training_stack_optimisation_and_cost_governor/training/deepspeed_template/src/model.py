"""
Model loading utilities for DeepSpeed training.

This module contains functions for loading and initializing models
for training with DeepSpeed optimization.
"""

import torch
from transformers import AutoModelForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM

from .utils import print_rank_0
from .models import Model3B, ModelConfig, KroneckerConfig, KroneckerEmbeddings


def get_qwen2_moe_model(device=None, print_info=True):
    """
    Create a Qwen2 MoE model from scratch with custom configuration.

    This creates a small-scale Mixture of Experts (MoE) model with:
    - 8 experts, 2 active per token
    - 20 layers, 768 hidden size
    - Grouped-query attention (12 attention heads, 4 key-value heads)
    - Router auxiliary loss for load balancing

    Args:
        device: Device to load model on (default: None, let model decide)
        print_info: Whether to print model information

    Returns:
        The initialized Qwen2 MoE model
    """
    if print_info:
        print_rank_0("  Creating Qwen2 MoE model from scratch...")

    # Configure Qwen2 model with MoE
    config = Qwen2MoeConfig(
            # Vocabulary
            vocab_size=151936,
            
            # Model architecture
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            
            # MoE architecture
            moe_intermediate_size=2048,           # Per-expert FFN (4x hidden_size)
            num_experts=8,                         # Total experts per layer
            num_experts_per_tok=2,                 # Top-K routing
            shared_expert_intermediate_size=2048,  # Shared expert FFN
            
            # Router configuration
            norm_topk_prob=True,
            output_router_logits=True,
            router_jitter_noise=0.0,
            
            # Standard settings
            rms_norm_eps=1e-6,
            use_cache=False,
            tie_word_embeddings=False,
            rope_theta=1000000.0,
            attention_dropout=0.0,
            torch_dtype=torch.bfloat16,
        )

    # Initialize model from config
    model = Qwen2MoeForCausalLM(config)

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    if print_info:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print_rank_0("  Model created: Qwen2 MoE")
        print_rank_0("  Configuration:")
        print_rank_0(f"    - Hidden size: {config.hidden_size}")
        print_rank_0(f"    - Layers: {config.num_hidden_layers}")
        print_rank_0(f"    - Attention heads: {config.num_attention_heads}")
        print_rank_0(f"    - KV heads: {config.num_key_value_heads}")
        print_rank_0(f"    - MoE experts: {config.num_experts}")
        print_rank_0(f"    - Active experts per token: {config.num_experts_per_tok}")
        print_rank_0(f"  Total parameters: {num_params:,}")
        print_rank_0(f"  Trainable parameters: {num_trainable_params:,}")
        print_rank_0("  Gradient checkpointing: Enabled")

    # Move to device if specified
    if device is not None:
        model = model.to(device)
        if print_info:
            print_rank_0(f"  Model moved to device: {device}")

    return model


def get_reversible_model(device=None, print_info=True, embedding_type="kronecker", tokenizer=None):
    """
    Create a Reversible 3B model from scratch with custom configuration.
    
    This creates a memory-efficient reversible model with:
    - Reversible midpoint integration for memory savings
    - Hybrid Gated DeltaNet + Gated Sparse Attention
    - MoE with null experts
    - Multi-Token Prediction (MTP)
    - Multi-Head Composition (mHC)
    
    Args:
        device: Device to load model on (default: None, let model decide)
        print_info: Whether to print model information
        embedding_type: Type of embedding ("kronecker" or "standard")
        tokenizer: Optional tokenizer to extract vocabulary from (required for Kronecker embeddings)
    
    Returns:
        The initialized reversible model
    """
    if print_info:
        print_rank_0("  Creating Reversible 3B Model from scratch...")
    
    # Create model configuration
    config = ModelConfig()
    
    # Update vocab size to match tokenizer (required for proper token ID handling)
    # IMPORTANT: Use len(tokenizer) not tokenizer.vocab_size to include special tokens!
    if tokenizer is not None:
        # len(tokenizer) includes special tokens (pad, eos, etc.) which may have IDs >= vocab_size
        config.vocab_size = len(tokenizer)
        if print_info:
            print_rank_0(f"  Updated model vocab_size to match tokenizer: {config.vocab_size:,}")
            print_rank_0(f"    (tokenizer.vocab_size={tokenizer.vocab_size}, len(tokenizer)={len(tokenizer)})")
    else:
        if print_info:
            print_rank_0(f"  ⚠️  WARNING: No tokenizer provided, using default vocab_size: {config.vocab_size:,}")
            print_rank_0(f"  This may cause index errors if tokenizer vocab size doesn't match!")
    
    # For Kronecker embeddings, we need to create vocabulary and codec
    if embedding_type == "kronecker":
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required for Kronecker embeddings. "
                "Please pass tokenizer=tokenizer when creating the model, "
                "or use embedding_type='standard' instead."
            )
        
        # Extract vocabulary words from tokenizer
        vocab_size = config.vocab_size
        # Convert token IDs to their string representations
        bpe_vocab = []
        for i in range(vocab_size):
            try:
                token = tokenizer.decode([i])
                bpe_vocab.append(token if token else f"<unk_{i}>")
            except:
                bpe_vocab.append(f"<unk_{i}>")
        
        # Create Kronecker codec
        pf_config = KroneckerConfig(
            CHAR_DIM=256,
            POS_DIM=32,
            D=8192,
            length_normalize=True,
            truncate_long_words=True
        )
        pf_codec = KroneckerEmbeddings(pf_config)
        
        if print_info:
            print_rank_0(f"  Using Kronecker Product Embeddings (byte-level)")
            print_rank_0(f"  Loaded vocabulary from tokenizer: {vocab_size:,} tokens")
    else:
        bpe_vocab = None
        pf_codec = None
        if print_info:
            print_rank_0("  Using Standard Embeddings")
    
    # Create model
    model = Model3B(
        config=config,
        embedding_type=embedding_type,
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec
    )
    
    # IMPORTANT: Disable gradient checkpointing for reversible models
    # Reversible models handle memory efficiency through their architecture
    # and don't need/shouldn't use gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
    
    if print_info:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print_rank_0("  Model created: Reversible 3B")
        print_rank_0("  Configuration:")
        print_rank_0(f"    - Hidden size: {config.hidden_size}")
        print_rank_0(f"    - Layers: {config.num_layers}")
        print_rank_0(f"    - DeltaNet layers: {config.num_deltanet_layers} ({config.num_deltanet_layers/config.num_layers*100:.0f}%)")
        print_rank_0(f"    - GSA layers: {config.num_gsa_layers} ({config.num_gsa_layers/config.num_layers*100:.0f}%)")
        print_rank_0(f"    - MoE experts: {config.num_real_experts} real + {config.num_null_experts} null")
        print_rank_0(f"    - Top-k: {config.top_k} (active experts)")
        print_rank_0(f"    - MTP predictions: {config.mtp_num_predictions}" if config.enable_mtp else "    - MTP: Disabled")
        print_rank_0(f"    - Context: {config.max_seq_len:,} tokens")
        print_rank_0(f"    - Reversible: Yes (Midpoint Integration, step_size=0.25, a=0.5)")
        print_rank_0(f"  Total parameters: {num_params:,} (~{num_params/1e9:.2f}B)")
        print_rank_0(f"  Trainable parameters: {num_trainable_params:,}")
        print_rank_0("  ⚠️  Gradient checkpointing: Disabled (using reversible architecture)")
    
    # Move to device if specified
    if device is not None:
        model = model.to(device)
        if print_info:
            print_rank_0(f"  Model moved to device: {device}")
    
    return model


def get_model(model_name, device=None, print_info=True):
    """
    Load a pretrained model for causal language modeling.

    Args:
        model_name: Name of the pretrained model from HuggingFace Hub
        device: Device to load model on (default: None, let model decide)
        print_info: Whether to print model information

    Returns:
        The loaded model
    """
    if print_info:
        print_rank_0(f"  Loading model: {model_name}")

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if print_info:
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print_rank_0(f"  Model loaded: {model_name}")
        print_rank_0(f"  Total parameters: {num_params:,}")
        print_rank_0(f"  Trainable parameters: {num_trainable_params:,}")

    # Move to device if specified
    if device is not None:
        model = model.to(device)
        if print_info:
            print_rank_0(f"  Model moved to device: {device}")

    return model
