"""
Model loading utilities for DeepSpeed training.

This module contains functions for loading and initializing models
for training with DeepSpeed optimization.
"""

import torch
from transformers import AutoModelForCausalLM, Qwen2MoeConfig, Qwen2MoeForCausalLM

from .utils import print_rank_0


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
