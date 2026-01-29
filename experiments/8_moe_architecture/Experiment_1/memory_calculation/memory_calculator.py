#!/usr/bin/env python3
"""
Complete Memory Calculator for MoE Models
==========================================

Calculates ALL memory components:
1. Static: Weights + Gradients + Optimizer states
2. Activations: Attention (O(S²)) + FFN + Router
3. Buffers: Temporary allocations

Usage:
    python memory_calculator.py --model 3b_moe --batch 8 --seq 2048
    python memory_calculator.py --all
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration for memory calculation."""
    name: str
    total_params: float          # Total parameters
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    # MoE specific
    num_routed_experts: int = 0
    num_shared_experts: int = 0
    top_k: int = 0
    is_moe: bool = False


# Model configurations
MODELS = {
    "1b_dense": ModelConfig(
        name="1B Dense",
        total_params=1.0e9,
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5504,
        is_moe=False,
    ),
    "3b_moe": ModelConfig(
        name="3B MoE-8",
        total_params=3.0e9,
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        num_kv_heads=4,
        intermediate_size=5504,
        num_routed_experts=8,
        num_shared_experts=2,
        top_k=2,
        is_moe=True,
    ),
    "8b_moe": ModelConfig(
        name="8B MoE-8",
        total_params=8.0e9,
        hidden_size=4096,
        num_layers=48,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=11008,
        num_routed_experts=8,
        num_shared_experts=2,
        top_k=2,
        is_moe=True,
    ),
    "70b_moe": ModelConfig(
        name="70B MoE-64",
        total_params=70e9,
        hidden_size=4096,
        num_layers=80,
        num_heads=32,
        num_kv_heads=8,
        intermediate_size=11008,
        num_routed_experts=64,
        num_shared_experts=4,
        top_k=4,
        is_moe=True,
    ),
}


def calculate_static_memory(total_params: float, dtype_bytes: int = 2) -> dict:
    """
    Calculate static memory (independent of batch size).
    
    Components:
    - Model weights: dtype_bytes per param
    - Gradients: dtype_bytes per param
    - Optimizer (AdamW): 8 bytes per param (m + v in fp32)
    """
    model_weights = total_params * dtype_bytes
    gradients = total_params * dtype_bytes
    optimizer = total_params * 8  # Adam m + v in fp32
    
    return {
        'model_weights_bytes': model_weights,
        'gradients_bytes': gradients,
        'optimizer_bytes': optimizer,
        'total_bytes': model_weights + gradients + optimizer,
    }


def calculate_attention_activation(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    dtype_bytes: int = 2,
    use_flash_attention: bool = False,
) -> dict:
    """
    Calculate attention activation memory per layer.
    
    Key insight: Attention scores are O(S²) without Flash Attention!
    """
    B, S, H = batch_size, seq_length, hidden_size
    H_kv = hidden_size // (num_heads // num_kv_heads)  # KV head dimension
    
    # Input to attention
    input_memory = B * S * H * dtype_bytes
    
    # Q, K, V projections
    q_memory = B * S * H * dtype_bytes
    k_memory = B * S * H_kv * dtype_bytes  # GQA: fewer KV heads
    v_memory = B * S * H_kv * dtype_bytes
    
    # Attention scores: B × num_heads × S × S
    # This is the KILLER for memory!
    if use_flash_attention:
        # Flash Attention doesn't materialize full attention matrix
        # Instead uses tiling with O(S) memory
        attn_scores_memory = B * S * H * dtype_bytes
    else:
        # Full attention matrix: O(S²)
        attn_scores_memory = B * num_heads * S * S * dtype_bytes
    
    # Attention output
    attn_output_memory = B * S * H * dtype_bytes
    
    # Output projection
    o_proj_memory = B * S * H * dtype_bytes
    
    return {
        'input_bytes': input_memory,
        'qkv_bytes': q_memory + k_memory + v_memory,
        'attention_scores_bytes': attn_scores_memory,
        'attention_output_bytes': attn_output_memory,
        'o_proj_bytes': o_proj_memory,
        'total_bytes': (input_memory + q_memory + k_memory + v_memory + 
                       attn_scores_memory + attn_output_memory + o_proj_memory),
        'is_flash': use_flash_attention,
    }


def calculate_ffn_activation_dense(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    intermediate_size: int,
    dtype_bytes: int = 2,
) -> dict:
    """Calculate FFN activation memory for dense layer."""
    B, S, H = batch_size, seq_length, hidden_size
    I = intermediate_size
    
    # Input to FFN
    input_memory = B * S * H * dtype_bytes
    
    # SwiGLU: W1 (gate) and W3 (up) projections
    gate_memory = B * S * I * dtype_bytes
    up_memory = B * S * I * dtype_bytes
    
    # After SwiGLU activation
    activation_memory = B * S * I * dtype_bytes
    
    # Down projection output
    output_memory = B * S * H * dtype_bytes
    
    return {
        'input_bytes': input_memory,
        'gate_up_bytes': gate_memory + up_memory,
        'activation_bytes': activation_memory,
        'output_bytes': output_memory,
        'total_bytes': input_memory + gate_memory + up_memory + activation_memory + output_memory,
    }


def calculate_moe_activation(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    intermediate_size: int,
    num_routed_experts: int,
    num_shared_experts: int,
    top_k: int,
    dtype_bytes: int = 2,
) -> dict:
    """Calculate MoE FFN activation memory."""
    B, S, H = batch_size, seq_length, hidden_size
    I = intermediate_size
    
    # Router activations
    router_scores = B * S * (num_routed_experts + 1) * dtype_bytes  # +1 for null
    router_indices = B * S * top_k * 4  # int32
    router_weights = B * S * top_k * dtype_bytes
    router_total = router_scores + router_indices + router_weights
    
    # Shared experts (always computed for ALL tokens)
    shared_per_expert = B * S * I * 3 * dtype_bytes  # gate, up, activation
    shared_output = B * S * H * dtype_bytes
    shared_total = num_shared_experts * shared_per_expert + shared_output
    
    # Routed experts (only top_k computed, but for ALL tokens that route there)
    # In the worst case, all tokens route to same experts
    # Average case: tokens distributed across experts
    routed_per_expert = B * S * I * 3 * dtype_bytes
    routed_output = B * S * H * dtype_bytes
    routed_total = top_k * routed_per_expert + routed_output
    
    # Combination
    combine_memory = B * S * H * dtype_bytes
    
    return {
        'router_bytes': router_total,
        'shared_experts_bytes': shared_total,
        'routed_experts_bytes': routed_total,
        'combine_bytes': combine_memory,
        'total_bytes': router_total + shared_total + routed_total + combine_memory,
    }


def calculate_layer_norm_activation(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_norms: int = 2,  # Usually 2 per layer: attention + FFN
    dtype_bytes: int = 2,
) -> int:
    """Calculate layer norm activation memory."""
    return num_norms * batch_size * seq_length * hidden_size * dtype_bytes


def calculate_total_activation(
    config: ModelConfig,
    batch_size: int,
    seq_length: int,
    use_flash_attention: bool = False,
    use_checkpointing: bool = False,
    dtype_bytes: int = 2,
) -> dict:
    """Calculate total activation memory for full model."""
    
    # Attention (same for dense and MoE)
    attn = calculate_attention_activation(
        batch_size, seq_length,
        config.hidden_size, config.num_heads, config.num_kv_heads,
        dtype_bytes, use_flash_attention
    )
    
    # FFN/MoE
    if config.is_moe:
        ffn = calculate_moe_activation(
            batch_size, seq_length,
            config.hidden_size, config.intermediate_size,
            config.num_routed_experts, config.num_shared_experts, config.top_k,
            dtype_bytes
        )
    else:
        ffn = calculate_ffn_activation_dense(
            batch_size, seq_length,
            config.hidden_size, config.intermediate_size,
            dtype_bytes
        )
    
    # Layer norms
    norm = calculate_layer_norm_activation(
        batch_size, seq_length, config.hidden_size, 2, dtype_bytes
    )
    
    # Per layer total
    per_layer = attn['total_bytes'] + ffn['total_bytes'] + norm
    
    # Total across all layers
    total_no_ckpt = per_layer * config.num_layers
    
    # With checkpointing: save activations only at checkpoints
    # Typically saves 60-70% (divide by ~3)
    if use_checkpointing:
        total = total_no_ckpt / 3
    else:
        total = total_no_ckpt
    
    return {
        'attention_per_layer_bytes': attn['total_bytes'],
        'ffn_per_layer_bytes': ffn['total_bytes'],
        'norm_per_layer_bytes': norm,
        'per_layer_bytes': per_layer,
        'total_no_checkpoint_bytes': total_no_ckpt,
        'total_bytes': total,
        'use_checkpointing': use_checkpointing,
        'use_flash_attention': use_flash_attention,
        'attention_breakdown': attn,
        'ffn_breakdown': ffn,
    }


def calculate_total_memory(
    config: ModelConfig,
    batch_size: int = 8,
    seq_length: int = 2048,
    use_flash_attention: bool = False,
    use_checkpointing: bool = True,
    dtype_bytes: int = 2,
) -> dict:
    """Calculate complete training memory."""
    
    # Static memory
    static = calculate_static_memory(config.total_params, dtype_bytes)
    
    # Activation memory
    activation = calculate_total_activation(
        config, batch_size, seq_length,
        use_flash_attention, use_checkpointing, dtype_bytes
    )
    
    # Temporary buffers (~10% overhead)
    buffer_overhead = 0.1
    buffers = (static['total_bytes'] + activation['total_bytes']) * buffer_overhead
    
    # Total
    total = static['total_bytes'] + activation['total_bytes'] + buffers
    
    return {
        'config': config.name,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'static': static,
        'activation': activation,
        'buffers_bytes': buffers,
        'total_bytes': total,
        'total_gb': total / 1e9,
        # Breakdown percentages
        'static_pct': static['total_bytes'] / total * 100,
        'activation_pct': activation['total_bytes'] / total * 100,
        'buffers_pct': buffers / total * 100,
    }


def format_bytes(b: float) -> str:
    """Format bytes to human readable."""
    if b >= 1e12:
        return f"{b/1e12:.1f} TB"
    elif b >= 1e9:
        return f"{b/1e9:.1f} GB"
    elif b >= 1e6:
        return f"{b/1e6:.1f} MB"
    else:
        return f"{b:.0f} B"


def print_memory_analysis(result: dict, verbose: bool = True):
    """Print detailed memory analysis."""
    
    print(f"\n{'='*70}")
    print(f" {result['config']} Memory Analysis")
    print(f" Batch={result['batch_size']}, Seq={result['seq_length']}")
    print(f"{'='*70}")
    
    # Summary
    print(f"\n📊 TOTAL MEMORY: {format_bytes(result['total_bytes'])}")
    print(f"\n   Breakdown:")
    print(f"   ├── Static (weights+optimizer): {format_bytes(result['static']['total_bytes'])} ({result['static_pct']:.1f}%)")
    print(f"   ├── Activations:                {format_bytes(result['activation']['total_bytes'])} ({result['activation_pct']:.1f}%)")
    print(f"   └── Buffers:                    {format_bytes(result['buffers_bytes'])} ({result['buffers_pct']:.1f}%)")
    
    if verbose:
        # Static breakdown
        print(f"\n💾 Static Memory Breakdown:")
        print(f"   ├── Model weights:    {format_bytes(result['static']['model_weights_bytes'])}")
        print(f"   ├── Gradients:        {format_bytes(result['static']['gradients_bytes'])}")
        print(f"   └── Optimizer (Adam): {format_bytes(result['static']['optimizer_bytes'])}")
        
        # Activation breakdown
        act = result['activation']
        print(f"\n⚡ Activation Memory (per layer):")
        print(f"   ├── Attention:  {format_bytes(act['attention_per_layer_bytes'])}")
        if act['use_flash_attention']:
            print(f"   │   └── (Flash Attention enabled - O(S) instead of O(S²))")
        else:
            attn_scores = act['attention_breakdown']['attention_scores_bytes']
            print(f"   │   └── Attention scores: {format_bytes(attn_scores)} ⚠️ O(S²)!")
        print(f"   ├── FFN/MoE:    {format_bytes(act['ffn_per_layer_bytes'])}")
        print(f"   └── LayerNorm:  {format_bytes(act['norm_per_layer_bytes'])}")
        
        print(f"\n   Total per layer: {format_bytes(act['per_layer_bytes'])}")
        print(f"   × {result['batch_size']} batch × layers = {format_bytes(act['total_no_checkpoint_bytes'])} (no ckpt)")
        if act['use_checkpointing']:
            print(f"   With checkpointing: {format_bytes(act['total_bytes'])} (~66% saved)")
    
    # Recommendations
    print(f"\n🎯 GPU Requirements:")
    total_gb = result['total_gb']
    if total_gb <= 40:
        print(f"   ✓ Fits on 1× A100-40GB")
    elif total_gb <= 80:
        print(f"   ✓ Fits on 1× A100-80GB")
    elif total_gb <= 160:
        print(f"   → Need 2× A100-80GB with tensor parallelism")
    elif total_gb <= 640:
        print(f"   → Need 8× A100-80GB with ZeRO-2/3")
    else:
        gpus_needed = int(total_gb / 80) + 1
        print(f"   → Need {gpus_needed}× A100-80GB with ZeRO-3 + offloading")
    
    print(f"\n{'='*70}\n")


def compare_all_models(batch_size: int = 8, seq_length: int = 2048):
    """Compare memory across all models."""
    
    print(f"\n{'='*80}")
    print(f" Memory Comparison (Batch={batch_size}, Seq={seq_length})")
    print(f"{'='*80}")
    
    # Without optimizations
    print(f"\n📊 WITHOUT Optimizations (no checkpointing, no Flash Attention):")
    print(f"\n{'Model':<15} {'Static':<12} {'Activation':<15} {'Buffers':<10} {'TOTAL':<12}")
    print("-" * 70)
    
    for name, config in MODELS.items():
        result = calculate_total_memory(
            config, batch_size, seq_length,
            use_flash_attention=False,
            use_checkpointing=False
        )
        print(f"{config.name:<15} "
              f"{format_bytes(result['static']['total_bytes']):<12} "
              f"{format_bytes(result['activation']['total_bytes']):<15} "
              f"{format_bytes(result['buffers_bytes']):<10} "
              f"{format_bytes(result['total_bytes']):<12}")
    
    # With optimizations
    print(f"\n📊 WITH Optimizations (checkpointing + Flash Attention):")
    print(f"\n{'Model':<15} {'Static':<12} {'Activation':<15} {'Buffers':<10} {'TOTAL':<12} {'Savings':<10}")
    print("-" * 80)
    
    for name, config in MODELS.items():
        result_opt = calculate_total_memory(
            config, batch_size, seq_length,
            use_flash_attention=True,
            use_checkpointing=True
        )
        result_no_opt = calculate_total_memory(
            config, batch_size, seq_length,
            use_flash_attention=False,
            use_checkpointing=False
        )
        savings = (1 - result_opt['total_bytes'] / result_no_opt['total_bytes']) * 100
        
        print(f"{config.name:<15} "
              f"{format_bytes(result_opt['static']['total_bytes']):<12} "
              f"{format_bytes(result_opt['activation']['total_bytes']):<15} "
              f"{format_bytes(result_opt['buffers_bytes']):<10} "
              f"{format_bytes(result_opt['total_bytes']):<12} "
              f"{savings:.0f}%")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='MoE Memory Calculator')
    parser.add_argument('--model', choices=list(MODELS.keys()), help='Model to analyze')
    parser.add_argument('--all', action='store_true', help='Compare all models')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--seq', type=int, default=2048, help='Sequence length')
    parser.add_argument('--flash', action='store_true', help='Use Flash Attention')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpointing')
    
    args = parser.parse_args()
    
    print("\n" + "🧮 " * 20)
    print("  MoE Memory Calculator")
    print("  Complete RAM Analysis")
    print("🧮 " * 20)
    
    if args.all:
        compare_all_models(args.batch, args.seq)
    elif args.model:
        config = MODELS[args.model]
        result = calculate_total_memory(
            config,
            batch_size=args.batch,
            seq_length=args.seq,
            use_flash_attention=args.flash,
            use_checkpointing=not args.no_checkpoint,
        )
        print_memory_analysis(result, verbose=True)
    else:
        # Default: show comparison
        compare_all_models(args.batch, args.seq)


if __name__ == "__main__":
    main()
