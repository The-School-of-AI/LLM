#!/usr/bin/env python3
"""
MoE FLOPs & Sparsity Calculator
===============================

Calculate training FLOPs, memory requirements, and sparsity metrics
for MoE models across all growth stages.

Usage:
    python flops_calculator.py --model 3b_moe
    python flops_calculator.py --all
    python flops_calculator.py --custom --params 70e9 --active 12e9 --tokens 2e12
"""

import argparse
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class ModelSpec:
    """Model specification for calculations."""
    name: str
    total_params: float          # Total parameters
    active_params: float         # Active parameters per forward
    training_tokens: float       # Target training tokens
    
    # Architecture details
    hidden_size: int = 2048
    num_layers: int = 24
    num_routed_experts: int = 8
    num_shared_experts: int = 2
    num_null_experts: int = 1
    top_k: int = 2
    intermediate_size: int = 5504


# Pre-defined model specifications
MODEL_SPECS = {
    "1b_dense": ModelSpec(
        name="1B Dense",
        total_params=1.0e9,
        active_params=1.0e9,
        training_tokens=100e9,
        num_routed_experts=0,
        num_shared_experts=0,
        num_null_experts=0,
        top_k=0,
    ),
    "3b_moe": ModelSpec(
        name="3B MoE-8",
        total_params=3.0e9,
        active_params=1.2e9,
        training_tokens=500e9,
        num_routed_experts=8,
        num_shared_experts=2,
        num_null_experts=1,
        top_k=2,
    ),
    "8b_moe": ModelSpec(
        name="8B MoE-8",
        total_params=8.0e9,
        active_params=3.2e9,
        training_tokens=1e12,
        hidden_size=4096,
        num_layers=48,
        num_routed_experts=8,
        num_shared_experts=2,
        num_null_experts=1,
        top_k=2,
        intermediate_size=11008,
    ),
    "70b_moe": ModelSpec(
        name="70B MoE-64",
        total_params=70e9,
        active_params=12e9,
        training_tokens=2e12,
        hidden_size=4096,
        num_layers=80,
        num_routed_experts=64,
        num_shared_experts=4,
        num_null_experts=2,
        top_k=4,
        intermediate_size=11008,
    ),
}


# Sparse variants
SPARSE_SPECS = {
    "3b_moe_sparse": ModelSpec(
        name="3B MoE-8 (Sparse)",
        total_params=3.0e9,
        active_params=0.6e9,  # top_k=1, shared=1
        training_tokens=500e9,
        num_routed_experts=8,
        num_shared_experts=1,
        num_null_experts=1,
        top_k=1,
    ),
    "8b_moe_sparse": ModelSpec(
        name="8B MoE-8 (Sparse)",
        total_params=8.0e9,
        active_params=1.6e9,  # top_k=1, shared=1
        training_tokens=1e12,
        hidden_size=4096,
        num_layers=48,
        num_routed_experts=8,
        num_shared_experts=1,
        num_null_experts=1,
        top_k=1,
    ),
    "70b_moe_sparse": ModelSpec(
        name="70B MoE-64 (Sparse)",
        total_params=70e9,
        active_params=6e9,  # top_k=2, shared=2
        training_tokens=2e12,
        hidden_size=4096,
        num_layers=80,
        num_routed_experts=64,
        num_shared_experts=2,
        num_null_experts=2,
        top_k=2,
    ),
}


def calculate_training_flops(active_params: float, tokens: float) -> float:
    """
    Calculate total training FLOPs.
    
    Formula: FLOPs = 6 × N_active × D
    - 6 = forward (2N) + backward (4N)
    - N_active = active parameters
    - D = training tokens
    
    Args:
        active_params: Active parameters per forward pass
        tokens: Total training tokens
        
    Returns:
        Total FLOPs for training
    """
    return 6 * active_params * tokens


def calculate_inference_flops_per_token(active_params: float) -> float:
    """
    Calculate FLOPs per token for inference.
    
    Formula: FLOPs = 2 × N_active (forward only)
    """
    return 2 * active_params


def calculate_sparsity(total_params: float, active_params: float) -> float:
    """
    Calculate model sparsity.
    
    Sparsity = 1 - (Active / Total)
    """
    return 1 - (active_params / total_params)


def calculate_active_ratio(total_params: float, active_params: float) -> float:
    """Calculate active parameter ratio."""
    return active_params / total_params


def calculate_training_memory(total_params: float, precision: str = "bf16") -> float:
    """
    Calculate training memory requirements.
    
    For bf16 training with AdamW:
    - Model: 2 bytes/param
    - Gradients: 2 bytes/param
    - Optimizer states: 8 bytes/param (m + v in fp32)
    Total: 12 bytes/param
    
    Args:
        total_params: Total model parameters
        precision: Training precision (bf16, fp16, fp32)
        
    Returns:
        Memory in bytes
    """
    if precision == "bf16" or precision == "fp16":
        bytes_per_param = 12  # 2 + 2 + 8 (Adam)
    elif precision == "fp32":
        bytes_per_param = 16  # 4 + 4 + 8
    else:
        bytes_per_param = 12
    
    return total_params * bytes_per_param


def calculate_inference_memory(total_params: float, precision: str = "bf16") -> float:
    """
    Calculate inference memory requirements.
    
    Just model weights loaded.
    """
    if precision == "bf16" or precision == "fp16":
        bytes_per_param = 2
    elif precision == "fp32":
        bytes_per_param = 4
    else:
        bytes_per_param = 2
    
    return total_params * bytes_per_param


def calculate_gpu_hours(
    flops: float,
    gpu_tflops: float = 312,  # A100 bf16 peak
    utilization: float = 0.4   # Typical LLM training utilization
) -> float:
    """
    Calculate GPU hours required for training.
    
    Args:
        flops: Total training FLOPs
        gpu_tflops: GPU theoretical peak TFLOPS
        utilization: Actual/theoretical utilization (30-50% typical)
        
    Returns:
        GPU hours
    """
    effective_flops_per_sec = gpu_tflops * 1e12 * utilization
    seconds = flops / effective_flops_per_sec
    hours = seconds / 3600
    return hours


def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable form."""
    if flops >= 1e21:
        return f"{flops / 1e21:.2f} ZFLOPs"
    elif flops >= 1e18:
        return f"{flops / 1e18:.2f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    else:
        return f"{flops:.2e} FLOPs"


def format_memory(bytes_val: float) -> str:
    """Format bytes in human-readable form."""
    if bytes_val >= 1e12:
        return f"{bytes_val / 1e12:.1f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val / 1e9:.1f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val / 1e6:.1f} MB"
    else:
        return f"{bytes_val:.0f} B"


def format_params(params: float) -> str:
    """Format parameter count."""
    if params >= 1e12:
        return f"{params / 1e12:.1f}T"
    elif params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    else:
        return f"{params:.0f}"


def format_tokens(tokens: float) -> str:
    """Format token count."""
    if tokens >= 1e12:
        return f"{tokens / 1e12:.1f}T"
    elif tokens >= 1e9:
        return f"{tokens / 1e9:.0f}B"
    elif tokens >= 1e6:
        return f"{tokens / 1e6:.0f}M"
    else:
        return f"{tokens:.0f}"


def analyze_model(spec: ModelSpec, verbose: bool = True) -> dict:
    """
    Perform complete analysis of a model specification.
    
    Args:
        spec: Model specification
        verbose: Print results
        
    Returns:
        Dictionary with all metrics
    """
    # Core calculations
    training_flops = calculate_training_flops(spec.active_params, spec.training_tokens)
    sparsity = calculate_sparsity(spec.total_params, spec.active_params)
    active_ratio = calculate_active_ratio(spec.total_params, spec.active_params)
    
    # Memory
    train_memory = calculate_training_memory(spec.total_params)
    infer_memory = calculate_inference_memory(spec.total_params)
    
    # GPU time
    gpu_hours_a100 = calculate_gpu_hours(training_flops)
    gpu_days_a100 = gpu_hours_a100 / 24
    
    # Cluster estimates (64 GPUs)
    cluster_days_64 = gpu_days_a100 / 64
    
    results = {
        "name": spec.name,
        "total_params": spec.total_params,
        "active_params": spec.active_params,
        "training_tokens": spec.training_tokens,
        "training_flops": training_flops,
        "sparsity": sparsity,
        "active_ratio": active_ratio,
        "train_memory_bytes": train_memory,
        "infer_memory_bytes": infer_memory,
        "gpu_hours_a100": gpu_hours_a100,
        "gpu_days_a100": gpu_days_a100,
        "cluster_days_64gpu": cluster_days_64,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" {spec.name} Analysis")
        print(f"{'='*60}")
        
        print(f"\n📊 Parameters:")
        print(f"   Total:  {format_params(spec.total_params)}")
        print(f"   Active: {format_params(spec.active_params)}")
        print(f"   Active Ratio: {active_ratio*100:.1f}%")
        print(f"   Sparsity: {sparsity*100:.1f}%")
        
        if spec.num_routed_experts > 0:
            print(f"\n🎯 MoE Configuration:")
            print(f"   Routed Experts: {spec.num_routed_experts}")
            print(f"   Shared Experts: {spec.num_shared_experts}")
            print(f"   Null Experts: {spec.num_null_experts}")
            print(f"   Top-K: {spec.top_k}")
        
        print(f"\n⚡ Training:")
        print(f"   Tokens: {format_tokens(spec.training_tokens)}")
        print(f"   FLOPs: {format_flops(training_flops)}")
        
        print(f"\n💾 Memory:")
        print(f"   Training (bf16+Adam): {format_memory(train_memory)}")
        print(f"   Inference (bf16): {format_memory(infer_memory)}")
        
        print(f"\n⏱️ GPU Time (A100, 40% util):")
        print(f"   Single GPU: {gpu_hours_a100:,.0f} hours ({gpu_days_a100:,.0f} days)")
        print(f"   64× A100 Cluster: {cluster_days_64:.1f} days")
        
        print(f"\n{'='*60}\n")
    
    return results


def compare_sparsity_options(base_spec: ModelSpec):
    """
    Compare different sparsity configurations for a model.
    """
    print(f"\n{'='*70}")
    print(f" Sparsity Options for {base_spec.name}")
    print(f"{'='*70}")
    
    configs = [
        ("Base", base_spec.top_k, base_spec.num_shared_experts),
        ("Medium Sparse", max(1, base_spec.top_k - 1), max(1, base_spec.num_shared_experts - 1)),
        ("High Sparse", 1, 1),
    ]
    
    # Calculate expert size
    expert_size = 3 * base_spec.hidden_size * base_spec.intermediate_size
    attention_size = base_spec.hidden_size * base_spec.hidden_size * 4  # Approx
    
    print(f"\n{'Config':<15} {'top_k':<6} {'shared':<7} {'Active Exp':<12} {'Sparsity':<10} {'FLOPs Ratio':<12}")
    print("-" * 70)
    
    base_active = base_spec.num_shared_experts + base_spec.top_k
    
    for name, top_k, shared in configs:
        active_experts = shared + top_k
        
        # Recalculate active params
        active_params = (
            base_spec.total_params * 0.05 +  # Embeddings + attention (rough)
            active_experts * expert_size * base_spec.num_layers
        )
        
        sparsity = 1 - (active_params / base_spec.total_params)
        flops_ratio = active_experts / base_active
        
        print(f"{name:<15} {top_k:<6} {shared:<7} {active_experts:<12} {sparsity*100:>6.1f}%    {flops_ratio:>6.2f}×")
    
    print(f"\n{'='*70}\n")


def calculate_flops_reduction_options():
    """Show FLOPs reduction options across all models."""
    print(f"\n{'='*80}")
    print(" FLOPs Reduction Options Summary")
    print(f"{'='*80}")
    
    print(f"\n{'Model':<15} {'Config':<15} {'Active':<10} {'FLOPs':<15} {'vs Base':<10}")
    print("-" * 80)
    
    for model_key in ["3b_moe", "8b_moe", "70b_moe"]:
        base = MODEL_SPECS[model_key]
        sparse_key = f"{model_key}_sparse"
        sparse = SPARSE_SPECS.get(sparse_key)
        
        # Base config
        base_flops = calculate_training_flops(base.active_params, base.training_tokens)
        print(f"{base.name:<15} {'Base':<15} {format_params(base.active_params):<10} {format_flops(base_flops):<15} {'1.0×':<10}")
        
        # Sparse config
        if sparse:
            sparse_flops = calculate_training_flops(sparse.active_params, sparse.training_tokens)
            ratio = sparse_flops / base_flops
            print(f"{'':<15} {'Sparse':<15} {format_params(sparse.active_params):<10} {format_flops(sparse_flops):<15} {ratio:.2f}×")
        
        print()
    
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='MoE FLOPs & Sparsity Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flops_calculator.py --model 3b_moe
  python flops_calculator.py --all
  python flops_calculator.py --compare 70b_moe
  python flops_calculator.py --custom --params 70e9 --active 12e9 --tokens 2e12
"""
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(MODEL_SPECS.keys()) + list(SPARSE_SPECS.keys()),
        help='Model to analyze'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Analyze all models'
    )
    
    parser.add_argument(
        '--compare',
        type=str,
        choices=list(MODEL_SPECS.keys()),
        help='Compare sparsity options for a model'
    )
    
    parser.add_argument(
        '--reduction',
        action='store_true',
        help='Show FLOPs reduction options'
    )
    
    parser.add_argument(
        '--custom',
        action='store_true',
        help='Custom model calculation'
    )
    
    parser.add_argument('--params', type=float, help='Total parameters (e.g., 70e9)')
    parser.add_argument('--active', type=float, help='Active parameters (e.g., 12e9)')
    parser.add_argument('--tokens', type=float, help='Training tokens (e.g., 2e12)')
    
    args = parser.parse_args()
    
    print("\n" + "🧮 " * 20)
    print("  MoE FLOPs & Sparsity Calculator")
    print("  Team 8 - Expert Expansion & Routing")
    print("🧮 " * 20)
    
    if args.all:
        # Analyze all models
        print("\n" + "="*60)
        print(" ALL MODELS SUMMARY")
        print("="*60)
        
        results = []
        for spec in MODEL_SPECS.values():
            r = analyze_model(spec, verbose=False)
            results.append(r)
        
        # Print summary table
        print(f"\n{'Model':<15} {'Total':<8} {'Active':<8} {'Sparsity':<10} {'FLOPs':<12} {'A100 Days':<12}")
        print("-" * 75)
        for r in results:
            print(f"{r['name']:<15} {format_params(r['total_params']):<8} {format_params(r['active_params']):<8} "
                  f"{r['sparsity']*100:>6.1f}%   {format_flops(r['training_flops']):<12} {r['gpu_days_a100']:>8,.0f}")
        
        print("\n\n📊 DETAILED ANALYSIS:")
        for spec in MODEL_SPECS.values():
            analyze_model(spec, verbose=True)
    
    elif args.model:
        # Analyze specific model
        all_specs = {**MODEL_SPECS, **SPARSE_SPECS}
        spec = all_specs[args.model]
        analyze_model(spec, verbose=True)
    
    elif args.compare:
        # Compare sparsity options
        spec = MODEL_SPECS[args.compare]
        analyze_model(spec, verbose=True)
        compare_sparsity_options(spec)
    
    elif args.reduction:
        # Show reduction options
        calculate_flops_reduction_options()
    
    elif args.custom:
        # Custom calculation
        if not all([args.params, args.active, args.tokens]):
            parser.error("--custom requires --params, --active, and --tokens")
        
        spec = ModelSpec(
            name="Custom Model",
            total_params=args.params,
            active_params=args.active,
            training_tokens=args.tokens,
        )
        analyze_model(spec, verbose=True)
    
    else:
        # Default: show summary
        print("\nUse --help for options, or --all to analyze all models.\n")
        
        # Quick summary
        print("Quick Summary:")
        print(f"\n{'Model':<15} {'FLOPs':<15} {'Sparsity':<10}")
        print("-" * 45)
        for spec in MODEL_SPECS.values():
            flops = calculate_training_flops(spec.active_params, spec.training_tokens)
            sparsity = calculate_sparsity(spec.total_params, spec.active_params)
            print(f"{spec.name:<15} {format_flops(flops):<15} {sparsity*100:.0f}%")


if __name__ == "__main__":
    main()
