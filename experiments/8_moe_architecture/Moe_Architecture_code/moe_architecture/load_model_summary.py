#!/usr/bin/env python3
"""
MoE Model Summary Script
========================

Loads MoE models and prints parameter summaries for each stage.
Designed for Apple Silicon (M1/M2) Macs using MPS backend.

Usage:
    python load_model_summary.py              # Load 3B MoE (default)
    python load_model_summary.py --config 1b_dense
    python load_model_summary.py --config 8b_moe
    python load_model_summary.py --all        # Show all loadable configs
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn

from configs import get_config, CONFIGS
from model.transformer import MoETransformer, create_model


def get_device():
    """Get the best available device for Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable format."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def get_layer_params(module: nn.Module) -> dict:
    """Get parameter breakdown by layer type."""
    breakdown = {}
    for name, child in module.named_modules():
        if len(list(child.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in child.parameters(recurse=False))
            if params > 0:
                layer_type = type(child).__name__
                if layer_type not in breakdown:
                    breakdown[layer_type] = 0
                breakdown[layer_type] += params
    return breakdown


def print_model_summary(model: MoETransformer, config_name: str):
    """Print detailed model summary."""
    print("\n" + "=" * 70)
    print(f"MODEL SUMMARY: {config_name.upper()}")
    print("=" * 70)
    
    # Basic info
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n📊 PARAMETER COUNTS:")
    print(f"   Total Parameters:     {total_params:>15,} ({format_params(total_params)})")
    print(f"   Trainable Parameters: {trainable_params:>15,} ({format_params(trainable_params)})")
    print(f"   Non-trainable:        {total_params - trainable_params:>15,}")
    
    # Memory estimate (assuming float32)
    memory_fp32 = total_params * 4 / (1024**3)  # GB
    memory_fp16 = total_params * 2 / (1024**3)  # GB
    memory_bf16 = total_params * 2 / (1024**3)  # GB
    
    print(f"\n💾 MEMORY ESTIMATES (model weights only):")
    print(f"   FP32: {memory_fp32:.2f} GB")
    print(f"   FP16/BF16: {memory_fp16:.2f} GB")
    
    # Model configuration
    config = model.config
    print(f"\n⚙️  MODEL CONFIGURATION:")
    print(f"   Model Type:      {config.model_type}")
    print(f"   Hidden Size:     {config.hidden_size}")
    print(f"   Num Layers:      {config.num_layers}")
    print(f"   Num Heads:       {config.attention.num_attention_heads}")
    print(f"   Vocab Size:      {config.tokenizer.vocab_size}")
    print(f"   Max Seq Length:  {config.max_position_embeddings}")
    
    # MoE specific info
    if hasattr(config, 'num_routed_experts') and config.num_routed_experts > 0:
        print(f"\n🔀 MoE CONFIGURATION:")
        print(f"   Routed Experts:  {config.num_routed_experts}")
        print(f"   Shared Experts:  {config.num_shared_experts}")
        print(f"   Null Experts:    {config.num_null_experts}")
        print(f"   Top-K:           {config.router.top_k}")
        print(f"   Intermediate:    {config.expert.intermediate_size}")
    
    # Layer-by-layer breakdown
    print(f"\n📋 PARAMETER BREAKDOWN BY COMPONENT:")
    print("-" * 50)
    
    component_params = {}
    
    # Embeddings
    if hasattr(model, 'embed_tokens'):
        embed_params = count_parameters(model.embed_tokens)
        component_params['Token Embeddings'] = embed_params
        
    # LM Head
    if hasattr(model, 'lm_head'):
        lm_params = count_parameters(model.lm_head)
        component_params['LM Head'] = lm_params
    
    # Layers
    if hasattr(model, 'layers'):
        for i, layer in enumerate(model.layers):
            layer_params = count_parameters(layer)
            if i == 0:
                component_params[f'Transformer Layers (x{len(model.layers)})'] = layer_params * len(model.layers)
            
            # Detailed first layer breakdown (example)
            if i == 0:
                print(f"\n   Layer 0 Breakdown (representative):")
                if hasattr(layer, 'attention'):
                    attn_params = count_parameters(layer.attention)
                    print(f"      Attention:     {attn_params:>12,} ({format_params(attn_params)})")
                if hasattr(layer, 'ffn'):
                    ffn_params = count_parameters(layer.ffn)
                    ffn_type = "MoE Block" if "MoE" in type(layer.ffn).__name__ else "Dense FFN"
                    print(f"      {ffn_type}:  {ffn_params:>12,} ({format_params(ffn_params)})")
                if hasattr(layer, 'input_layernorm'):
                    norm_params = count_parameters(layer.input_layernorm)
                    print(f"      LayerNorms:    {norm_params * 2:>12,} ({format_params(norm_params * 2)})")
    
    # Final norm
    if hasattr(model, 'norm'):
        norm_params = count_parameters(model.norm)
        component_params['Final Norm'] = norm_params
    
    print(f"\n   Component Summary:")
    for name, params in component_params.items():
        pct = (params / total_params) * 100
        print(f"      {name:<35} {params:>12,} ({format_params(params):>8}) [{pct:>5.1f}%]")
    
    print("\n" + "=" * 70)
    
    return total_params


def print_torchsummary_style(model: MoETransformer, config_name: str):
    """Print a torchsummary-style output showing each layer."""
    print("\n" + "=" * 80)
    print(f"LAYER-BY-LAYER SUMMARY: {config_name.upper()}")
    print("=" * 80)
    print(f"{'Layer (type)':<45} {'Output Shape':<20} {'Param #':<15}")
    print("-" * 80)
    
    total_params = 0
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters(recurse=False))
            if params > 0:
                # Truncate long names
                display_name = name if len(name) < 43 else "..." + name[-40:]
                print(f"{display_name:<45} {'--':<20} {params:>12,}")
                total_params += params
    
    print("-" * 80)
    print(f"{'Total Trainable Params:':<45} {'':<20} {total_params:>12,}")
    print("=" * 80)


def load_and_summarize(config_name: str, device: torch.device, detailed: bool = True):
    """Load a model and print its summary."""
    print(f"\n🚀 Loading {config_name} model...")
    print(f"   Device: {device}")
    
    try:
        # Get configuration
        config = get_config(config_name)
        print(f"   Configuration loaded successfully")
        
        # Create model
        model = create_model(config)
        print(f"   Model created successfully")
        
        # Move to device (for MPS, we might want to keep on CPU for large models)
        if device.type == "mps" and config_name in ["70b_moe"]:
            print(f"   ⚠️  Model too large for MPS, keeping on CPU")
            device = torch.device("cpu")
        
        model = model.to(device)
        model.eval()
        print(f"   Model moved to {device}")
        
        # Print summary
        total_params = print_model_summary(model, config_name)
        
        if detailed:
            print_torchsummary_style(model, config_name)
        
        return model, total_params
        
    except Exception as e:
        print(f"   ❌ Error loading {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return None, 0


def main():
    parser = argparse.ArgumentParser(description="Load MoE models and print summaries")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="3b_moe",
        choices=list(CONFIGS.keys()),
        help="Configuration to load (default: 3b_moe)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Load and summarize all available configurations (except 70B)"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        default=True,
        help="Print detailed layer-by-layer summary"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU device (useful for memory-constrained systems)"
    )
    
    args = parser.parse_args()
    
    # Get device
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = get_device()
    
    print(f"\n{'='*70}")
    print("MoE MODEL SUMMARY TOOL")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"PyTorch Version: {torch.__version__}")
    
    if args.all:
        # Load all configs except 70B (too large)
        configs_to_load = ["1b_dense", "3b_moe", "8b_moe"]
        print(f"\nLoading configurations: {configs_to_load}")
        print("(Skipping 70B due to memory constraints)")
        
        results = {}
        for cfg in configs_to_load:
            model, params = load_and_summarize(cfg, device, args.detailed)
            results[cfg] = params
            # Clear model from memory
            if model is not None:
                del model
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
        
        # Summary table
        print("\n\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        print(f"{'Configuration':<20} {'Parameters':<20} {'Memory (FP16)':<15}")
        print("-" * 55)
        for cfg, params in results.items():
            mem_gb = params * 2 / (1024**3)
            print(f"{cfg:<20} {format_params(params):<20} {mem_gb:.2f} GB")
        print("=" * 70)
        
    else:
        # Load single config
        load_and_summarize(args.config, device, args.detailed)


if __name__ == "__main__":
    main()
