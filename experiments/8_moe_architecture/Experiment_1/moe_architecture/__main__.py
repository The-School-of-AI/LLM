#!/usr/bin/env python3
"""
MoE Architecture Main Entry Point
==================================

Run the MoE model with different configurations.

Usage:
    # Show configuration summary
    python -m moe_architecture --config 3b_moe --info
    
    # Create and test model
    python -m moe_architecture --config 3b_moe --test
    
    # Run forward pass with sample input
    python -m moe_architecture --config 3b_moe --run
    
    # Export model to checkpoint
    python -m moe_architecture --config 3b_moe --export model.pt

Available Configurations:
    - 1b_dense: Stage 1 - 1B Dense Foundation
    - 3b_moe: Stage 2 - 3B MoE-8 (Learn Routing)
    - 8b_moe: Stage 3 - 8B MoE-8 (Scale Dimensions)
    - 70b_moe: Stage 4 - 70B MoE-64 (Expert Expansion)
"""

import argparse
import torch
import time
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from moe_architecture.config import get_config, print_config_summary
from moe_architecture.model.transformer import MoETransformer, create_model
from moe_architecture.utils.model_utils import (
    count_parameters,
    print_parameter_summary,
    save_checkpoint,
    verify_lossless_init,
)
from moe_architecture.utils.telemetry import create_default_telemetry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MoE Architecture - Team 8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='3b_moe',
        choices=['1b_dense', '3b_moe', '8b_moe', '70b_moe'],
        help='Model configuration to use'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show configuration summary'
    )
    
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run basic model tests'
    )
    
    parser.add_argument(
        '--run', '-r',
        action='store_true',
        help='Run forward pass with sample input'
    )
    
    parser.add_argument(
        '--export', '-e',
        type=str,
        help='Export model checkpoint to path'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type for model'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=2,
        help='Batch size for testing'
    )
    
    parser.add_argument(
        '--seq-length', '-s',
        type=int,
        default=128,
        help='Sequence length for testing'
    )
    
    return parser.parse_args()


def get_dtype(dtype_str: str):
    """Get torch dtype from string."""
    return {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }[dtype_str]


def show_info(config):
    """Show configuration information."""
    print_config_summary(config)


def run_tests(model, config, device, dtype, batch_size, seq_length):
    """Run basic model tests."""
    print("\n" + "="*60)
    print(" Running Model Tests")
    print("="*60)
    
    # Test 1: Forward pass
    print("\n📋 Test 1: Forward Pass")
    try:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs['logits']
        print(f"   ✓ Input shape: {input_ids.shape}")
        print(f"   ✓ Output shape: {logits.shape}")
        print(f"   ✓ Output dtype: {logits.dtype}")
        print(f"   ✓ Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test 2: Router info
    if hasattr(config, 'expert') and config.expert.num_routed_experts > 0:
        print("\n📋 Test 2: Router Information")
        try:
            outputs = model(input_ids, return_router_info=True)
            
            if 'router_info' in outputs and outputs['router_info']:
                router_info = outputs['router_info'][0]  # First layer
                print(f"   ✓ Top-K used: {router_info.get('top_k_used', 'N/A')}")
                print(f"   ✓ Score mean: {router_info.get('score_mean', 0):.4f}")
                print(f"   ✓ Score std: {router_info.get('score_std', 0):.4f}")
                if 'null_rate' in router_info:
                    print(f"   ✓ Null rate: {router_info.get('overall_null_rate', 0):.1%}")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
    
    # Test 3: Memory usage
    print("\n📋 Test 3: Memory Usage")
    if device == 'cuda':
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   ✓ Allocated: {allocated:.2f} GB")
        print(f"   ✓ Reserved: {reserved:.2f} GB")
    else:
        print("   ℹ CPU mode - skipping GPU memory check")
    
    # Test 4: Speed benchmark
    print("\n📋 Test 4: Speed Benchmark")
    try:
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_ids)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        tokens_per_sec = (batch_size * seq_length * num_runs) / elapsed
        ms_per_token = (elapsed / (batch_size * seq_length * num_runs)) * 1000
        
        print(f"   ✓ Time per batch: {elapsed/num_runs*1000:.1f} ms")
        print(f"   ✓ Tokens/second: {tokens_per_sec:,.0f}")
        print(f"   ✓ ms/token: {ms_per_token:.3f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "="*60)
    print(" All Tests Completed!")
    print("="*60)
    
    return True


def run_forward(model, config, device, batch_size, seq_length):
    """Run a forward pass and display results."""
    print("\n" + "="*60)
    print(" Running Forward Pass")
    print("="*60)
    
    # Create sample input
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
    
    # Add some padding to test null routing
    input_ids[:, -10:] = 0  # Padding tokens
    
    print(f"\n📥 Input:")
    print(f"   Shape: {input_ids.shape}")
    print(f"   Device: {input_ids.device}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, return_router_info=True)
    
    # Display outputs
    print(f"\n📤 Outputs:")
    print(f"   Logits shape: {outputs['logits'].shape}")
    
    if 'router_info' in outputs and outputs['router_info']:
        print(f"\n🔀 Routing Info (first MoE layer):")
        info = outputs['router_info'][0]
        for key, value in info.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    
    # Generate a token
    logits = outputs['logits'][:, -1, :]  # Last position
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(probs, dim=-1)
    
    print(f"\n🔮 Prediction:")
    print(f"   Next token IDs: {next_token.tolist()}")
    print(f"   Confidence: {probs.max(dim=-1).values.tolist()}")


def export_model(model, config, path, step=0):
    """Export model to checkpoint."""
    print(f"\n💾 Exporting model to {path}...")
    save_checkpoint(model, None, config, step, path)
    print(f"   ✓ Saved!")
    print(f"   ✓ File size: {Path(path).stat().st_size / 1e6:.1f} MB")


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "="*60)
    print(" MoE Architecture - Team 8")
    print("="*60)
    print(f"\n Configuration: {args.config}")
    print(f" Device: {args.device}")
    print(f" Dtype: {args.dtype}")
    
    # Load configuration
    config = get_config(args.config)
    
    # Show info if requested
    if args.info:
        show_info(config)
        if not (args.test or args.run or args.export):
            return
    
    # Create model
    print("\n🔧 Creating model...")
    dtype = get_dtype(args.dtype)
    
    try:
        model = create_model(config)
        model = model.to(args.device)
        if dtype != torch.float32:
            model = model.to(dtype)
        model.eval()
        
        # Print parameter summary
        print_parameter_summary(model, config.model_name)
        
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests if requested
    if args.test:
        run_tests(model, config, args.device, dtype, args.batch_size, args.seq_length)
    
    # Run forward pass if requested
    if args.run:
        run_forward(model, config, args.device, args.batch_size, args.seq_length)
    
    # Export if requested
    if args.export:
        export_model(model, config, args.export)
    
    print("\n✅ Done!\n")


if __name__ == '__main__':
    main()
