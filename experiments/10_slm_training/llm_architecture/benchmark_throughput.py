"""
Throughput Benchmark for LLM Variants
=====================================

Measures tokens/second for inference and training across different sequence lengths.
Supports CUDA, Intel XPU, and CPU backends.

Usage:
    python benchmark_throughput.py configs/1b_gsa.yaml
    python benchmark_throughput.py configs/1b_base.yaml --batch-size 8 --seq-lengths 128,256,512,1024
    python benchmark_throughput.py configs/1b_gsa.yaml --device xpu --output results/gsa_benchmark.json
"""

import sys
import argparse
import time
import json
from datetime import datetime
import torch
from pathlib import Path

# Try importing Intel Extension for PyTorch (optional optimization)
try:
    import intel_extension_for_pytorch as ipex  # type: ignore
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.model_config import ModelConfig
from models.llm import LLM


def _has_xpu_backend() -> bool:
    return hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None))


def _get_device_info(device: str) -> dict:
    info: dict = {"device": device, "torch_version": torch.__version__}
    if HAS_IPEX:
        try:
            import intel_extension_for_pytorch as ipex_local  # type: ignore

            info["ipex_version"] = getattr(ipex_local, "__version__", "")
        except Exception:
            pass

    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_memory_bytes": int(props.total_memory),
                "major": int(getattr(props, "major", 0)),
                "minor": int(getattr(props, "minor", 0)),
                "multi_processor_count": int(getattr(props, "multi_processor_count", 0)),
                "cuda_version": torch.version.cuda,
            }
        )
        return info

    if device == "xpu" and _has_xpu_backend() and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        info.update(
            {
                "name": torch.xpu.get_device_name(0),
                "total_memory_bytes": int(getattr(props, "total_memory", 0)),
                "max_compute_units": int(getattr(props, "max_compute_units", 0)),
                "gpu_eu_count": int(getattr(props, "gpu_eu_count", 0)),
                "gpu_subslice_count": int(getattr(props, "gpu_subslice_count", 0)),
                "driver_version": str(getattr(props, "driver_version", "")),
                "platform_name": str(getattr(props, "platform_name", "")),
                "vendor": str(getattr(props, "vendor", "")),
                "version": str(getattr(props, "version", "")),
                "has_fp16": bool(getattr(props, "has_fp16", False)),
                "has_fp64": bool(getattr(props, "has_fp64", False)),
                "sub_group_sizes": list(getattr(props, "sub_group_sizes", [])),
                "max_work_group_size": int(getattr(props, "max_work_group_size", 0)),
                "max_num_sub_groups": int(getattr(props, "max_num_sub_groups", 0)),
            }
        )
        return info

    return info


def _apply_benchmark_profile(config: ModelConfig, profile: str | None) -> ModelConfig:
    """Scale model down for throughput testing without editing YAML.

    Keeps architecture choices (attention type, connection type, position type, MTP) but
    reduces sizes to fit constrained devices.
    
    Profiles:
        micro: For 4-8k context testing on limited memory (256 hidden, 4 layers)
        tiny:  Standard small model testing (512 hidden, 6 layers)
        small: Larger scale testing (1024 hidden, 12 layers)
    """
    if not profile:
        return config

    profile = profile.lower().strip()
    if profile not in {"micro", "tiny", "small"}:
        raise ValueError("Unsupported profile. Use one of: micro, tiny, small")

    if profile == "micro":
        config.model_name = f"{config.model_name} (micro)"
        config.hidden_size = 256
        config.num_hidden_layers = 4
        config.max_position_embeddings = 16384  # Support up to 16k context
        config.attention.num_attention_heads = 4
        config.attention.num_key_value_heads = 1
        config.ffn.intermediate_size = 1024
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 1.5)
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 2)

    if profile == "tiny":
        config.model_name = f"{config.model_name} (tiny)"
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.max_position_embeddings = max(2048, min(config.max_position_embeddings, 4096))
        config.attention.num_attention_heads = 8
        config.attention.num_key_value_heads = 2
        config.ffn.intermediate_size = 2048
        # mHC can explode params on some configs; keep it enabled but reduce cost.
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 2.0)
        # Keep MTP enabled if requested, but smaller is safer.
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 2)

    if profile == "small":
        config.model_name = f"{config.model_name} (small)"
        config.hidden_size = 1024
        config.num_hidden_layers = 12
        config.max_position_embeddings = max(4096, min(config.max_position_embeddings, 8192))
        config.attention.num_attention_heads = 16
        config.attention.num_key_value_heads = 4
        config.ffn.intermediate_size = 4096
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 2.0)
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 2)

    # Ensure head_dim & other derived values are consistent
    config.__post_init__()
    return config


def benchmark_throughput(
    config_path: str,
    batch_size: int = 4,
    seq_lengths: list = [128, 256, 512],
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
    dtype: str = "float32",
    compile_model: bool = False,
    device_override: str = None,
    output_path: str = None,
    inference_only: bool = False,
    profile: str | None = None,
):
    """Run throughput benchmark for a model configuration."""
    
    # Load config
    config = ModelConfig.load(config_path)
    config = _apply_benchmark_profile(config, profile)
    
    print("=" * 70)
    print(f"THROUGHPUT BENCHMARK: {config.model_name}")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Attention: {config.attention.attention_type.value}")
    print(f"Position: {config.position.position_type.value}")
    print(f"Connection: {config.connection.connection_type.value}")
    print(f"MTP: {config.head.use_multi_token_prediction}")
    print(f"Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")
    print()
    
    # Device setup - support CUDA, XPU, or CPU
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
    elif _has_xpu_backend() and torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"
    
    device_info = _get_device_info(device)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {device_info.get('name', '')}")
        print(f"CUDA Version: {device_info.get('cuda_version', '')}")
    elif device == "xpu":
        print(f"XPU: {device_info.get('name', '')}")
        if HAS_IPEX:
            import intel_extension_for_pytorch as ipex_local  # type: ignore
            print(f"Intel Extension for PyTorch: {ipex_local.__version__}")
    
    # Dtype setup
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float32)
    print(f"Dtype: {dtype}")
    print()
    
    # Create model
    model = LLM(config).to(device).to(torch_dtype)
    
    # Optional: torch.compile for PyTorch 2.0+
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
    
    # Optimize for XPU if available
    if device == 'xpu' and HAS_IPEX:
        import intel_extension_for_pytorch as ipex_opt  # type: ignore
        model = ipex_opt.optimize(model, dtype=torch_dtype)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print()
    
    # Memory info helper
    def get_peak_memory_gb():
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        if device == "xpu" and _has_xpu_backend():
            return torch.xpu.max_memory_allocated() / 1e9
        return 0
    
    def reset_memory_stats():
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device == "xpu" and _has_xpu_backend():
            torch.xpu.reset_peak_memory_stats()
    
    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "xpu" and _has_xpu_backend():
            torch.xpu.synchronize()
    
    # Initial memory info
    reset_memory_stats()
    print(f"Peak Memory (GB): {get_peak_memory_gb():.2f}")
    
    print()
    print("-" * 70)
    print("INFERENCE THROUGHPUT (Forward Pass Only)")
    print("-" * 70)
    print(f"{'Seq Len':<10} {'Tokens/s':<15} {'Samples/s':<15} {'ms/sample':<15} {'Memory (GB)':<12}")
    print("-" * 70)
    
    model.eval()
    inference_results = []
    
    for seq_len in seq_lengths:
        reset_memory_stats()
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(input_ids)
        
        synchronize()
        
        # Benchmark
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(benchmark_iters):
                _ = model(input_ids)
        
        synchronize()
        
        elapsed = time.perf_counter() - start
        
        total_tokens = batch_size * seq_len * benchmark_iters
        tokens_per_sec = total_tokens / elapsed
        samples_per_sec = (batch_size * benchmark_iters) / elapsed
        ms_per_sample = (elapsed / (batch_size * benchmark_iters)) * 1000

        mem_gb = get_peak_memory_gb()
        
        print(f"{seq_len:<10} {tokens_per_sec:<15,.0f} {samples_per_sec:<15.2f} {ms_per_sample:<15.1f} {mem_gb:<12.2f}")
        
        inference_results.append({
            'seq_len': seq_len,
            'tokens_per_sec': tokens_per_sec,
            'samples_per_sec': samples_per_sec,
            'ms_per_sample': ms_per_sample,
            'memory_gb': mem_gb,
        })
    
    training_results = []
    
    if not inference_only:
        print()
        print("-" * 70)
        print("TRAINING THROUGHPUT (Forward + Backward)")
        print("-" * 70)
        print(f"{'Seq Len':<10} {'Tokens/s':<15} {'Samples/s':<15} {'ms/sample':<15} {'Memory (GB)':<12}")
        print("-" * 70)
        
        model.train()
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
            
            reset_memory_stats()
            
            # Warmup
            for _ in range(warmup_iters):
                output = model(input_ids, labels=labels)
                output.loss.backward()
                model.zero_grad()
            
            synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(benchmark_iters):
                output = model(input_ids, labels=labels)
                output.loss.backward()
                model.zero_grad()
            
            synchronize()
            
            elapsed = time.perf_counter() - start
            
            total_tokens = batch_size * seq_len * benchmark_iters
            tokens_per_sec = total_tokens / elapsed
            samples_per_sec = (batch_size * benchmark_iters) / elapsed
            ms_per_sample = (elapsed / (batch_size * benchmark_iters)) * 1000

            mem_gb = get_peak_memory_gb()
            
            print(f"{seq_len:<10} {tokens_per_sec:<15,.0f} {samples_per_sec:<15.2f} {ms_per_sample:<15.1f} {mem_gb:<12.2f}")
            
            training_results.append({
                'seq_len': seq_len,
                'tokens_per_sec': tokens_per_sec,
                'samples_per_sec': samples_per_sec,
                'ms_per_sample': ms_per_sample,
                'memory_gb': mem_gb,
            })
    else:
        print()
        print("(Training benchmark skipped - inference only mode)")
    
    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config_path': config_path,
        'config_name': config.model_name,
        'device': device,
        'dtype': dtype,
        'batch_size': batch_size,
        'profile': profile,
        'parameters': total_params,
        'parameters_billions': total_params / 1e9,
        'device_info': device_info,
        'model_settings': {
            'attention': config.attention.attention_type.value,
            'position': config.position.position_type.value,
            'connection': config.connection.connection_type.value,
            'mtp': config.head.use_multi_token_prediction,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
        },
        'inference': inference_results,
        'training': training_results,
    }
    
    # Save results to file if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark LLM throughput')
    parser.add_argument('config', type=str, help='Path to YAML config file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--seq-lengths', type=str, default='128,256,512', 
                        help='Comma-separated sequence lengths (default: 128,256,512)')
    parser.add_argument('--warmup', type=int, default=3, help='Warmup iterations (default: 3)')
    parser.add_argument('--iters', type=int, default=10, help='Benchmark iterations (default: 10)')
    parser.add_argument('--dtype', type=str, default='float32', 
                        choices=['float32', 'float16', 'bfloat16'], help='Data type')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--device', type=str, default=None, 
                        choices=['cuda', 'xpu', 'cpu'], help='Device override (auto-detect if not set)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path for results (e.g., results/benchmark.json)')
    parser.add_argument('--inference-only', action='store_true',
                        help='Skip training benchmark (useful for memory-constrained devices)')
    parser.add_argument('--profile', type=str, default=None, choices=['micro', 'tiny', 'small'],
                        help='Scale model down for throughput testing (micro=4-8k context, tiny, small)')
    
    args = parser.parse_args()
    
    seq_lengths = [int(x) for x in args.seq_lengths.split(',')]
    
    benchmark_throughput(
        config_path=args.config,
        batch_size=args.batch_size,
        seq_lengths=seq_lengths,
        warmup_iters=args.warmup,
        benchmark_iters=args.iters,
        dtype=args.dtype,
        compile_model=args.compile,
        device_override=args.device,
        output_path=args.output,
        inference_only=args.inference_only,
        profile=args.profile,
    )


if __name__ == '__main__':
    main()
