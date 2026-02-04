"""
SOTA Throughput Benchmark for LLM Variants
==========================================

Comprehensive benchmarking suite for LLM architecture variants with deep metrics,
statistical analysis, and actionable insights.

Features:
---------
- Multi-backend support: CUDA, Intel XPU, MPS, CPU
- Comprehensive metrics: Throughput, Latency, Memory, FLOPs, Efficiency
- Statistical analysis: Mean, Std, P50/P95/P99 latencies
- Architecture-aware profiling: Attention patterns, memory breakdown
- Comparative analysis across configurations
- Memory efficiency analysis (KV cache, activation memory)
- Auto-generated insights and recommendations

Usage:
------
    # Single benchmark
    python benchmark_throughput.py configs/1b_gsa.yaml
    
    # Extended benchmark with all metrics
    python benchmark_throughput.py configs/1b_gsa.yaml --full-analysis
    
    # Comparative benchmark
    python benchmark_throughput.py configs/1b_base.yaml configs/1b_gsa.yaml configs/1b_mhc.yaml --compare
    
    # Memory-constrained profiling
    python benchmark_throughput.py configs/1b_deepseek_gsa.yaml --profile micro --device mps
    
    # Export for visualization
    python benchmark_throughput.py configs/1b_full.yaml --output results/benchmark.json --format detailed
"""

import sys
import os
import argparse
import time
import json
import math
import statistics
import gc
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn

# Try importing Intel Extension for PyTorch (optional optimization)
try:
    import intel_extension_for_pytorch as ipex  # type: ignore
    HAS_IPEX = True
except ImportError:
    HAS_IPEX = False

# Try importing for advanced profiling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try importing psutil for CPU memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.model_config import (
    ModelConfig,
    AttentionType,
    PositionEmbeddingType,
    ConnectionType,
    FFNType
)
from models.llm import LLM


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class LatencyStats:
    """Statistical breakdown of latency measurements."""
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


@dataclass 
class MemoryMetrics:
    """Memory usage breakdown."""
    peak_allocated_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    model_size_gb: float = 0.0
    activation_memory_gb: float = 0.0
    kv_cache_gb: float = 0.0
    gradient_memory_gb: float = 0.0
    memory_efficiency: float = 0.0  # useful / total


@dataclass
class ThroughputMetrics:
    """Throughput measurements."""
    tokens_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    tflops_achieved: float = 0.0
    mfu: float = 0.0  # Model FLOPs Utilization


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single configuration."""
    seq_len: int = 0
    batch_size: int = 0
    
    # Core metrics
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    latency: LatencyStats = field(default_factory=LatencyStats)
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    
    # Architecture-specific
    attention_efficiency: float = 0.0
    ffn_efficiency: float = 0.0
    embedding_efficiency: float = 0.0


@dataclass
class ModelProfile:
    """Complete model profiling information."""
    name: str = ""
    total_params: int = 0
    trainable_params: int = 0
    params_billions: float = 0.0
    
    # Architecture breakdown
    embedding_params: int = 0
    attention_params: int = 0
    ffn_params: int = 0
    head_params: int = 0
    norm_params: int = 0
    connection_params: int = 0  # For mHC
    
    # FLOPs estimation
    forward_flops: int = 0
    backward_flops: int = 0
    total_training_flops: int = 0


@dataclass
class InsightReport:
    """Generated insights and recommendations."""
    bottleneck: str = ""
    memory_recommendation: str = ""
    throughput_recommendation: str = ""
    attention_insight: str = ""
    comparison_summary: str = ""
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Device Utilities
# =============================================================================

def _has_xpu_backend() -> bool:
    """Check if Intel XPU backend is available."""
    return hasattr(torch, "xpu") and callable(getattr(torch.xpu, "is_available", None))


def _has_mps_backend() -> bool:
    """Check if Apple MPS backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(device_override: Optional[str] = None) -> str:
    """Get the best available device."""
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    if _has_xpu_backend() and torch.xpu.is_available():
        return "xpu"
    if _has_mps_backend():
        return "mps"
    return "cpu"


def get_device_info(device: str) -> Dict[str, Any]:
    """Get comprehensive device information."""
    info: Dict[str, Any] = {
        "device": device,
        "torch_version": torch.__version__,
        "has_ipex": HAS_IPEX,
    }
    
    if HAS_IPEX:
        try:
            import intel_extension_for_pytorch as ipex_local
            info["ipex_version"] = getattr(ipex_local, "__version__", "")
        except Exception:
            pass

    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "name": torch.cuda.get_device_name(0),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / 1e9,
            "multi_processor_count": props.multi_processor_count,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "tensor_cores": props.major >= 7,
            "theoretical_tflops_fp16": _estimate_cuda_tflops(props, "fp16"),
            "theoretical_tflops_fp32": _estimate_cuda_tflops(props, "fp32"),
        })
        
    elif device == "xpu" and _has_xpu_backend() and torch.xpu.is_available():
        props = torch.xpu.get_device_properties(0)
        info.update({
            "name": torch.xpu.get_device_name(0),
            "total_memory_gb": getattr(props, "total_memory", 0) / 1e9,
            "max_compute_units": getattr(props, "max_compute_units", 0),
            "gpu_eu_count": getattr(props, "gpu_eu_count", 0),
            "driver_version": str(getattr(props, "driver_version", "")),
            "has_fp16": getattr(props, "has_fp16", False),
            "has_fp64": getattr(props, "has_fp64", False),
        })
        
    elif device == "mps":
        info.update({
            "name": "Apple MPS",
            "backend": "Metal Performance Shaders",
            "theoretical_tflops_fp16": _estimate_mps_tflops("fp16"),
            "theoretical_tflops_fp32": _estimate_mps_tflops("fp32"),
        })
    
    elif device == "cpu":
        import platform
        cpu_name = platform.processor() or "Unknown CPU"
        info.update({
            "name": cpu_name,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        })
        if HAS_PSUTIL:
            info["total_memory_gb"] = psutil.virtual_memory().total / 1e9
        
    return info


def _estimate_cuda_tflops(props, precision: str) -> float:
    """Estimate theoretical peak TFLOPS for CUDA device."""
    sm_count = props.multi_processor_count
    # Use actual clock speed if available, otherwise estimate
    clock_ghz = getattr(props, 'clock_rate', 1500000) / 1e6  # kHz to GHz
    
    if props.major >= 9:  # Hopper (H100)
        flops_per_sm = 512 if precision == "fp16" else 128
    elif props.major >= 8:  # Ampere (A100, RTX 30xx)
        flops_per_sm = 256 if precision == "fp16" else 64
    elif props.major >= 7:  # Volta/Turing (V100, T4, RTX 20xx)
        flops_per_sm = 128 if precision == "fp16" else 64
    else:  # Pascal and older
        flops_per_sm = 64 if precision == "fp16" else 32
    
    return sm_count * flops_per_sm * clock_ghz * 2 / 1000  # *2 for FMA


def _estimate_mps_tflops(precision: str) -> float:
    """Estimate theoretical peak TFLOPS for Apple MPS (conservative)."""
    # Conservative estimates for Apple Silicon
    # M1: ~2.6 TFLOPS FP32, ~5.2 TFLOPS FP16
    # M1 Pro/Max: ~5-10 TFLOPS FP32
    # M2: ~3.6 TFLOPS FP32
    # M3 Max: ~14 TFLOPS FP32
    # Default to M1-class conservative estimate
    return 5.0 if precision == "fp16" else 2.5


def synchronize(device: str):
    """Synchronize device for accurate timing."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu" and _has_xpu_backend():
        torch.xpu.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    # CPU is synchronous by default, no action needed


# CPU memory tracking state
_cpu_memory_baseline_gb: float = 0.0
_cpu_peak_memory_gb: float = 0.0


def get_peak_memory_gb(device: str, model: nn.Module = None, dtype: torch.dtype = None) -> float:
    """Get peak memory allocated in GB."""
    global _cpu_peak_memory_gb
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if device == "xpu" and _has_xpu_backend():
        return torch.xpu.max_memory_allocated() / 1e9
    if device == "cpu":
        # For CPU, estimate from model parameters if provided (more accurate)
        if model is not None:
            bytes_per_param = 4  # default float32
            if dtype in (torch.float16, torch.bfloat16):
                bytes_per_param = 2
            param_memory = sum(p.numel() * bytes_per_param for p in model.parameters())
            return param_memory / 1e9
        elif HAS_PSUTIL:
            current = psutil.Process().memory_info().rss / 1e9
            _cpu_peak_memory_gb = max(_cpu_peak_memory_gb, current - _cpu_memory_baseline_gb)
            return _cpu_peak_memory_gb
    return 0.0


def get_reserved_memory_gb(device: str) -> float:
    """Get reserved memory in GB."""
    if device == "cuda":
        return torch.cuda.max_memory_reserved() / 1e9
    if device == "xpu" and _has_xpu_backend():
        return torch.xpu.max_memory_reserved() / 1e9
    if device == "cpu" and HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1e9
    return 0.0


def reset_memory_stats(device: str):
    """Reset memory statistics."""
    global _cpu_memory_baseline_gb, _cpu_peak_memory_gb
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    elif device == "xpu" and _has_xpu_backend():
        torch.xpu.reset_peak_memory_stats()
        torch.xpu.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    elif device == "cpu" and HAS_PSUTIL:
        _cpu_memory_baseline_gb = psutil.Process().memory_info().rss / 1e9
        _cpu_peak_memory_gb = 0.0


# =============================================================================
# Model Profiling
# =============================================================================

def profile_model(model: nn.Module, config: ModelConfig) -> ModelProfile:
    """Profile model architecture and compute estimates."""
    prof = ModelProfile()
    prof.name = config.model_name
    
    param_groups = defaultdict(int)
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if "embed_tokens" in name:
            param_groups["embedding"] += num_params
        elif "lm_head" in name or "mtp_heads" in name:
            param_groups["head"] += num_params
        elif "norm" in name:
            param_groups["norm"] += num_params
        elif "attn" in name or "attention" in name:
            param_groups["attention"] += num_params
        elif "ffn" in name or "mlp" in name or "feed" in name:
            param_groups["ffn"] += num_params
        elif "mhc" in name or "hyper" in name:
            param_groups["connection"] += num_params
        else:
            param_groups["other"] += num_params
    
    prof.total_params = sum(p.numel() for p in model.parameters())
    prof.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prof.params_billions = prof.total_params / 1e9
    
    prof.embedding_params = param_groups["embedding"]
    prof.attention_params = param_groups["attention"]
    prof.ffn_params = param_groups["ffn"]
    prof.head_params = param_groups["head"]
    prof.norm_params = param_groups["norm"]
    prof.connection_params = param_groups["connection"]
    
    prof.forward_flops = estimate_forward_flops(config)
    prof.backward_flops = prof.forward_flops * 2
    prof.total_training_flops = prof.forward_flops * 3
    
    return prof


def estimate_forward_flops(config: ModelConfig) -> int:
    """Estimate forward pass FLOPs per token."""
    H = config.hidden_size
    L = config.num_hidden_layers
    V = config.vocab_size
    I = config.ffn.intermediate_size
    h = config.attention.num_attention_heads
    d = config.attention.head_dim
    kv_heads = config.attention.num_key_value_heads
    
    attn_qkvo = 2 * H * (h * d + 2 * kv_heads * d + H)
    
    if config.ffn.ffn_type == FFNType.SWIGLU:
        ffn_flops = 3 * 2 * H * I
    else:
        ffn_flops = 2 * 2 * H * I
    
    layer_flops = attn_qkvo + ffn_flops
    head_flops = 0 if config.head.tie_word_embeddings else 2 * H * V
    total_flops = L * layer_flops + head_flops
    
    if config.head.use_multi_token_prediction:
        mtp_flops = config.head.num_predict_tokens * 2 * H * V
        total_flops += mtp_flops
    
    if config.connection.connection_type == ConnectionType.MHC:
        n = config.connection.mhc_expansion_rate
        mhc_flops_per_layer = 2 * H * n * (2 * n + n * n)
        total_flops += L * mhc_flops_per_layer
    
    return total_flops


def estimate_kv_cache_size(config: ModelConfig, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    """Estimate KV cache size in GB."""
    L = config.num_hidden_layers
    kv_heads = config.attention.num_key_value_heads
    d = config.attention.head_dim
    
    bytes_per_element = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    kv_cache_bytes = 2 * L * batch_size * kv_heads * seq_len * d * bytes_per_element
    
    if config.attention.attention_type in [AttentionType.GATED_SPARSE, AttentionType.DEEPSEEK_GSA]:
        k_base = config.attention.gsa_k_base
        sparsity_factor = min(1.0, k_base / seq_len)
        kv_cache_bytes *= sparsity_factor
    
    return kv_cache_bytes / 1e9


def estimate_activation_memory(config: ModelConfig, batch_size: int, seq_len: int, dtype: torch.dtype) -> float:
    """Estimate activation memory in GB (for training)."""
    H = config.hidden_size
    L = config.num_hidden_layers
    I = config.ffn.intermediate_size
    
    bytes_per_element = {torch.float32: 4, torch.float16: 2, torch.bfloat16: 2}.get(dtype, 4)
    per_layer_activations = batch_size * seq_len * (H + 3 * H + H + I)
    total_activation_bytes = L * per_layer_activations * bytes_per_element
    return total_activation_bytes / 1e9


# =============================================================================
# Benchmark Profile Scaling
# =============================================================================

def apply_benchmark_profile(config: ModelConfig, profile_name: Optional[str]) -> ModelConfig:
    """Scale model for benchmarking on constrained devices."""
    if not profile_name:
        return config

    profile_name = profile_name.lower().strip()
    if profile_name not in {"micro", "tiny", "small"}:
        raise ValueError(f"Unsupported profile: {profile_name}. Use: micro, tiny, small")

    import copy
    config = copy.deepcopy(config)

    if profile_name == "micro":
        config.model_name = f"{config.model_name} (micro)"
        config.hidden_size = 256
        config.num_hidden_layers = 4
        config.max_position_embeddings = 8192
        config.attention.num_attention_heads = 4
        config.attention.num_key_value_heads = 1
        config.attention.head_dim = 64
        config.ffn.intermediate_size = 1024
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 2)
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 2)
        config.attention.gsa_k_base = min(config.attention.gsa_k_base, 512)
        config.attention.gsa_k_max = min(config.attention.gsa_k_max, 1024)

    elif profile_name == "tiny":
        config.model_name = f"{config.model_name} (tiny)"
        config.hidden_size = 512
        config.num_hidden_layers = 8
        config.max_position_embeddings = 4096
        config.attention.num_attention_heads = 8
        config.attention.num_key_value_heads = 2
        config.attention.head_dim = 64
        config.ffn.intermediate_size = 2048
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 3)
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 2)
        config.attention.gsa_k_base = min(config.attention.gsa_k_base, 1024)
        config.attention.gsa_k_max = min(config.attention.gsa_k_max, 2048)

    elif profile_name == "small":
        config.model_name = f"{config.model_name} (small)"
        config.hidden_size = 1024
        config.num_hidden_layers = 12
        config.max_position_embeddings = max(4096, min(config.max_position_embeddings, 8192))
        config.attention.num_attention_heads = 16
        config.attention.num_key_value_heads = 4
        config.attention.head_dim = 64
        config.ffn.intermediate_size = 4096
        config.connection.mhc_expansion_rate = min(config.connection.mhc_expansion_rate, 4)
        if config.head.use_multi_token_prediction:
            config.head.num_predict_tokens = min(config.head.num_predict_tokens, 3)

    config.__post_init__()
    return config


# =============================================================================
# Core Benchmark Functions
# =============================================================================

def run_inference_benchmark(
    model: nn.Module,
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    warmup_iters: int,
    benchmark_iters: int,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> BenchmarkResult:
    """Run inference benchmark with comprehensive metrics."""
    
    result = BenchmarkResult(seq_len=seq_len, batch_size=batch_size)
    model.eval()
    reset_memory_stats(device)
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_ids)
    
    synchronize(device)
    reset_memory_stats(device)
    
    latencies = []
    with torch.no_grad():
        for _ in range(benchmark_iters):
            synchronize(device)
            start = time.perf_counter()
            _ = model(input_ids)
            synchronize(device)
            latencies.append((time.perf_counter() - start) * 1000)
            # Track peak memory during execution (important for CPU)
            if device == "cpu" and HAS_PSUTIL:
                get_peak_memory_gb(device)
    
    result.memory.peak_allocated_gb = get_peak_memory_gb(device, model=model, dtype=dtype)
    result.memory.peak_reserved_gb = get_reserved_memory_gb(device)
    
    total_time_sec = sum(latencies) / 1000
    total_tokens = batch_size * seq_len * benchmark_iters
    
    result.throughput.tokens_per_sec = total_tokens / total_time_sec
    result.throughput.samples_per_sec = (batch_size * benchmark_iters) / total_time_sec
    
    forward_flops_per_seq = estimate_forward_flops(config) * seq_len
    total_flops = forward_flops_per_seq * batch_size * benchmark_iters
    result.throughput.tflops_achieved = total_flops / total_time_sec / 1e12
    
    device_info = get_device_info(device)
    if "theoretical_tflops_fp16" in device_info:
        theoretical = device_info["theoretical_tflops_fp16"]
        result.throughput.mfu = result.throughput.tflops_achieved / theoretical if theoretical > 0 else 0
    
    if latencies:
        result.latency.mean_ms = statistics.mean(latencies)
        result.latency.std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
        result.latency.min_ms = min(latencies)
        result.latency.max_ms = max(latencies)
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        result.latency.p50_ms = sorted_lat[n // 2]
        result.latency.p95_ms = sorted_lat[int(n * 0.95)]
        result.latency.p99_ms = sorted_lat[int(n * 0.99)]
    
    return result


def run_training_benchmark(
    model: nn.Module,
    config: ModelConfig,
    batch_size: int,
    seq_len: int,
    warmup_iters: int,
    benchmark_iters: int,
    device: str,
    dtype: torch.dtype,
) -> BenchmarkResult:
    """Run training benchmark (forward + backward)."""
    
    result = BenchmarkResult(seq_len=seq_len, batch_size=batch_size)
    model.train()
    reset_memory_stats(device)
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    amp_enabled = device in ["cuda", "xpu"]
    amp_dtype = dtype if dtype in [torch.float16, torch.bfloat16] else torch.bfloat16
    
    for _ in range(warmup_iters):
        if amp_enabled:
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                output = model(input_ids, labels=labels)
                loss = output.loss
        else:
            output = model(input_ids, labels=labels)
            loss = output.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    
    synchronize(device)
    reset_memory_stats(device)
    
    latencies = []
    for _ in range(benchmark_iters):
        synchronize(device)
        start = time.perf_counter()
        
        if amp_enabled:
            with torch.amp.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                output = model(input_ids, labels=labels)
                loss = output.loss
        else:
            output = model(input_ids, labels=labels)
            loss = output.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        
        synchronize(device)
        latencies.append((time.perf_counter() - start) * 1000)
    
    result.memory.peak_allocated_gb = get_peak_memory_gb(device, model=model, dtype=dtype)
    result.memory.peak_reserved_gb = get_reserved_memory_gb(device)
    
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    result.memory.model_size_gb = sum(p.numel() for p in model.parameters()) * dtype_size / 1e9
    result.memory.gradient_memory_gb = result.memory.model_size_gb
    result.memory.activation_memory_gb = estimate_activation_memory(config, batch_size, seq_len, dtype)
    result.memory.kv_cache_gb = estimate_kv_cache_size(config, batch_size, seq_len, dtype)
    
    total_time_sec = sum(latencies) / 1000
    total_tokens = batch_size * seq_len * benchmark_iters
    
    result.throughput.tokens_per_sec = total_tokens / total_time_sec
    result.throughput.samples_per_sec = (batch_size * benchmark_iters) / total_time_sec
    
    training_flops_per_seq = estimate_forward_flops(config) * seq_len * 3
    total_flops = training_flops_per_seq * batch_size * benchmark_iters
    result.throughput.tflops_achieved = total_flops / total_time_sec / 1e12
    
    result.latency.mean_ms = statistics.mean(latencies)
    result.latency.std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
    result.latency.min_ms = min(latencies)
    result.latency.max_ms = max(latencies)
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    result.latency.p50_ms = sorted_lat[n // 2]
    result.latency.p95_ms = sorted_lat[int(n * 0.95)]
    result.latency.p99_ms = sorted_lat[int(n * 0.99)]
    
    return result


# =============================================================================
# Insight Generation
# =============================================================================

def generate_insights(
    model_profile: ModelProfile,
    config: ModelConfig,
    inference_results: List[BenchmarkResult],
    training_results: List[BenchmarkResult],
    device_info: Dict[str, Any],
) -> InsightReport:
    """Generate actionable insights from benchmark results."""
    
    insights = InsightReport()
    
    if inference_results:
        avg_mem = statistics.mean([r.memory.peak_allocated_gb for r in inference_results])
        total_device_mem = device_info.get("total_memory_gb", 16)
        
        if avg_mem > total_device_mem * 0.9:
            insights.bottleneck = "MEMORY_BOUND: Peak memory >90% of device capacity"
            insights.warnings.append("⚠️ Memory pressure - consider gradient checkpointing")
        else:
            mfu_values = [r.throughput.mfu for r in inference_results if r.throughput.mfu > 0]
            if mfu_values:
                avg_mfu = statistics.mean(mfu_values)
                if avg_mfu > 0.5:
                    insights.bottleneck = "COMPUTE_BOUND: Good MFU indicates efficient utilization"
                else:
                    insights.bottleneck = f"SUB_OPTIMAL: MFU of {avg_mfu:.1%} - optimization opportunities exist"
            else:
                insights.bottleneck = "UNKNOWN: MFU not available (CPU or unsupported device)"
    
    attention_type = config.attention.attention_type
    if attention_type == AttentionType.GROUPED_QUERY:
        reduction = config.attention.num_attention_heads / config.attention.num_key_value_heads
        insights.memory_recommendation = f"GQA: {reduction:.0f}x KV cache reduction vs MHA"
    elif attention_type in [AttentionType.GATED_SPARSE, AttentionType.DEEPSEEK_GSA]:
        insights.memory_recommendation = f"GSA k_base={config.attention.gsa_k_base}: sparse attention for long sequences"
    elif attention_type == AttentionType.DEEPSEEK_SPARSE:
        compressed = config.attention.ds_compressed_dim
        full = config.attention.num_attention_heads * config.attention.head_dim
        insights.memory_recommendation = f"DeepSeek MLA: {1-compressed/full:.1%} KV cache reduction"
    
    if training_results and len(training_results) > 1:
        seq_throughputs = [(r.seq_len, r.throughput.tokens_per_sec) for r in training_results]
        short = min(seq_throughputs, key=lambda x: x[0])
        long = max(seq_throughputs, key=lambda x: x[0])
        drop = 1 - long[1]/short[1]
        insights.throughput_recommendation = f"Seq scaling: {drop:.0%} throughput drop from {short[0]} to {long[0]} tokens"
    
    if attention_type == AttentionType.GATED_SPARSE:
        insights.attention_insight = "GSA (2601.15305v1): Monitor indexer confidence for k tuning"
    elif attention_type == AttentionType.DEEPSEEK_GSA:
        insights.attention_insight = f"DeepSeek GSA: {config.attention.gsa_adaptive_k_method} adaptive k"
    elif config.connection.connection_type == ConnectionType.MHC:
        overhead = (model_profile.connection_params / model_profile.total_params) * 100 if model_profile.total_params > 0 else 0
        insights.attention_insight = f"mHC n={config.connection.mhc_expansion_rate}: {overhead:.2f}% overhead (paper 2512.24880)"
    
    return insights


def generate_comparison_insights(results_by_config: Dict[str, Dict[str, Any]]) -> str:
    """Generate comparison insights across configurations."""
    
    if len([k for k in results_by_config if not k.startswith("_")]) < 2:
        return "Need at least 2 configurations for comparison"
    
    lines = ["\n" + "=" * 80, "📊 COMPARATIVE ANALYSIS", "=" * 80]
    
    inference_data = {}
    for cfg, data in results_by_config.items():
        if cfg.startswith("_"):
            continue
        if "inference" in data and data["inference"]:
            longest = max(data["inference"], key=lambda x: x.get("seq_len", 0))
            inference_data[cfg] = longest
    
    if inference_data:
        lines.append("\n🎯 INFERENCE THROUGHPUT (longest sequence):")
        sorted_by_tps = sorted(
            inference_data.items(),
            key=lambda x: x[1].get("throughput", {}).get("tokens_per_sec", 0),
            reverse=True
        )
        best_name, best_data = sorted_by_tps[0]
        best_tps = best_data.get("throughput", {}).get("tokens_per_sec", 0)
        
        for cfg, data in sorted_by_tps:
            tps = data.get("throughput", {}).get("tokens_per_sec", 0)
            seq = data.get("seq_len", 0)
            pct = (tps / best_tps * 100) if best_tps > 0 else 0
            mem = data.get("memory", {}).get("peak_allocated_gb", 0)
            marker = "🥇" if cfg == best_name else "  "
            lines.append(f"  {marker} {cfg}: {tps:,.0f} tok/s @ seq={seq} ({pct:.0f}%, {mem:.2f}GB)")
    
    lines.append("\n💾 MEMORY EFFICIENCY (tokens/s per GB):")
    for cfg, data in inference_data.items():
        mem = data.get("memory", {}).get("peak_allocated_gb", 0)
        tps = data.get("throughput", {}).get("tokens_per_sec", 0)
        eff = tps / mem if mem > 0 else 0
        lines.append(f"  {cfg}: {eff:,.0f} tok/s/GB")
    
    return "\n".join(lines)


# =============================================================================
# Main Benchmark Orchestration
# =============================================================================

def benchmark_throughput(
    config_paths: List[str],
    batch_size: int = 4,
    seq_lengths: List[int] = [128, 256, 512, 1024],
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    dtype: str = "bfloat16",
    compile_model: bool = False,
    device_override: Optional[str] = None,
    output_path: Optional[str] = None,
    inference_only: bool = False,
    profile: Optional[str] = None,
    full_analysis: bool = False,
    compare_mode: bool = False,
    output_format: str = "standard",
) -> Dict[str, Any]:
    """Run comprehensive throughput benchmark."""
    
    device = get_device(device_override)
    device_info = get_device_info(device)
    
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    # Header
    print("\n" + "=" * 80)
    print("🚀 SOTA LLM THROUGHPUT BENCHMARK")
    print("=" * 80)
    print(f"Device: {device} ({device_info.get('name', 'N/A')})")
    print(f"PyTorch: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA: {device_info.get('cuda_version', 'N/A')} | Memory: {device_info.get('total_memory_gb', 0):.1f}GB")
    print(f"Dtype: {dtype} | Batch: {batch_size} | Sequences: {seq_lengths}")
    if profile:
        print(f"Profile: {profile} (model scaled)")
    print("=" * 80)
    
    all_results = {}
    
    for config_path in config_paths:
        print(f"\n{'='*80}")
        print(f"📋 CONFIG: {Path(config_path).name}")
        print("=" * 80)
        
        config = ModelConfig.load(config_path)
        config = apply_benchmark_profile(config, profile)
        
        print(f"Model: {config.model_name}")
        print(f"  Hidden: {config.hidden_size} | Layers: {config.num_hidden_layers}")
        print(f"  Attention: {config.attention.attention_type.value} | Heads: {config.attention.num_attention_heads}/{config.attention.num_key_value_heads} | HeadDim: {config.attention.head_dim}")
        print(f"  FFN: {config.ffn.ffn_type.value} ({config.ffn.intermediate_size}) | Position: {config.position.position_type.value}")
        print(f"  Connection: {config.connection.connection_type.value} | MTP: {config.head.use_multi_token_prediction}")
        
        print(f"\n🔧 Loading model...")
        model = LLM(config).to(device).to(torch_dtype)
        
        if compile_model and hasattr(torch, 'compile'):
            print("  Compiling with torch.compile()...")
            model = torch.compile(model, mode="reduce-overhead")
        
        if device == 'xpu' and HAS_IPEX:
            import intel_extension_for_pytorch as ipex_opt
            model = ipex_opt.optimize(model, dtype=torch_dtype)
        
        model_prof = profile_model(model, config)
        print(f"\n📊 Parameters: {model_prof.total_params:,} ({model_prof.params_billions:.3f}B)")
        print(f"  Embedding: {model_prof.embedding_params:,} | Attention: {model_prof.attention_params:,} | FFN: {model_prof.ffn_params:,}")
        
        # Inference benchmark
        print(f"\n{'─'*80}")
        print("📈 INFERENCE (Forward Pass)")
        print("─" * 80)
        print(f"{'Seq':<8} {'Tok/s':<12} {'Samp/s':<10} {'Lat(ms)':<12} {'P95(ms)':<10} {'Mem(GB)':<10} {'TFLOPS':<8}")
        print("─" * 80)
        
        inference_results = []
        for seq_len in seq_lengths:
            if seq_len > config.max_position_embeddings:
                print(f"{seq_len:<8} SKIP (exceeds max_pos={config.max_position_embeddings})")
                continue
            
            try:
                result = run_inference_benchmark(model, config, batch_size, seq_len, warmup_iters, benchmark_iters, device, torch_dtype)
                print(f"{seq_len:<8} {result.throughput.tokens_per_sec:<12,.0f} {result.throughput.samples_per_sec:<10.2f} "
                      f"{result.latency.mean_ms:<12.2f} {result.latency.p95_ms:<10.2f} "
                      f"{result.memory.peak_allocated_gb:<10.2f} {result.throughput.tflops_achieved:<8.2f}")
                inference_results.append(result)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{seq_len:<8} OOM")
                    reset_memory_stats(device)
                else:
                    print(f"{seq_len:<8} ERROR: {str(e)[:40]}")
        
        # Training benchmark
        training_results = []
        if not inference_only:
            print(f"\n{'─'*80}")
            print("🏋️ TRAINING (Forward + Backward)")
            print("─" * 80)
            print(f"{'Seq':<8} {'Tok/s':<12} {'Samp/s':<10} {'Lat(ms)':<12} {'P95(ms)':<10} {'Mem(GB)':<10} {'TFLOPS':<8}")
            print("─" * 80)
            
            for seq_len in seq_lengths:
                if seq_len > config.max_position_embeddings:
                    continue
                try:
                    result = run_training_benchmark(model, config, batch_size, seq_len, warmup_iters, benchmark_iters, device, torch_dtype)
                    print(f"{seq_len:<8} {result.throughput.tokens_per_sec:<12,.0f} {result.throughput.samples_per_sec:<10.2f} "
                          f"{result.latency.mean_ms:<12.2f} {result.latency.p95_ms:<10.2f} "
                          f"{result.memory.peak_allocated_gb:<10.2f} {result.throughput.tflops_achieved:<8.2f}")
                    training_results.append(result)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{seq_len:<8} OOM")
                        reset_memory_stats(device)
                    else:
                        print(f"{seq_len:<8} ERROR: {str(e)[:40]}")
        
        # Insights
        insights = generate_insights(model_prof, config, inference_results, training_results, device_info)
        print(f"\n{'─'*80}")
        print("💡 INSIGHTS")
        print("─" * 80)
        print(f"Bottleneck: {insights.bottleneck}")
        print(f"Memory: {insights.memory_recommendation}")
        if insights.throughput_recommendation:
            print(f"Throughput: {insights.throughput_recommendation}")
        if insights.attention_insight:
            print(f"Architecture: {insights.attention_insight}")
        for w in insights.warnings:
            print(f"  {w}")
        
        # Store results
        config_name = Path(config_path).stem
        all_results[config_name] = {
            "timestamp": datetime.now().isoformat(),
            "config_path": str(config_path),
            "config_name": config.model_name,
            "device": device,
            "device_info": device_info,
            "dtype": dtype,
            "batch_size": batch_size,
            "profile": profile,
            "model_profile": asdict(model_prof),
            "model_settings": {
                "attention": config.attention.attention_type.value,
                "position": config.position.position_type.value,
                "connection": config.connection.connection_type.value,
                "ffn": config.ffn.ffn_type.value,
                "mtp": config.head.use_multi_token_prediction,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_hidden_layers,
                "head_dim": config.attention.head_dim,
                "num_heads": config.attention.num_attention_heads,
                "num_kv_heads": config.attention.num_key_value_heads,
            },
            "inference": [asdict(r) for r in inference_results],
            "training": [asdict(r) for r in training_results],
            "insights": asdict(insights),
        }
        
        del model
        reset_memory_stats(device)
    
    # Comparison
    if compare_mode and len(all_results) > 1:
        comparison = generate_comparison_insights(all_results)
        print(comparison)
        all_results["_comparison"] = comparison
    
    # Save
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_format == "minimal":
            output_data = {
                k: {"model": v["config_name"],
                    "inference_tps": [r["throughput"]["tokens_per_sec"] for r in v["inference"]],
                    "training_tps": [r["throughput"]["tokens_per_sec"] for r in v["training"]]}
                for k, v in all_results.items() if not k.startswith("_")
            }
        else:
            output_data = all_results
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n✅ Saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("🏁 BENCHMARK COMPLETE")
    print("=" * 80)
    
    return all_results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SOTA LLM Throughput Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_throughput.py configs/1b_gsa.yaml
  python benchmark_throughput.py configs/1b_base.yaml configs/1b_gsa.yaml --compare
  python benchmark_throughput.py configs/1b_deepseek_gsa.yaml --profile micro --device mps
  python benchmark_throughput.py configs/1b_full.yaml -o results/benchmark.json
  python benchmark_throughput.py configs/1b_yarn.yaml --seq-lengths 512,1024,2048,4096
        """
    )
    
    parser.add_argument('configs', type=str, nargs='+', help='YAML config file(s)')
    parser.add_argument('--batch-size', '-b', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--seq-lengths', '-s', type=str, default='128,256,512,1024', help='Sequence lengths')
    parser.add_argument('--warmup', '-w', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--iters', '-i', type=int, default=20, help='Benchmark iterations')
    parser.add_argument('--dtype', '-d', type=str, default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'xpu', 'mps', 'cpu'])
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON path')
    parser.add_argument('--inference-only', action='store_true', help='Skip training benchmark')
    parser.add_argument('--profile', '-p', type=str, default=None, choices=['micro', 'tiny', 'small'])
    parser.add_argument('--full-analysis', action='store_true', help='Full profiling')
    parser.add_argument('--compare', action='store_true', help='Comparative mode')
    parser.add_argument('--format', type=str, default='standard', choices=['minimal', 'standard', 'detailed'])
    
    args = parser.parse_args()
    seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(',')]
    
    benchmark_throughput(
        config_paths=args.configs,
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
        full_analysis=args.full_analysis,
        compare_mode=args.compare,
        output_format=args.format,
    )


if __name__ == '__main__':
    main()
