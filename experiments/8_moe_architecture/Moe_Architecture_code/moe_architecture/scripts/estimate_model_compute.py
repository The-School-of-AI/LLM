#!/usr/bin/env python3
"""
Estimate FLOPs and memory for model configs.

This is an approximate estimator intended for quick comparisons.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from configs import CONFIGS, get_config  # noqa: E402


DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate FLOPs and memory for model configs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(CONFIGS.keys()),
        help=f"Configs to analyze: {list(CONFIGS.keys())}",
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=DTYPE_BYTES.keys(), default="bf16")
    parser.add_argument("--optimizer", choices=["none", "adamw"], default="adamw")
    parser.add_argument("--optimizer-state-dtype", choices=DTYPE_BYTES.keys(), default=None)
    parser.add_argument("--activation-factor", type=float, default=3.0)
    parser.add_argument("--include-backward", action="store_true")
    return parser.parse_args()


def _bytes_to_gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def _estimate_attention_flops(config, seq_len: int) -> float:
    h = config.hidden_size
    nh = config.attention.num_attention_heads
    nkv = config.attention.num_kv_heads
    head_dim = config.attention.head_dim

    proj_q = 2 * h * h
    proj_k = 2 * h * h * (nkv / nh)
    proj_v = 2 * h * h * (nkv / nh)
    proj_o = 2 * h * h
    proj_total = proj_q + proj_k + proj_v + proj_o

    if config.attention.attention_type == "gsa":
        # Gates and indexer projections
        gate_v = 2 * h * h * (nkv / nh)
        gate_o = 2 * h * h
        index_q = 2 * h * (config.attention.gsa_indexer_heads * config.attention.gsa_indexer_dim)
        index_k = 2 * h * (config.attention.gsa_indexer_heads * config.attention.gsa_indexer_dim)
        index_w = 2 * h * config.attention.gsa_indexer_heads
        proj_total += gate_v + gate_o + index_q + index_k + index_w

        # GSA compute cost: indexer + sparse attention
        k = min(config.attention.gsa_k_base, seq_len)
        indexer_cost = 2 * (seq_len ** 2) * config.attention.gsa_indexer_dim * config.attention.gsa_indexer_heads
        sparse_attn = 2 * seq_len * k * h
        return proj_total + indexer_cost + sparse_attn

    # Dense attention cost: QK + AV
    dense_attn = 4 * seq_len * h
    return proj_total + dense_attn


def _estimate_ffn_flops(config, expected_real: float) -> float:
    h = config.hidden_size
    d = config.expert.intermediate_size
    per_expert = 6 * h * d
    if config.model_type.value == "dense":
        return per_expert
    return (config.num_shared_experts + expected_real) * per_expert


def estimate_flops(config, seq_len: int, batch_size: int) -> Dict[str, float]:
    if config.model_type.value == "dense":
        expected_real = 0.0
    else:
        if config.router.router_type.value == "null_expert":
            expected_real = config.router.top_k * config.router.data_sparsity
        else:
            expected_real = config.router.top_k

    attn = _estimate_attention_flops(config, seq_len)
    ffn_dense = _estimate_ffn_flops(config, expected_real)

    per_layer = attn + ffn_dense
    total = per_layer * config.num_layers
    total *= batch_size * seq_len
    return {
        "per_token": per_layer,
        "per_batch": total,
    }


def estimate_memory(
    config,
    seq_len: int,
    batch_size: int,
    dtype: str,
    optimizer: str,
    activation_factor: float,
    optimizer_state_dtype: str | None,
) -> Dict[str, float]:
    bytes_per = DTYPE_BYTES[dtype]
    params = config.estimated_total_params
    param_mem = params * bytes_per

    grad_mem = params * bytes_per if optimizer != "none" else 0
    if optimizer == "adamw":
        state_dtype = optimizer_state_dtype or dtype
        state_bytes = DTYPE_BYTES[state_dtype]
        opt_mem = params * state_bytes * 2
    else:
        opt_mem = 0

    activation_mem = batch_size * seq_len * config.hidden_size * bytes_per * config.num_layers * activation_factor

    kv_mem = 0
    if config.attention.attention_type in {"gqa", "gsa"}:
        kv_mem = (
            batch_size
            * seq_len
            * config.attention.num_kv_heads
            * config.attention.head_dim
            * bytes_per
            * 2
            * config.num_layers
        )

    return {
        "param_gb": _bytes_to_gb(param_mem),
        "grad_gb": _bytes_to_gb(grad_mem),
        "opt_gb": _bytes_to_gb(opt_mem),
        "activation_gb": _bytes_to_gb(activation_mem),
        "kv_cache_gb": _bytes_to_gb(kv_mem),
        "total_train_gb": _bytes_to_gb(param_mem + grad_mem + opt_mem + activation_mem),
        "total_infer_gb": _bytes_to_gb(param_mem + kv_mem),
    }


def main() -> None:
    global args
    args = parse_args()

    for name in args.configs:
        config = get_config(name)
        flops = estimate_flops(config, args.seq_len, args.batch_size)
        mem = estimate_memory(
            config,
            args.seq_len,
            args.batch_size,
            args.dtype,
            args.optimizer,
            args.activation_factor,
            args.optimizer_state_dtype,
        )

        print("=" * 80)
        print(f"Config: {name}")
        print(f"  Params (est): {config.estimated_total_params/1e9:.2f}B")
        print(f"  Attention: {config.attention.attention_type}")
        if config.model_type.value != "dense":
            print(f"  Router: {config.router.router_type.value}, top_k={config.router.top_k}, rho={config.router.data_sparsity}")
        print(f"  FLOPs per token per layer: {flops['per_token'] / 1e9:.3f} GFLOPs")
        print(f"  FLOPs per batch: {flops['per_batch'] / 1e12:.3f} TFLOPs")
        if args.include_backward:
            print(f"  FLOPs per batch (fwd+bwd ~2x): {flops['per_batch'] * 2 / 1e12:.3f} TFLOPs")
        print("  Memory (GB):")
        print(f"    Params:     {mem['param_gb']:.2f}")
        print(f"    Gradients:  {mem['grad_gb']:.2f}")
        print(f"    Optimizer:  {mem['opt_gb']:.2f}")
        print(f"    Activations:{mem['activation_gb']:.2f}")
        print(f"    KV Cache:   {mem['kv_cache_gb']:.2f}")
        print(f"    Total Train:{mem['total_train_gb']:.2f}")
        print(f"    Total Infer:{mem['total_infer_gb']:.2f}")


if __name__ == "__main__":
    main()
