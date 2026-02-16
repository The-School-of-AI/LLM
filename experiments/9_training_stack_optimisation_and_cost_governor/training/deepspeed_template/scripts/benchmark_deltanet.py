#!/usr/bin/env python3
"""
Benchmark DeltaNet fused-kernel path for preflight validation.

Usage:
  python scripts/benchmark_deltanet.py --seq-lengths 4096,8192,16384
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernels import HAS_FLA  # noqa: E402
from src.models.recurrence_model_1b import GatedDeltaNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeltaNet fused-kernel benchmark")
    parser.add_argument("--seq-lengths", type=str, default="4096,8192,16384")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--backward", action="store_true", help="Include backward pass timing")
    parser.add_argument(
        "--require-fused",
        action="store_true",
        default=True,
        help="Crash if fused DeltaNet kernel is unavailable/failing (default: true)",
    )
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def parse_seq_lengths(seq_str: str) -> List[int]:
    return [int(x.strip()) for x in seq_str.split(",") if x.strip()]


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def run_once(
    model: torch.nn.Module,
    x: torch.Tensor,
    warmup: int,
    steps: int,
    backward: bool,
) -> Dict[str, float]:
    times: List[float] = []

    model.train(mode=backward)
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        y = model(x)
        if backward:
            loss = y.float().mean()
            loss.backward()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(x.device)
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        y = model(x)
        if backward:
            loss = y.float().mean()
            loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_s = statistics.mean(times)
    p95_s = sorted(times)[max(0, int(0.95 * len(times)) - 1)]
    peak_gb = torch.cuda.max_memory_allocated(x.device) / (1024**3)
    return {"avg_step_s": avg_s, "p95_step_s": p95_s, "peak_mem_gb": peak_gb}


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeltaNet benchmark.")
    if args.require_fused and not HAS_FLA:
        raise RuntimeError(
            "FLA fused DeltaNet kernel is unavailable (HAS_FLA=False) and "
            "--require-fused is enabled."
        )

    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda")
    seq_lengths = parse_seq_lengths(args.seq_lengths)

    model = GatedDeltaNet(
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        max_seq_len=max(seq_lengths),
        require_fused_kernel=args.require_fused,
    ).to(device=device, dtype=dtype)

    results = []
    print("\nDeltaNet Benchmark")
    print("=" * 80)
    print(
        f"device={device}, dtype={dtype}, backward={args.backward}, "
        f"require_fused={args.require_fused}, HAS_FLA={HAS_FLA}"
    )
    print("=" * 80)

    for seq_len in seq_lengths:
        x = torch.randn(args.batch_size, seq_len, args.hidden_size, device=device, dtype=dtype)
        try:
            stats = run_once(model, x, args.warmup, args.steps, args.backward)
            tokens_per_sec = (args.batch_size * seq_len) / stats["avg_step_s"]
            row = {
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "tokens_per_sec": tokens_per_sec,
                **stats,
            }
            results.append(row)
            print(
                f"seq={seq_len:6d} | toks/s={tokens_per_sec:10.1f} | "
                f"avg={stats['avg_step_s']:.4f}s | p95={stats['p95_step_s']:.4f}s | "
                f"peak={stats['peak_mem_gb']:.2f}GB"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            raise

    payload = {
        "benchmark": "deltanet",
        "dtype": args.dtype,
        "backward": args.backward,
        "require_fused": args.require_fused,
        "has_fla": HAS_FLA,
        "results": results,
    }
    if args.json_out:
        out_path = Path(args.json_out)
        os.makedirs(out_path.parent, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
