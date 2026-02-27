"""
Benchmark B1-B3: Cross-Entropy Kernel Comparison
=================================================

Tests different CE implementations to optimize the TRAINING step:

  B1: Standard (lm_head + PyTorch CE) vs Codebase Fused Linear CE
  B3: Memory comparison — which CE uses least VRAM?

CE was 29% of step time at 4096 in profiling. At 512 it will be smaller
proportionally, but still worth optimizing since it runs every training step.

Usage:
    python benchmark_b1b3_cross_entropy.py --dtype bf16
    python benchmark_b1b3_cross_entropy.py --dtype fp16
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Resolve imports ──────────────────────────────────────────────────────────

def _setup_imports():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    # Try latest code first — Compiled_new has fused projections
    candidates = [
        os.path.join(repo_root, "experiments", "tests",
                     "Test_14_Compiled_new", "code"),
        os.path.join(repo_root, "experiments", "tests",
                     "Test_16_New_Code", "code"),
        os.path.join(repo_root, "experiments", "tests",
                     "Test_14_gsa_only_liger_kernels_1000steps-28k", "code"),
        os.path.join(repo_root, "experiments", "tests",
                     "Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3", "code"),
    ]
    for code_dir in candidates:
        if os.path.isdir(os.path.join(code_dir, "src", "kernels")):
            if code_dir not in sys.path:
                sys.path.insert(0, code_dir)
            print(f"  Using codebase: {os.path.basename(os.path.dirname(code_dir))}")
            return code_dir
    return None

CODE_DIR = _setup_imports()


@dataclass
class BenchConfig:
    batch_size: int = 4
    hidden_size: int = 4096
    vocab_size: int = 131072
    seq_lens: Tuple[int, ...] = (512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── Benchmarking Utility ─────────────────────────────────────────────────────

def bench_fn(fn, warmup, iters, label):
    for _ in range(warmup):
        fn()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    drop = max(1, iters // 10)
    times = times[drop:]
    return {
        "label": label,
        "avg_ms": sum(times) / len(times),
        "peak_mem_gb": peak_mem,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  B1: PyTorch CE vs Triton Fused CE
# ══════════════════════════════════════════════════════════════════════════════

def run_b1_ce_comparison(cfg):
    """Compare standard CE vs codebase fused CE (FusedLinearCrossEntropyLoss)."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    D, V = cfg.hidden_size, cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  B1: Standard (lm_head + PyTorch CE) vs Codebase Fused Linear CE")
    print(f"  Codebase CE fuses lm_head + CE into one Triton kernel (chunks internally)")
    print(f"  Vocab: {V} | Hidden: {D} | dtype={cfg.dtype_str}")
    print(f"{'━'*80}")

    # Check for codebase FusedLinearCE
    fused_ce_cls = None
    try:
        from src.kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss
        fused_ce_cls = FusedLinearCrossEntropyLoss
        print("  ✅ Codebase FusedLinearCrossEntropyLoss available (Triton)")
    except Exception as e:
        print(f"  ⚠️  Codebase fused CE unavailable: {e}")


    results = []
    lm_head_weight = torch.randn(V, D, device=device, dtype=dtype)

    print(f"\n  {'Seq Len':>8}  {'Method':>22}  {'Time (ms)':>10}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        B = cfg.batch_size
        N = B * T

        # Method 1: Separate lm_head + PyTorch CE
        sep_result = None
        try:
            def separate_ce():
                h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
                targets = torch.randint(0, V, (N,), device=device)
                logits = F.linear(h, lm_head_weight)
                loss = F.cross_entropy(logits, targets)
                loss.backward()

            sep_result = bench_fn(separate_ce, cfg.warmup_iters, cfg.bench_iters, f"separate_T{T}")
            results.append({"seq_len": T, "method": "Separate", **sep_result})
            print(f"  {T:>8}  {'Separate (lm+CE)':>22}  {sep_result['avg_ms']:>10.2f}  {sep_result['peak_mem_gb']:>10.2f}")
        except Exception as e:
            print(f"  {T:>8}  {'Separate (lm+CE)':>22}  OOM ❌")

        # Method 2: Codebase FusedLinearCE (Triton)
        if fused_ce_cls is not None:
            try:
                fused_ce = fused_ce_cls(max_chunk_gb=2.0)

                def codebase_fused():
                    h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
                    targets = torch.randint(0, V, (N,), device=device)
                    loss = fused_ce(h, lm_head_weight, targets)
                    loss.backward()

                fused_result = bench_fn(codebase_fused, cfg.warmup_iters, cfg.bench_iters, f"fused_T{T}")
                results.append({"seq_len": T, "method": "Codebase FusedCE", **fused_result})
                speedup = sep_result['avg_ms'] / fused_result['avg_ms'] if sep_result else 0
                print(f"  {T:>8}  {'Codebase FusedCE':>22}  {fused_result['avg_ms']:>10.2f}  {fused_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
            except Exception as e:
                print(f"  {T:>8}  {'Codebase FusedCE':>22}  FAILED: {e}")


        print()
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  B2: Separate lm_head + CE vs Fused Linear CE
# ══════════════════════════════════════════════════════════════════════════════

def run_b2_fused_linear_ce(cfg):
    """B2 is now covered by B1 (which compares separate vs fused). Return empty."""
    print(f"\n{'━'*80}")
    print(f"  B2: (Covered by B1 — separate vs fused comparison already done)")
    print(f"{'━'*80}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
#  B3: Memory Comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_b3_memory(cfg):
    """Compare peak memory of each CE approach at T=512."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    D, V = cfg.hidden_size, cfg.vocab_size
    T = 512
    B = cfg.batch_size
    N = B * T

    print(f"\n{'━'*80}")
    print(f"  B3: Peak Memory Comparison at T={T}")
    print(f"  Key question: Does FusedLinearCE avoid materializing [{N}, {V}] logits?")
    print(f"  Logit tensor size: [{N}, {V}] = {N * V * 2 / 1e9:.2f} GB (fp16)")
    print(f"{'━'*80}")

    lm_head_weight = torch.randn(V, D, device=device, dtype=dtype)

    methods = {}

    # Separate
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        logits = F.linear(h, lm_head_weight)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        methods["Separate (lm+CE)"] = torch.cuda.max_memory_allocated() / 1e9
        del h, targets, logits, loss
        torch.cuda.empty_cache()
    except Exception:
        methods["Separate (lm+CE)"] = float('inf')

    # Codebase FusedLinearCE
    try:
        from src.kernels.triton_cross_entropy import FusedLinearCrossEntropyLoss
        fused_ce = FusedLinearCrossEntropyLoss(max_chunk_gb=2.0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        loss = fused_ce(h, lm_head_weight, targets)
        loss.backward()
        methods["Codebase FusedCE"] = torch.cuda.max_memory_allocated() / 1e9
        del h, targets, loss
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  (Codebase FusedCE skipped: {e})")



    print(f"\n  {'Method':>22}  {'Peak Mem (GB)':>14}  {'Savings':>10}")
    print(f"  {'─'*22}  {'─'*14}  {'─'*10}")
    baseline = list(methods.values())[0] if methods else 0
    for name, mem in methods.items():
        if mem == float('inf'):
            print(f"  {name:>22}  {'OOM':>14}")
        else:
            saved = f"{baseline - mem:.2f} GB" if mem < baseline else "baseline"
            print(f"  {name:>22}  {mem:>14.2f}  {saved:>10}")

    return methods


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(cfg):
    print(f"\n{'='*80}")
    print(f"  BENCHMARK B1-B3: Cross-Entropy Comparison")
    print(f"  Optimizing CE for TRAINING step (29% of step time at 4096)")
    print(f"  GPU: {torch.cuda.get_device_name()} | dtype={cfg.dtype_str}")
    print(f"{'='*80}")

    b1_results = run_b1_ce_comparison(cfg)
    b2_results = run_b2_fused_linear_ce(cfg)
    b3_results = run_b3_memory(cfg)

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/b1b3_cross_entropy.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "B1B3_cross_entropy",
            "config": {
                "batch_size": cfg.batch_size, "hidden_size": cfg.hidden_size,
                "vocab_size": cfg.vocab_size, "dtype": cfg.dtype_str,
                "gpu": torch.cuda.get_device_name(),
            },
            "b1_ce_comparison": b1_results,
            "b2_fused_linear_ce": b2_results,
            "b3_memory": b3_results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="B1-B3: Cross-Entropy Comparison")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-lens", type=str, default="512,1024,2048,4096")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size, warmup_iters=args.warmup,
        bench_iters=args.iters, dtype_str=args.dtype,
        seq_lens=tuple(int(x) for x in args.seq_lens.split(",")),
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        exit(1)

    run_benchmark(cfg)
