"""
Benchmark B1-B3: Cross-Entropy Kernel Comparison
=================================================

Tests different CE implementations to optimize the TRAINING step:

  B1: Standard PyTorch CE vs Triton fused CE (from codebase)
  B2: Liger FusedLinearCE (fuses lm_head + CE into one kernel) vs separate
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
    candidates = [
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
    """Compare standard CE vs Triton fused CE from codebase."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    V = cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  B1: Standard PyTorch CE vs Triton Fused CE")
    print(f"  Vocab: {V} | dtype={cfg.dtype_str}")
    print(f"{'━'*80}")

    # Check for Triton CE from codebase
    triton_ce = None
    try:
        from src.kernels.triton_cross_entropy import triton_cross_entropy
        triton_ce = triton_cross_entropy
        print("  ✅ Triton fused CE available from codebase")
    except Exception as e:
        print(f"  ⚠️  Triton CE unavailable: {e}")

    results = []

    print(f"\n  {'Seq Len':>8}  {'Method':>16}  {'Time (ms)':>10}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        B = cfg.batch_size
        N = B * T  # total tokens

        # Standard PyTorch CE
        def pytorch_ce():
            logits = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
            targets = torch.randint(0, V, (N,), device=device)
            loss = F.cross_entropy(logits, targets)
            loss.backward()

        try:
            pt_result = bench_fn(pytorch_ce, cfg.warmup_iters, cfg.bench_iters, f"pytorch_T{T}")
            results.append({"seq_len": T, "method": "PyTorch CE", **pt_result})
        except Exception as e:
            print(f"  {T:>8}  {'PyTorch CE':>16}  FAILED: {e}")
            pt_result = None

        # Triton fused CE (from codebase)
        tri_result = None
        if triton_ce is not None:
            def triton_ce_fn():
                logits = torch.randn(N, V, device=device, dtype=dtype, requires_grad=True)
                targets = torch.randint(0, V, (N,), device=device)
                loss = triton_ce(logits, targets)
                loss.backward()

            try:
                tri_result = bench_fn(triton_ce_fn, cfg.warmup_iters, cfg.bench_iters, f"triton_T{T}")
                results.append({"seq_len": T, "method": "Triton CE", **tri_result})
            except Exception as e:
                print(f"  {T:>8}  {'Triton CE':>16}  FAILED: {e}")
                tri_result = None

        # Print comparison
        if pt_result:
            print(f"  {T:>8}  {'PyTorch CE':>16}  {pt_result['avg_ms']:>10.2f}  {pt_result['peak_mem_gb']:>10.2f}")
        if tri_result:
            speedup = pt_result['avg_ms'] / tri_result['avg_ms'] if pt_result else 0
            print(f"  {T:>8}  {'Triton CE':>16}  {tri_result['avg_ms']:>10.2f}  {tri_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
        if not tri_result and pt_result:
            print(f"  {T:>8}  {'Triton CE':>16}  N/A")
        print()

        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  B2: Separate lm_head + CE vs Fused Linear CE
# ══════════════════════════════════════════════════════════════════════════════

def run_b2_fused_linear_ce(cfg):
    """Compare separate lm_head+CE vs Liger FusedLinearCE."""
    device = torch.device("cuda")
    dtype = cfg.dtype
    D, V = cfg.hidden_size, cfg.vocab_size

    print(f"\n{'━'*80}")
    print(f"  B2: Separate (lm_head → CE) vs FusedLinearCE")
    print(f"  This is the key comparison: does fusing the projection save time?")
    print(f"  lm_head: {D} → {V} | dtype={cfg.dtype_str}")
    print(f"{'━'*80}")

    # Check for Liger FusedLinearCE
    liger_available = False
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        liger_available = True
        print("  ✅ Liger FusedLinearCE available")
    except Exception as e:
        print(f"  ⚠️  Liger unavailable: {e}")

    # Check for codebase Triton CE
    triton_ce = None
    try:
        from src.kernels.triton_cross_entropy import triton_cross_entropy
        triton_ce = triton_cross_entropy
        print("  ✅ Triton CE available (for separate path)")
    except Exception:
        pass

    results = []

    print(f"\n  {'Seq Len':>8}  {'Method':>22}  {'Time (ms)':>10}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*8}")

    lm_head = nn.Linear(D, V, bias=False, device=device, dtype=dtype)
    lm_head_weight = lm_head.weight.data  # [V, D]

    for T in cfg.seq_lens:
        B = cfg.batch_size
        N = B * T

        # Method 1: Separate lm_head + PyTorch CE
        def separate_ce():
            h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
            targets = torch.randint(0, V, (N,), device=device)
            logits = F.linear(h, lm_head_weight)
            loss = F.cross_entropy(logits, targets)
            loss.backward()

        try:
            sep_result = bench_fn(separate_ce, cfg.warmup_iters, cfg.bench_iters, f"separate_T{T}")
            results.append({"seq_len": T, "method": "Separate", **sep_result})
        except Exception as e:
            print(f"  {T:>8}  {'Separate':>22}  FAILED: {e}")
            sep_result = None

        # Method 2: Separate lm_head + Triton CE
        tri_sep_result = None
        if triton_ce is not None:
            def separate_triton_ce():
                h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
                targets = torch.randint(0, V, (N,), device=device)
                logits = F.linear(h, lm_head_weight)
                loss = triton_ce(logits, targets)
                loss.backward()

            try:
                tri_sep_result = bench_fn(separate_triton_ce, cfg.warmup_iters, cfg.bench_iters, f"sep_triton_T{T}")
                results.append({"seq_len": T, "method": "Separate+TritonCE", **tri_sep_result})
            except Exception as e:
                tri_sep_result = None

        # Method 3: Liger FusedLinearCE
        liger_result = None
        if liger_available:
            fused_ce = LigerFusedLinearCrossEntropyLoss()

            def fused_linear_ce():
                h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
                targets = torch.randint(0, V, (N,), device=device)
                loss = fused_ce(h, lm_head_weight, targets)
                loss.backward()

            try:
                liger_result = bench_fn(fused_linear_ce, cfg.warmup_iters, cfg.bench_iters, f"fused_T{T}")
                results.append({"seq_len": T, "method": "FusedLinearCE", **liger_result})
            except Exception as e:
                print(f"  {T:>8}  {'FusedLinearCE':>22}  FAILED: {e}")

        # Print results
        if sep_result:
            print(f"  {T:>8}  {'Separate (lm+CE)':>22}  {sep_result['avg_ms']:>10.2f}  {sep_result['peak_mem_gb']:>10.2f}")
        if tri_sep_result:
            sp = sep_result['avg_ms'] / tri_sep_result['avg_ms'] if sep_result else 0
            print(f"  {T:>8}  {'Separate+TritonCE':>22}  {tri_sep_result['avg_ms']:>10.2f}  {tri_sep_result['peak_mem_gb']:>10.2f}  {sp:>7.2f}×")
        if liger_result:
            sp = sep_result['avg_ms'] / liger_result['avg_ms'] if sep_result else 0
            print(f"  {T:>8}  {'Liger FusedLinearCE':>22}  {liger_result['avg_ms']:>10.2f}  {liger_result['peak_mem_gb']:>10.2f}  {sp:>7.2f}×")
        print()

        torch.cuda.empty_cache()

    return results


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
    print(f"  Key question: Does FusedLinearCE avoid materializing [N, V] logits?")
    print(f"  Logit tensor size: [{N}, {V}] = {N * V * 2 / 1e9:.2f} GB (fp16)")
    print(f"{'━'*80}")

    lm_head_weight = torch.randn(V, D, device=device, dtype=dtype)

    methods = {}

    # Separate
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

    # Triton CE
    try:
        from src.kernels.triton_cross_entropy import triton_cross_entropy
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        logits = F.linear(h, lm_head_weight)
        loss = triton_cross_entropy(logits, targets)
        loss.backward()
        methods["Separate+TritonCE"] = torch.cuda.max_memory_allocated() / 1e9
        del h, targets, logits, loss
        torch.cuda.empty_cache()
    except Exception:
        pass

    # Liger FusedLinearCE
    try:
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
        fused_ce = LigerFusedLinearCrossEntropyLoss()
        torch.cuda.reset_peak_memory_stats()
        h = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
        targets = torch.randint(0, V, (N,), device=device)
        loss = fused_ce(h, lm_head_weight, targets)
        loss.backward()
        methods["Liger FusedLinearCE"] = torch.cuda.max_memory_allocated() / 1e9
        del h, targets, loss
        torch.cuda.empty_cache()
    except Exception:
        pass

    print(f"\n  {'Method':>22}  {'Peak Mem (GB)':>14}  {'Savings':>10}")
    print(f"  {'─'*22}  {'─'*14}  {'─'*10}")
    baseline = list(methods.values())[0] if methods else 0
    for name, mem in methods.items():
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
