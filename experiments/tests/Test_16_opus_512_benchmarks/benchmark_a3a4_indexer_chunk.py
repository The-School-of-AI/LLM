"""
Benchmark A3+A4: GSA Indexer Scaling & DeltaNet Chunk Size Sweep
================================================================

A3: Does the GSA indexer's overhead scale down at 512, or is it fixed cost?
    - Profiles the indexer (fused_indexer_topk) at each seq length
    - Answers: "If GSA is used at 512, how much time is JUST the indexer?"

A4: What chunk_size is optimal for DeltaNet's fla kernel at 512?
    - Sweeps chunk_size in [16, 32, 64, 128, 256, 512]
    - At seq_len=512, chunk_size=512 means ONE chunk (no recurrence) — possibly fastest

Usage:
    python benchmark_a3a4_indexer_chunk.py --dtype bf16        # AWS A100
    python benchmark_a3a4_indexer_chunk.py --dtype fp16        # Colab T4
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
    # Try latest code first (Test_16_New_Code with BLOCK_Q=1 sparse attn),
    # then fall back to Test_14 variants
    candidates = [
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
    # DeltaNet
    dn_num_heads: int = 32
    dn_head_dim: int = 128
    # GSA
    gsa_num_heads: int = 16
    gsa_head_dim: int = 256
    gsa_k_base: int = 128
    gsa_k_min: int = 32
    gsa_k_max: int = 256
    gsa_indexer_heads: int = 4
    gsa_d_idx: int = 128
    # Benchmark
    seq_lens: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ══════════════════════════════════════════════════════════════════════════════
#  A3: GSA INDEXER SCALING
# ══════════════════════════════════════════════════════════════════════════════

def run_a3_indexer_scaling(cfg: BenchConfig):
    """Test whether the GSA indexer overhead scales with sequence length."""
    device = torch.device("cuda")
    dtype = cfg.dtype

    print(f"\n{'='*80}")
    print(f"  BENCHMARK A3: GSA Indexer Scaling at Different Seq Lengths")
    print(f"  Config: B={cfg.batch_size} | indexer_heads={cfg.gsa_indexer_heads} | d_idx={cfg.gsa_d_idx} | k_base={cfg.gsa_k_base}")
    print(f"  GPU: {torch.cuda.get_device_name()} | dtype={cfg.dtype_str}")
    print(f"{'='*80}\n")

    results = []

    # Try to import real fused_indexer_topk
    has_fused_indexer = False
    try:
        from src.kernels.triton_indexer_streaming import fused_indexer_topk
        has_fused_indexer = True
        print("  ✅ Using REAL fused_indexer_topk (Triton)")
    except Exception as e:
        print(f"  ⚠️  fused_indexer_topk unavailable: {e}")
        print("  Using PyTorch simulation for indexer")

    # Indexer projections (same as real model)
    W_Iq = nn.Linear(cfg.hidden_size, cfg.gsa_indexer_heads * cfg.gsa_d_idx,
                      bias=False, device=device, dtype=dtype)
    W_Ik = nn.Linear(cfg.hidden_size, cfg.gsa_d_idx,
                      bias=False, device=device, dtype=dtype)
    W_Iw = nn.Linear(cfg.hidden_size, cfg.gsa_indexer_heads,
                      bias=False, device=device, dtype=dtype)
    gate_bias = nn.Parameter(torch.zeros(cfg.gsa_indexer_heads, device=device, dtype=dtype))
    variance_ema = torch.tensor(1.0, device=device)

    print(f"\n  {'Seq Len':>8}  {'Indexer (ms)':>14}  {'Per Token (µs)':>16}  {'Mem (GB)':>10}  {'vs 4096':>8}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*16}  {'─'*10}  {'─'*8}")

    t4096_time = None

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        # Warmup
        for _ in range(cfg.warmup_iters):
            with torch.no_grad():
                q_I = W_Iq(x).view(cfg.batch_size, T, cfg.gsa_indexer_heads, cfg.gsa_d_idx)
                k_I = W_Ik(x)
                w_raw = W_Iw(x)

                if has_fused_indexer:
                    _, _, _ = fused_indexer_topk(
                        q=q_I, k=k_I, w=w_raw, b=gate_bias,
                        scale=1.0 / (cfg.gsa_d_idx ** 0.5), causal=True,
                        k_base=cfg.gsa_k_base, k_min=cfg.gsa_k_min, k_max=cfg.gsa_k_max,
                        variance_ema=variance_ema, is_training=False, sink_size=4,
                    )
                else:
                    # PyTorch simulation
                    import math
                    q_I_t = q_I.permute(0, 2, 1, 3)
                    k_I_exp = k_I.unsqueeze(1).expand(cfg.batch_size, cfg.gsa_indexer_heads, T, cfg.gsa_d_idx)
                    scores = torch.matmul(q_I_t, k_I_exp.transpose(-1, -2)) / math.sqrt(cfg.gsa_d_idx)
                    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
                    scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    scores_avg = scores.mean(dim=1)
                    _, _ = scores_avg.topk(min(cfg.gsa_k_base, T), dim=-1)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        idx_times = []
        for _ in range(cfg.bench_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                q_I = W_Iq(x).view(cfg.batch_size, T, cfg.gsa_indexer_heads, cfg.gsa_d_idx)
                k_I = W_Ik(x)
                w_raw = W_Iw(x)

                if has_fused_indexer:
                    _, _, _ = fused_indexer_topk(
                        q=q_I, k=k_I, w=w_raw, b=gate_bias,
                        scale=1.0 / (cfg.gsa_d_idx ** 0.5), causal=True,
                        k_base=cfg.gsa_k_base, k_min=cfg.gsa_k_min, k_max=cfg.gsa_k_max,
                        variance_ema=variance_ema, is_training=False, sink_size=4,
                    )
                else:
                    import math
                    q_I_t = q_I.permute(0, 2, 1, 3)
                    k_I_exp = k_I.unsqueeze(1).expand(cfg.batch_size, cfg.gsa_indexer_heads, T, cfg.gsa_d_idx)
                    scores = torch.matmul(q_I_t, k_I_exp.transpose(-1, -2)) / math.sqrt(cfg.gsa_d_idx)
                    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
                    scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    scores_avg = scores.mean(dim=1)
                    _, _ = scores_avg.topk(min(cfg.gsa_k_base, T), dim=-1)

            torch.cuda.synchronize()
            idx_times.append((time.perf_counter() - t0) * 1000)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        drop = max(1, cfg.bench_iters // 10)
        idx_times = idx_times[drop:]
        avg = sum(idx_times) / len(idx_times)
        per_token = avg / T * 1000  # microseconds per token

        if T == 4096:
            t4096_time = avg
        ratio = f"{avg/t4096_time:.2f}×" if t4096_time else "1.00×"

        r = {"seq_len": T, "indexer_ms": avg, "per_token_us": per_token, "peak_mem_gb": peak_mem}
        results.append(r)
        print(f"  {T:>8}  {avg:>14.3f}  {per_token:>16.2f}  {peak_mem:>10.2f}  {ratio:>8}")

        torch.cuda.empty_cache()

    # Summary
    print(f"\n  Key finding: Indexer at 512 = {results[1]['indexer_ms']:.3f}ms, "
          f"at 4096 = {results[-1]['indexer_ms']:.3f}ms "
          f"({results[-1]['indexer_ms']/results[1]['indexer_ms']:.1f}× slower)")
    scales_linearly = results[-1]['indexer_ms'] / results[1]['indexer_ms'] > 5
    if scales_linearly:
        print("  → Indexer scales roughly linearly with T ✅ (good scaling)")
    else:
        print("  → Indexer has significant fixed overhead ⚠️ (doesn't shrink much at 512)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  A4: DELTANET CHUNK SIZE SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_a4_chunk_sweep(cfg: BenchConfig):
    """Sweep chunk_size for DeltaNet's fla kernel at seq_len=512."""
    device = torch.device("cuda")
    dtype = cfg.dtype

    print(f"\n{'='*80}")
    print(f"  BENCHMARK A4: DeltaNet Chunk Size Sweep at T=512")
    print(f"  Config: B={cfg.batch_size} | H={cfg.dn_num_heads} | D={cfg.dn_head_dim}")
    print(f"  GPU: {torch.cuda.get_device_name()} | dtype={cfg.dtype_str}")
    print(f"{'='*80}\n")

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        print("  ✅ fla kernel available")
    except ImportError:
        print("  ❌ fla not installed. Run: pip install git+https://github.com/fla-org/flash-linear-attention.git")
        return []

    results = []
    chunk_sizes = [16, 32, 64, 128, 256, 512]
    T = 512  # Fixed at OPUS scoring length
    B, H, D = cfg.batch_size, cfg.dn_num_heads, cfg.dn_head_dim

    print(f"  {'Chunk':>8}  {'Chunks':>8}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}  {'vs best':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    best_total = float('inf')

    for chunk_size in chunk_sizes:
        if chunk_size > T:
            continue

        n_chunks = T // chunk_size

        def make_inputs():
            return (
                torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                torch.randn(B, T, H, device=device, dtype=torch.float32, requires_grad=True),
                torch.rand(B, T, H, device=device, dtype=torch.float32, requires_grad=True),
            )

        def kernel_fn(q, k, v, g, beta):
            o, _ = chunk_gated_delta_rule(
                q, k, v, g, beta, scale=1.0,
                output_final_state=False,
                chunk_size=chunk_size,
            )
            return o

        # Warmup
        for _ in range(cfg.warmup_iters):
            inputs = make_inputs()
            out = kernel_fn(*inputs)
            out.sum().backward()
            del inputs, out
        torch.cuda.empty_cache()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        fwd_times, bwd_times = [], []
        for _ in range(cfg.bench_iters):
            inputs = make_inputs()
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = kernel_fn(*inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            loss = out.sum()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()

            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((t3 - t2) * 1000)
            del inputs, out

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        drop = max(1, cfg.bench_iters // 10)
        fwd_avg = sum(fwd_times[drop:]) / len(fwd_times[drop:])
        bwd_avg = sum(bwd_times[drop:]) / len(bwd_times[drop:])
        total = fwd_avg + bwd_avg

        if total < best_total:
            best_total = total

        r = {
            "chunk_size": chunk_size, "n_chunks": n_chunks,
            "fwd_ms": fwd_avg, "bwd_ms": bwd_avg,
            "total_ms": total, "peak_mem_gb": peak_mem,
        }
        results.append(r)

        ratio = f"{total/best_total:.2f}×" if best_total > 0 else "1.00×"
        marker = " ◀ best" if total <= best_total else ""
        print(f"  {chunk_size:>8}  {n_chunks:>8}  {fwd_avg:>10.3f}  {bwd_avg:>10.3f}  {total:>12.3f}  {peak_mem:>10.2f}  {ratio:>8}{marker}")

        torch.cuda.empty_cache()

    # Find best
    best = min(results, key=lambda r: r["total_ms"])
    default = next((r for r in results if r["chunk_size"] == 64), results[0])
    print(f"\n  Best chunk_size: {best['chunk_size']} ({best['total_ms']:.3f}ms)")
    print(f"  Default (64): {default['total_ms']:.3f}ms")
    if best['chunk_size'] != 64:
        speedup = default['total_ms'] / best['total_ms']
        print(f"  Switching to chunk_size={best['chunk_size']} gives {speedup:.2f}× speedup")
    else:
        print(f"  Default chunk_size=64 is already optimal")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(cfg: BenchConfig):
    a3_results = run_a3_indexer_scaling(cfg)
    a4_results = run_a4_chunk_sweep(cfg)

    # Save all results
    os.makedirs("results", exist_ok=True)
    out_path = "results/a3a4_indexer_chunk.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "A3A4_indexer_chunk",
            "config": {
                "batch_size": cfg.batch_size,
                "dtype": cfg.dtype_str,
                "gpu": torch.cuda.get_device_name(),
            },
            "a3_indexer_scaling": a3_results,
            "a4_chunk_sweep": a4_results,
        }, f, indent=2)
    print(f"\n  All results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A3+A4: Indexer Scaling & Chunk Size Sweep")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
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
