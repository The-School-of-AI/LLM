"""
Benchmark A2: GSA (Gated Sparse Attention) vs Flash Attention at seq_len=512
=============================================================================

Tests whether standard Flash Attention is faster than GSA's sparse attention
pipeline at short sequences (OPUS scoring at 512).

GSA includes:
  1. Lightning Indexer (learned sparse pattern selection)
  2. Q/K/V projections
  3. Sparse attention (only attend to k selected keys per query)
  4. Dual output gating

Since the Triton sparse attention kernel may not be available on Colab,
we benchmark TWO variants of GSA:
  - GSA-Simulated: Uses PyTorch gather + dense attention on selected indices
    (same computation, no custom Triton kernel needed)
  - FlashAttn: Standard SDPA with Flash Attention 2 backend

Sweeps seq_len: [256, 512, 1024, 2048, 4096] to find the crossover point.

Usage:
    python benchmark_a2_gsa_vs_flash.py --dtype fp16     # Colab T4
    python benchmark_a2_gsa_vs_flash.py --dtype bf16     # A100/H100
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    """1B model GSA config."""
    batch_size: int = 4
    hidden_size: int = 4096
    # GSA config from ModelConfig
    gsa_num_heads: int = 16       # hidden_size / gsa_head_dim = 4096 / 256
    gsa_head_dim: int = 256
    gsa_k_base: int = 128        # keys per query
    gsa_k_min: int = 32
    gsa_k_max: int = 256
    gsa_indexer_heads: int = 4
    gsa_d_idx: int = 128         # indexer projection dim
    seq_lens: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── GSA Benchmark (Simulated with PyTorch) ───────────────────────────────────

class GSABench(nn.Module):
    """
    Simulated GSA layer matching 1B model's GatedSparseAttention.
    Uses PyTorch gather + dense attention on selected keys (no Triton needed).
    Includes: Indexer → Q/K/V proj → sparse gather → attention → gate → output proj
    """
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H = cfg.gsa_num_heads
        D = cfg.gsa_head_dim
        C = cfg.hidden_size
        self.num_heads = H
        self.head_dim = D
        self.hidden_size = C
        self.k_base = cfg.gsa_k_base
        self.k_max = cfg.gsa_k_max
        self.indexer_heads = cfg.gsa_indexer_heads
        self.d_idx = cfg.gsa_d_idx

        # Lightning Indexer projections
        self.W_Iq = nn.Linear(C, cfg.gsa_indexer_heads * cfg.gsa_d_idx, bias=False, device=device, dtype=dtype)
        self.W_Ik = nn.Linear(C, cfg.gsa_d_idx, bias=False, device=device, dtype=dtype)
        self.W_Iw = nn.Linear(C, cfg.gsa_indexer_heads, bias=False, device=device, dtype=dtype)
        self.gate_bias = nn.Parameter(torch.zeros(cfg.gsa_indexer_heads, device=device, dtype=dtype))

        # Attention projections
        self.W_q = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.W_k = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.W_v = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(C, C, bias=False, device=device, dtype=dtype)

        # Dual gating
        self.W_go = nn.Linear(C, C, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # ── Step 1: Lightning Indexer (find top-k keys per query) ────────
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)  # [B, T, 4, 128]
        k_I = self.W_Ik(x)  # [B, T, 128]
        scale_idx = 1.0 / math.sqrt(self.d_idx)

        # Simplified indexer: compute importance scores and select top-k
        # Real model uses fused_indexer_topk Triton kernel
        # We simulate with: scores = Q_I @ K_I^T, then topk
        k_I_expanded = k_I.unsqueeze(1).expand(B, self.indexer_heads, T, self.d_idx)  # [B, 4, T, 128]
        q_I_t = q_I.permute(0, 2, 1, 3)  # [B, 4, T, 128]
        
        # Use chunked computation to avoid T×T matrix at large T
        k_limit = min(self.k_base, T)
        
        # For each query, score all keys and pick top-k (causal)
        # At 512 tokens, this is a [B, 4, 512, 512] matmul — manageable
        scores = torch.matmul(q_I_t, k_I_expanded.transpose(-1, -2)) * scale_idx  # [B, 4, T, T]
        
        # Causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Average across indexer heads and take top-k
        scores_avg = scores.mean(dim=1)  # [B, T, T]
        _, top_indices = scores_avg.topk(k_limit, dim=-1)  # [B, T, k_limit]

        # ── Step 2: Q/K/V Projections ────────────────────────────────────
        q = self.W_q(x).view(B, T, H, D)  # [B, T, 16, 256]
        k = self.W_k(x).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)

        scale_attn = 1.0 / math.sqrt(D)

        # ── Step 3: Sparse Attention (gather + attend to selected keys) ──
        # Gather selected K, V using top_indices
        # top_indices: [B, T, k_limit] → need to gather from K, V: [B, T, H, D]
        idx_expanded = top_indices.unsqueeze(2).unsqueeze(-1).expand(B, T, H, k_limit, D)  # [B, T, H, k, D]
        k_selected = torch.gather(
            k.unsqueeze(2).expand(B, T, H, T, D),      # [B, T, H, T, D] — expanded view
            dim=3,                                        # gather along T dimension
            index=idx_expanded                            # [B, T, H, k, D]
        )  # [B, T, H, k, D]
        v_selected = torch.gather(
            v.unsqueeze(2).expand(B, T, H, T, D),
            dim=3,
            index=idx_expanded
        )  # [B, T, H, k, D]

        # Sparse attention: Q @ K_selected^T → softmax → @ V_selected
        attn_scores = torch.matmul(
            q.unsqueeze(3),          # [B, T, H, 1, D]
            k_selected.transpose(-1, -2)  # [B, T, H, D, k]
        ).squeeze(3) * scale_attn    # [B, T, H, k]

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, H, k]

        o = torch.matmul(
            attn_weights.unsqueeze(3),  # [B, T, H, 1, k]
            v_selected                   # [B, T, H, k, D]
        ).squeeze(3)                     # [B, T, H, D]

        # ── Step 4: Output Gate + Projection ─────────────────────────────
        o = o.contiguous().view(B, T, C)
        g_o = torch.sigmoid(self.W_go(x))

        return self.o_proj(o * g_o)


# ── Flash Attention Benchmark (same as A1 but with GSA dimensions) ───────────

class FlashAttnBench(nn.Module):
    """
    Standard MHA using Flash Attention 2, same dimensions as GSA.
    """
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H = cfg.gsa_num_heads    # 16 heads
        D = cfg.gsa_head_dim     # 256 dim
        C = cfg.hidden_size

        self.num_heads = H
        self.head_dim = D

        self.q_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(H * D, C, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o = o.transpose(1, 2).reshape(B, T, H * D)
        return self.o_proj(o)


# ── Benchmarking Utility ─────────────────────────────────────────────────────

def benchmark_forward_backward(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    label: str,
) -> Dict[str, float]:
    """Benchmark forward + backward, return timing & memory stats."""
    model.train()

    # Warmup
    for _ in range(warmup):
        x_in = x.clone().requires_grad_(True)
        out = model(x_in)
        loss = out.sum()
        loss.backward()
        del out, loss
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    bwd_times = []

    for _ in range(iters):
        x_in = x.clone().requires_grad_(True)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = model(x_in)
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

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    drop = max(1, iters // 10)
    fwd_times = fwd_times[drop:]
    bwd_times = bwd_times[drop:]

    fwd_avg = sum(fwd_times) / len(fwd_times)
    bwd_avg = sum(bwd_times) / len(bwd_times)

    return {
        "label": label,
        "fwd_ms": fwd_avg,
        "bwd_ms": bwd_avg,
        "total_ms": fwd_avg + bwd_avg,
        "peak_mem_gb": peak_mem,
    }


# ── Main Benchmark ──────────────────────────────────────────────────────────

def run_benchmark(cfg: BenchConfig):
    device = torch.device("cuda")
    dtype = cfg.dtype
    print(f"\n{'='*80}")
    print(f"  BENCHMARK A2: GSA (Gated Sparse Attention) vs Flash Attention")
    print(f"  Config: 1B model | B={cfg.batch_size} | H={cfg.gsa_num_heads} | D={cfg.gsa_head_dim} | k_base={cfg.gsa_k_base}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Warmup: {cfg.warmup_iters} | Bench: {cfg.bench_iters} iters | dtype={cfg.dtype_str}")
    print(f"{'='*80}\n")

    results = []

    print("━" * 80)
    print("  Full Layer: GSA (indexer + sparse attn + gate) vs Flash Attention (dense)")
    print("━" * 80)
    print(f"  {'Seq Len':>8}  {'Kernel':>14}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        # GSA (simulated)
        try:
            gsa = GSABench(cfg, device, dtype)
            gsa_result = benchmark_forward_backward(gsa, x, cfg.warmup_iters, cfg.bench_iters, f"gsa_T{T}")
            gsa_result["seq_len"] = T
            gsa_result["kernel"] = "GSA-Sim"
            results.append(gsa_result)
            del gsa
        except Exception as e:
            print(f"  {T:>8}  {'GSA-Sim':>14}  FAILED: {e}")
            gsa_result = None

        # Flash Attention
        try:
            flash = FlashAttnBench(cfg, device, dtype)
            fa_result = benchmark_forward_backward(flash, x, cfg.warmup_iters, cfg.bench_iters, f"flash_T{T}")
            fa_result["seq_len"] = T
            fa_result["kernel"] = "FlashAttn"
            results.append(fa_result)
            del flash
        except Exception as e:
            print(f"  {T:>8}  {'FlashAttn':>14}  FAILED: {e}")
            fa_result = None

        # Print comparison
        if gsa_result and fa_result:
            speedup = gsa_result["total_ms"] / fa_result["total_ms"]
            print(f"  {T:>8}  {'GSA-Sim':>14}  {gsa_result['fwd_ms']:>10.2f}  {gsa_result['bwd_ms']:>10.2f}  {gsa_result['total_ms']:>12.2f}  {gsa_result['peak_mem_gb']:>10.2f}")
            print(f"  {T:>8}  {'FlashAttn':>14}  {fa_result['fwd_ms']:>10.2f}  {fa_result['bwd_ms']:>10.2f}  {fa_result['total_ms']:>12.2f}  {fa_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
            print()

        torch.cuda.empty_cache()

    # ── Component Breakdown: Indexer Cost ────────────────────────────────
    print("\n" + "━" * 80)
    print("  Component Breakdown: Indexer Cost at Each Seq Length")
    print("  (How much of GSA's time is spent just finding which keys to attend to?)")
    print("━" * 80)
    print(f"  {'Seq Len':>8}  {'Indexer (ms)':>14}  {'% of GSA Fwd':>14}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}")

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        try:
            gsa = GSABench(cfg, device, dtype)
            gsa.eval()

            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = gsa(x)

            # Measure indexer only
            idx_times = []
            for _ in range(30):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    q_I = gsa.W_Iq(x).view(cfg.batch_size, T, gsa.indexer_heads, gsa.d_idx)
                    k_I = gsa.W_Ik(x)
                    scale_idx = 1.0 / math.sqrt(gsa.d_idx)
                    k_I_exp = k_I.unsqueeze(1).expand(cfg.batch_size, gsa.indexer_heads, T, gsa.d_idx)
                    q_I_t = q_I.permute(0, 2, 1, 3)
                    scores = torch.matmul(q_I_t, k_I_exp.transpose(-1, -2)) * scale_idx
                    causal = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
                    scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
                    scores_avg = scores.mean(dim=1)
                    _, top_indices = scores_avg.topk(min(cfg.gsa_k_base, T), dim=-1)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                idx_times.append((t1 - t0) * 1000)

            idx_avg = sum(idx_times[3:]) / len(idx_times[3:])
            gsa_fwd = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "GSA-Sim"), None)
            pct = (idx_avg / gsa_fwd["fwd_ms"] * 100) if gsa_fwd else 0

            print(f"  {T:>8}  {idx_avg:>14.2f}  {pct:>13.1f}%")
            del gsa
        except Exception as e:
            print(f"  {T:>8}  FAILED: {e}")

        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  SUMMARY: Flash Attention Speedup over GSA")
    print("=" * 80)
    print(f"  {'Seq Len':>8}  {'Speedup':>10}  {'GSA Mem':>10}  {'Flash Mem':>10}  {'Recommendation':>20}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")

    for T in cfg.seq_lens:
        gsa_r = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "GSA-Sim"), None)
        fa_r = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "FlashAttn"), None)

        if gsa_r and fa_r:
            speedup = gsa_r["total_ms"] / fa_r["total_ms"]
            rec = "Use Flash Attn ✅" if speedup > 1.2 else ("Keep GSA ✅" if speedup < 0.8 else "~Similar ⚖️")
            print(f"  {T:>8}  {speedup:>9.2f}×  {gsa_r['peak_mem_gb']:>9.2f}G  {fa_r['peak_mem_gb']:>9.2f}G  {rec:>20}")

    # ── Save Results ─────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/a2_gsa_vs_flash.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "A2_gsa_vs_flash",
            "config": {
                "batch_size": cfg.batch_size,
                "gsa_num_heads": cfg.gsa_num_heads,
                "gsa_head_dim": cfg.gsa_head_dim,
                "gsa_k_base": cfg.gsa_k_base,
                "hidden_size": cfg.hidden_size,
                "dtype": cfg.dtype_str,
                "gpu": torch.cuda.get_device_name(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2: GSA vs Flash Attention Benchmark")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        dtype_str=args.dtype,
        seq_lens=tuple(int(x) for x in args.seq_lens.split(",")),
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        exit(1)

    run_benchmark(cfg)
