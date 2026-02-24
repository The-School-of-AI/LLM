"""
Benchmark A2: GSA (Gated Sparse Attention) vs Flash Attention at seq_len=512
=============================================================================

Tests whether standard Flash Attention is faster than GSA's real Triton
sparse attention pipeline at short sequences (OPUS scoring at 512).

This version imports the REAL GatedSparseAttention from the codebase,
using the actual Triton kernels (triton_sparse_attn, triton_indexer, etc.).

Usage:
    # On AWS (where Triton kernels work):
    python benchmark_a2_gsa_vs_flash.py --dtype bf16

    # On Colab T4 (Triton kernels may or may not work):
    python benchmark_a2_gsa_vs_flash.py --dtype fp16

    # With fallback PyTorch simulation if Triton unavailable:
    python benchmark_a2_gsa_vs_flash.py --dtype fp16 --fallback
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


# ── Resolve imports from the real codebase ────────────────────────────────────

def _setup_imports():
    """Add the latest codebase to sys.path so we can import real model classes."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

    # Try latest code first (28k), then fall back to OngoingRun3
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

    print("  WARNING: Could not find real codebase. Using PyTorch simulation.")
    return None

CODE_DIR = _setup_imports()


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    """1B model GSA config."""
    batch_size: int = 4
    hidden_size: int = 4096
    gsa_num_heads: int = 16
    gsa_head_dim: int = 256
    gsa_k_base: int = 128
    gsa_k_min: int = 32
    gsa_k_max: int = 256
    gsa_indexer_heads: int = 4
    seq_lens: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"
    use_fallback: bool = False

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── Import real model or build fallback ──────────────────────────────────────

def create_gsa_layer(cfg: BenchConfig, device, dtype, use_real: bool = True):
    """Try to create a real GatedSparseAttention layer, fall back to simulation."""
    if use_real and CODE_DIR is not None:
        try:
            from src.models.recurrence_model_1b import GatedSparseAttention
            layer = GatedSparseAttention(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.gsa_num_heads,
                max_seq_len=8192,  # Shorter for benchmark
                rope_base=10000,
                k_base=cfg.gsa_k_base,
                k_min=cfg.gsa_k_min,
                k_max=cfg.gsa_k_max,
                indexer_heads=cfg.gsa_indexer_heads,
                rope_original_max=8192,
                rope_scaling_factor=1.0,
                require_fused_kernel=True,
            ).to(device=device, dtype=dtype)
            print("  ✅ Using REAL GatedSparseAttention (Triton kernels)")
            return layer, "GSA-Real"
        except Exception as e:
            print(f"  ⚠️  Real GSA failed: {e}")
            print(f"  Falling back to PyTorch simulation...")

    # PyTorch simulation fallback
    import math
    layer = GSASimulated(cfg, device, dtype)
    print("  Using GSA-Simulated (PyTorch ops, no Triton)")
    return layer, "GSA-Sim"


class GSASimulated(nn.Module):
    """Fallback: PyTorch-only GSA simulation when Triton kernels unavailable."""
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        import math
        H, D, C = cfg.gsa_num_heads, cfg.gsa_head_dim, cfg.hidden_size
        self.num_heads, self.head_dim, self.hidden_size = H, D, C
        self.k_base = cfg.gsa_k_base
        self.indexer_heads, self.d_idx = cfg.gsa_indexer_heads, 128

        self.W_Iq = nn.Linear(C, self.indexer_heads * self.d_idx, bias=False, device=device, dtype=dtype)
        self.W_Ik = nn.Linear(C, self.d_idx, bias=False, device=device, dtype=dtype)
        self.W_q = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.W_k = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.W_v = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(C, C, bias=False, device=device, dtype=dtype)
        self.W_go = nn.Linear(C, C, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        import math
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Indexer
        q_I = self.W_Iq(x).view(B, T, self.indexer_heads, self.d_idx)
        k_I = self.W_Ik(x)
        scale_idx = 1.0 / math.sqrt(self.d_idx)
        q_I_t = q_I.permute(0, 2, 1, 3)
        k_I_exp = k_I.unsqueeze(1).expand(B, self.indexer_heads, T, self.d_idx)
        scores = torch.matmul(q_I_t, k_I_exp.transpose(-1, -2)) * scale_idx
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        scores_avg = scores.mean(dim=1)
        k_limit = min(self.k_base, T)
        _, top_indices = scores_avg.topk(k_limit, dim=-1)

        # Q/K/V
        q = self.W_q(x).view(B, T, H, D)
        k = self.W_k(x).view(B, T, H, D)
        v = self.W_v(x).view(B, T, H, D)
        scale_attn = 1.0 / math.sqrt(D)

        # Sparse gather
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1)
        k_sel = k[batch_idx, top_indices].permute(0, 1, 3, 2, 4)
        v_sel = v[batch_idx, top_indices].permute(0, 1, 3, 2, 4)

        attn = torch.matmul(q.unsqueeze(3), k_sel.transpose(-1, -2)).squeeze(3) * scale_attn
        attn = F.softmax(attn, dim=-1)
        o = torch.matmul(attn.unsqueeze(3), v_sel).squeeze(3)

        o = o.contiguous().view(B, T, C)
        g_o = torch.sigmoid(self.W_go(x))
        return self.o_proj(o * g_o)


# ── Flash Attention ──────────────────────────────────────────────────────────

class FlashAttnBench(nn.Module):
    """Standard MHA using Flash Attention 2, same dimensions as GSA."""
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H, D, C = cfg.gsa_num_heads, cfg.gsa_head_dim, cfg.hidden_size
        self.num_heads, self.head_dim = H, D
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
        return self.o_proj(o.transpose(1, 2).reshape(B, T, H * D))


# ── Benchmarking ─────────────────────────────────────────────────────────────

def benchmark_forward_backward(model, x, warmup, iters, label):
    model.train()
    for _ in range(warmup):
        x_in = x.clone().requires_grad_(True)
        out = model(x_in)
        out.sum().backward()
        del out
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times, bwd_times = [], []
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
    fwd_times, bwd_times = fwd_times[drop:], bwd_times[drop:]
    fwd_avg = sum(fwd_times) / len(fwd_times)
    bwd_avg = sum(bwd_times) / len(bwd_times)

    return {
        "label": label, "fwd_ms": fwd_avg, "bwd_ms": bwd_avg,
        "total_ms": fwd_avg + bwd_avg, "peak_mem_gb": peak_mem,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_benchmark(cfg: BenchConfig):
    device = torch.device("cuda")
    dtype = cfg.dtype
    print(f"\n{'='*80}")
    print(f"  BENCHMARK A2: GSA vs Flash Attention")
    print(f"  Config: 1B | B={cfg.batch_size} | H={cfg.gsa_num_heads} | D={cfg.gsa_head_dim} | k_base={cfg.gsa_k_base}")
    print(f"  GPU: {torch.cuda.get_device_name()} | dtype={cfg.dtype_str}")
    print(f"  Warmup: {cfg.warmup_iters} | Bench: {cfg.bench_iters} iters")
    print(f"{'='*80}\n")

    results = []

    print("━" * 80)
    print("  Full Layer: GSA vs Flash Attention (dense)")
    print("━" * 80)
    print(f"  {'Seq Len':>8}  {'Kernel':>14}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        # GSA (try real, fall back to simulated)
        gsa_result = None
        try:
            gsa, gsa_label = create_gsa_layer(cfg, device, dtype, use_real=not cfg.use_fallback)
            gsa_result = benchmark_forward_backward(gsa, x, cfg.warmup_iters, cfg.bench_iters, f"gsa_T{T}")
            gsa_result["seq_len"] = T
            gsa_result["kernel"] = gsa_label
            results.append(gsa_result)
            del gsa
        except Exception as e:
            print(f"  {T:>8}  {'GSA':>14}  FAILED: {e}")

        # Flash Attention
        fa_result = None
        try:
            flash = FlashAttnBench(cfg, device, dtype)
            fa_result = benchmark_forward_backward(flash, x, cfg.warmup_iters, cfg.bench_iters, f"flash_T{T}")
            fa_result["seq_len"] = T
            fa_result["kernel"] = "FlashAttn"
            results.append(fa_result)
            del flash
        except Exception as e:
            print(f"  {T:>8}  {'FlashAttn':>14}  FAILED: {e}")

        if gsa_result and fa_result:
            speedup = gsa_result["total_ms"] / fa_result["total_ms"]
            print(f"  {T:>8}  {gsa_result['kernel']:>14}  {gsa_result['fwd_ms']:>10.2f}  {gsa_result['bwd_ms']:>10.2f}  {gsa_result['total_ms']:>12.2f}  {gsa_result['peak_mem_gb']:>10.2f}")
            print(f"  {T:>8}  {'FlashAttn':>14}  {fa_result['fwd_ms']:>10.2f}  {fa_result['bwd_ms']:>10.2f}  {fa_result['total_ms']:>12.2f}  {fa_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
            print()

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY: Flash Attention Speedup over GSA")
    print("=" * 80)
    print(f"  {'Seq Len':>8}  {'Speedup':>10}  {'GSA Mem':>10}  {'Flash Mem':>10}  {'Recommendation':>20}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")

    for T in cfg.seq_lens:
        gsa_r = next((r for r in results if r.get("seq_len") == T and "GSA" in r.get("kernel", "")), None)
        fa_r = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "FlashAttn"), None)
        if gsa_r and fa_r:
            speedup = gsa_r["total_ms"] / fa_r["total_ms"]
            rec = "Use Flash Attn ✅" if speedup > 1.2 else ("Keep GSA ✅" if speedup < 0.8 else "~Similar ⚖️")
            print(f"  {T:>8}  {speedup:>9.2f}×  {gsa_r['peak_mem_gb']:>9.2f}G  {fa_r['peak_mem_gb']:>9.2f}G  {rec:>20}")

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/a2_gsa_vs_flash.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "A2_gsa_vs_flash",
            "config": {
                "batch_size": cfg.batch_size, "gsa_num_heads": cfg.gsa_num_heads,
                "gsa_head_dim": cfg.gsa_head_dim, "gsa_k_base": cfg.gsa_k_base,
                "dtype": cfg.dtype_str, "gpu": torch.cuda.get_device_name(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2: GSA vs Flash Attention")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096")
    parser.add_argument("--fallback", action="store_true",
                        help="Force PyTorch simulation even if Triton available")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size, warmup_iters=args.warmup,
        bench_iters=args.iters, dtype_str=args.dtype,
        seq_lens=tuple(int(x) for x in args.seq_lens.split(",")),
        use_fallback=args.fallback,
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        exit(1)

    run_benchmark(cfg)
