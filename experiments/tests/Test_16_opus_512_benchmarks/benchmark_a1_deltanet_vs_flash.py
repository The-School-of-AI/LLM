"""
Benchmark A1: DeltaNet (fla) vs Flash Attention at seq_len=512
==============================================================

Tests whether standard Flash Attention is faster than DeltaNet's
chunk_gated_delta_rule kernel at short sequences (OPUS scoring at 512).

This version imports the REAL GatedDeltaNet from the codebase,
using the actual fla Triton kernel and all layer components (conv1d,
RoPE, FusedRMSNormSwishGate, etc.).

Sweeps seq_len: [256, 512, 1024, 2048, 4096] to find the crossover point.

Usage:
    # On AWS (full real kernels):
    python benchmark_a1_deltanet_vs_flash.py --dtype bf16

    # On Colab (needs: pip install git+.../flash-linear-attention):
    python benchmark_a1_deltanet_vs_flash.py --dtype fp16
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

    print("  WARNING: Could not find real codebase.")
    return None

CODE_DIR = _setup_imports()


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    """1B model DeltaNet config."""
    batch_size: int = 4
    num_heads: int = 32        # delta_v_heads
    head_dim: int = 128        # delta_head_dim
    hidden_size: int = 4096
    seq_lens: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── Import real model or build fallback ──────────────────────────────────────

def create_deltanet_layer(cfg: BenchConfig, device, dtype):
    """Try to create a real GatedDeltaNet layer from the codebase."""
    if CODE_DIR is not None:
        try:
            from src.models.recurrence_model_1b import GatedDeltaNet
            layer = GatedDeltaNet(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                head_dim=cfg.head_dim,
                max_seq_len=8192,  # Shorter for benchmark
                rope_base=10000,
                rope_original_max=8192,
                rope_scaling_factor=1.0,
                conv_size=4,
                use_output_norm=True,
                require_fused_kernel=True,
            ).to(device=device, dtype=dtype)
            print("  ✅ Using REAL GatedDeltaNet (fla kernel + conv1d + RoPE + FusedRMSNormSwishGate)")
            return layer, "DeltaNet-Real"
        except Exception as e:
            print(f"  ⚠️  Real DeltaNet import failed: {e}")

    # Minimal fallback
    print("  ⚠️  Using minimal DeltaNet fallback (fla kernel only, no conv1d/RoPE/norm)")
    return DeltaNetFallback(cfg, device, dtype), "DeltaNet-Min"


class DeltaNetFallback(nn.Module):
    """Minimal DeltaNet when codebase import fails. Uses fla kernel directly."""
    def __init__(self, cfg, device, dtype):
        super().__init__()
        H, D, C = cfg.num_heads, cfg.head_dim, cfg.hidden_size
        self.num_heads, self.head_dim = H, D
        self.key_dim = H * D
        self.value_dim = H * D

        out_dim = 2 * self.key_dim + 2 * self.value_dim
        self.fused_qkvg_proj = nn.Linear(C, out_dim, bias=False, device=device, dtype=dtype)
        self.fused_bgk_proj = nn.Linear(C, 2 * H, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(self.value_dim, C, bias=False, device=device, dtype=dtype)
        self.D = nn.Parameter(torch.ones(H, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.zeros(H, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(H, device=device, dtype=torch.float32))

    def forward(self, x):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkvg = self.fused_qkvg_proj(x)
        q, k, v, g = torch.split(qkvg, [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1)
        bgk = self.fused_bgk_proj(x)
        b_logits, gk_logits = bgk.chunk(2, dim=-1)

        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        g = g.view(B, T, H, D)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        beta = torch.sigmoid(b_logits).unsqueeze(-1)
        dt = F.softplus(gk_logits + self.dt_bias.view(1, 1, -1))
        A = torch.exp(self.A_log)
        alpha = torch.exp(-A.view(1, 1, -1) * dt).unsqueeze(-1)

        q_f32, k_f32, v_f32 = q.float(), k.float(), v.float()
        g_fla = torch.log(alpha[:, :, :, 0].float().clamp(min=1e-6))
        beta_fla = beta[:, :, :, 0].float()

        o_fla, _ = chunk_gated_delta_rule(q_f32, k_f32, v_f32, g_fla, beta_fla,
                                           scale=1.0, output_final_state=False)

        D_weight = self.D.view(1, 1, H, 1)
        qk_dot = (q * k).sum(dim=-1, keepdim=True)
        o = o_fla.to(q.dtype) + D_weight * qk_dot * v
        o = o * torch.sigmoid(g)
        return self.o_proj(o.reshape(B, T, H * D))


# ── Flash Attention ──────────────────────────────────────────────────────────

class FlashAttnBench(nn.Module):
    """Standard MHA using Flash Attention 2, same dimensions as DeltaNet."""
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H, D, C = cfg.num_heads, cfg.head_dim, cfg.hidden_size
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
    print(f"  BENCHMARK A1: DeltaNet (fla) vs Flash Attention")
    print(f"  Config: 1B | B={cfg.batch_size} | H={cfg.num_heads} | D={cfg.head_dim} | dtype={cfg.dtype_str}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Warmup: {cfg.warmup_iters} | Bench: {cfg.bench_iters} iters")
    print(f"{'='*80}\n")

    results = []

    print("━" * 80)
    print("  Full Layer: DeltaNet vs Flash Attention")
    print("━" * 80)
    print(f"  {'Seq Len':>8}  {'Kernel':>16}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*16}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        # DeltaNet (real or fallback)
        dn_result = None
        try:
            deltanet, dn_label = create_deltanet_layer(cfg, device, dtype)
            dn_result = benchmark_forward_backward(deltanet, x, cfg.warmup_iters, cfg.bench_iters, f"dn_T{T}")
            dn_result["seq_len"] = T
            dn_result["kernel"] = dn_label
            results.append(dn_result)
            del deltanet
        except Exception as e:
            print(f"  {T:>8}  {'DeltaNet':>16}  FAILED: {e}")

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
            print(f"  {T:>8}  {'FlashAttn':>16}  FAILED: {e}")

        if dn_result and fa_result:
            speedup = dn_result["total_ms"] / fa_result["total_ms"]
            print(f"  {T:>8}  {dn_result['kernel']:>16}  {dn_result['fwd_ms']:>10.2f}  {dn_result['bwd_ms']:>10.2f}  {dn_result['total_ms']:>12.2f}  {dn_result['peak_mem_gb']:>10.2f}")
            print(f"  {T:>8}  {'FlashAttn':>16}  {fa_result['fwd_ms']:>10.2f}  {fa_result['bwd_ms']:>10.2f}  {fa_result['total_ms']:>12.2f}  {fa_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
            print()

        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("  SUMMARY: Flash Attention Speedup over DeltaNet")
    print("=" * 80)
    print(f"  {'Seq Len':>8}  {'Speedup':>10}  {'DN Mem':>10}  {'Flash Mem':>10}  {'Recommendation':>20}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")

    for T in cfg.seq_lens:
        dn_r = next((r for r in results if r.get("seq_len") == T and "DeltaNet" in r.get("kernel", "")), None)
        fa_r = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "FlashAttn"), None)
        if dn_r and fa_r:
            speedup = dn_r["total_ms"] / fa_r["total_ms"]
            rec = "Use Flash Attn ✅" if speedup > 1.2 else ("Keep DeltaNet ✅" if speedup < 0.8 else "~Similar ⚖️")
            print(f"  {T:>8}  {speedup:>9.2f}×  {dn_r['peak_mem_gb']:>9.2f}G  {fa_r['peak_mem_gb']:>9.2f}G  {rec:>20}")

    # Save
    os.makedirs("results", exist_ok=True)
    out_path = "results/a1_deltanet_vs_flash.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "A1_deltanet_vs_flash",
            "config": {
                "batch_size": cfg.batch_size, "num_heads": cfg.num_heads,
                "head_dim": cfg.head_dim, "hidden_size": cfg.hidden_size,
                "dtype": cfg.dtype_str, "gpu": torch.cuda.get_device_name(),
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A1: DeltaNet vs Flash Attention")
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
