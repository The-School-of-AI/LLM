"""
Benchmark A1: DeltaNet (fla) vs Flash Attention at seq_len=512
==============================================================

Tests whether standard Flash Attention is faster than DeltaNet's
chunk_gated_delta_rule kernel at short sequences (OPUS scoring at 512).

Sweeps seq_len: [256, 512, 1024, 2048, 4096] to find the crossover point.

Usage:
    # On Colab or any machine with CUDA:
    pip install fla triton
    python benchmark_a1_deltanet_vs_flash.py

    # Specify device:
    python benchmark_a1_deltanet_vs_flash.py --dtype fp16     # Colab T4
    python benchmark_a1_deltanet_vs_flash.py --dtype bf16     # A100/H100

Output:
    - Formatted table to stdout
    - JSON results to results/a1_deltanet_vs_flash.json
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BenchConfig:
    """1B model attention config."""
    batch_size: int = 4
    num_heads: int = 32        # DeltaNet heads (1B config)
    head_dim: int = 128        # DeltaNet head dim (1B config)
    hidden_size: int = 4096    # 1B hidden
    seq_lens: Tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    warmup_iters: int = 10
    bench_iters: int = 50
    dtype_str: str = "bf16"

    @property
    def dtype(self):
        return torch.bfloat16 if self.dtype_str == "bf16" else torch.float16


# ── DeltaNet Benchmark (uses fla kernel) ─────────────────────────────────────

class DeltaNetBench(nn.Module):
    """
    Minimal DeltaNet layer matching 1B model's GatedDeltaNet.
    Includes: fused_qkvg_proj → conv1d → L2 norm → RoPE → fla kernel → output norm → o_proj
    """
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H, D, C = cfg.num_heads, cfg.head_dim, cfg.hidden_size
        key_dim = H * D
        value_dim = H * D

        self.num_heads = H
        self.head_dim = D
        self.key_dim = key_dim
        self.value_dim = value_dim

        # Projections (same as 1B model)
        out_dim = 2 * key_dim + 2 * value_dim
        self.fused_qkvg_proj = nn.Linear(C, out_dim, bias=False, device=device, dtype=dtype)
        self.fused_bgk_proj = nn.Linear(C, 2 * H, bias=True, device=device, dtype=dtype)
        self.o_proj = nn.Linear(value_dim, C, bias=False, device=device, dtype=dtype)

        # Learnable params
        self.D = nn.Parameter(torch.ones(H, device=device, dtype=dtype))
        self.A_log = nn.Parameter(torch.zeros(H, device=device, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.zeros(H, device=device, dtype=torch.float32))

    def forward(self, x):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Projections
        qkvg = self.fused_qkvg_proj(x)
        q, k, v, g = torch.split(qkvg, [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1)

        bgk = self.fused_bgk_proj(x)
        b_logits, gk_logits = bgk.chunk(2, dim=-1)

        # Reshape to heads
        q = q.view(B, T, H, D)
        k = k.view(B, T, H, D)
        v = v.view(B, T, H, D)
        g = g.view(B, T, H, D)

        # L2 normalize
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Compute alpha and beta
        beta = torch.sigmoid(b_logits).unsqueeze(-1)  # (B, T, H, 1)
        dt = F.softplus(gk_logits + self.dt_bias.view(1, 1, -1))
        A = torch.exp(self.A_log)
        alpha = torch.exp(-A.view(1, 1, -1) * dt).unsqueeze(-1)  # (B, T, H, 1)

        # fla kernel
        q_f32 = q.float()
        k_f32 = k.float()
        v_f32 = v.float()
        g_fla = torch.log(alpha[:, :, :, 0].float().clamp(min=1e-6))
        beta_fla = beta[:, :, :, 0].float()

        o_fla, _ = chunk_gated_delta_rule(
            q_f32, k_f32, v_f32, g_fla, beta_fla,
            scale=1.0, output_final_state=False,
        )

        # D residual
        D_weight = self.D.view(1, 1, H, 1)
        qk_dot = (q * k).sum(dim=-1, keepdim=True)
        d_residual = D_weight * qk_dot * v
        o = o_fla.to(q.dtype) + d_residual

        # Output gate
        o = o * torch.sigmoid(g)

        # Output projection
        o = o.reshape(B, T, H * D)
        return self.o_proj(o)


# ── Flash Attention Benchmark ────────────────────────────────────────────────

class FlashAttnBench(nn.Module):
    """
    Standard Multi-Head Attention layer using PyTorch's SDPA (Flash Attention 2 backend).
    Same parameter count and API as DeltaNet for fair comparison.
    """
    def __init__(self, cfg: BenchConfig, device, dtype):
        super().__init__()
        H, D, C = cfg.num_heads, cfg.head_dim, cfg.hidden_size

        self.num_heads = H
        self.head_dim = D

        # Q, K, V projections (same total param count as DeltaNet's fused_qkvg)
        self.q_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.k_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.v_proj = nn.Linear(C, H * D, bias=False, device=device, dtype=dtype)
        self.o_proj = nn.Linear(H * D, C, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # PyTorch SDPA — automatically uses Flash Attention 2 on CUDA
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        o = o.transpose(1, 2).reshape(B, T, H * D)  # (B, T, C)
        return self.o_proj(o)


# ── Benchmarking Utilities ───────────────────────────────────────────────────

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

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    bwd_times = []
    total_times = []

    for _ in range(iters):
        x_in = x.clone().requires_grad_(True)
        torch.cuda.synchronize()

        # Forward
        t0 = time.perf_counter()
        out = model(x_in)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Backward
        loss = out.sum()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t3 - t2) * 1000)
        total_times.append((t3 - t0) * 1000)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Drop first few measurements (may still have warmup effects)
    drop = max(1, iters // 10)
    fwd_times = fwd_times[drop:]
    bwd_times = bwd_times[drop:]
    total_times = total_times[drop:]

    result = {
        "label": label,
        "fwd_ms": sum(fwd_times) / len(fwd_times),
        "bwd_ms": sum(bwd_times) / len(bwd_times),
        "total_ms": sum(total_times) / len(total_times),
        "peak_mem_gb": peak_mem,
        "fwd_std": (sum((t - sum(fwd_times)/len(fwd_times))**2 for t in fwd_times) / len(fwd_times)) ** 0.5,
        "bwd_std": (sum((t - sum(bwd_times)/len(bwd_times))**2 for t in bwd_times) / len(bwd_times)) ** 0.5,
    }
    return result


def benchmark_kernel_only(
    make_inputs_fn,
    kernel_fn,
    warmup: int,
    iters: int,
    label: str,
    backward: bool = True,
) -> Dict[str, float]:
    """Benchmark a raw kernel function (no model wrapper).
    
    make_inputs_fn: callable that returns a fresh tuple of input tensors each call.
    This ensures no retain_graph issues with kernels that free saved tensors.
    """
    # Warmup
    for _ in range(warmup):
        inputs = make_inputs_fn()
        out = kernel_fn(*inputs)
        if backward:
            out.sum().backward()
        del inputs, out
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_times = []
    bwd_times = []

    for _ in range(iters):
        inputs = make_inputs_fn()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        out = kernel_fn(*inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        if backward:
            loss = out.sum()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            bwd_times.append((t3 - t2) * 1000)
        else:
            t3 = t1

        fwd_times.append((t1 - t0) * 1000)
        del inputs, out

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    drop = max(1, iters // 10)

    fwd_times = fwd_times[drop:]
    bwd_ms = sum(bwd_times[drop:]) / len(bwd_times[drop:]) if bwd_times else 0.0

    return {
        "label": label,
        "fwd_ms": sum(fwd_times) / len(fwd_times),
        "bwd_ms": bwd_ms,
        "total_ms": sum(fwd_times) / len(fwd_times) + bwd_ms,
        "peak_mem_gb": peak_mem,
    }


# ── Main Benchmark ──────────────────────────────────────────────────────────

def run_benchmark(cfg: BenchConfig):
    device = torch.device("cuda")
    dtype = cfg.dtype
    print(f"\n{'='*80}")
    print(f"  BENCHMARK A1: DeltaNet (fla) vs Flash Attention")
    print(f"  Config: 1B model | B={cfg.batch_size} | H={cfg.num_heads} | D={cfg.head_dim} | dtype={cfg.dtype_str}")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Warmup: {cfg.warmup_iters} | Bench: {cfg.bench_iters} iters")
    print(f"{'='*80}\n")

    results = []

    # ── Part 1: Full Layer Comparison (projections + kernel + output) ────────
    print("━" * 70)
    print("  PART 1: Full Attention Layer (projections + kernel + output proj)")
    print("━" * 70)
    print(f"  {'Seq Len':>8}  {'Kernel':>12}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}  {'Speedup':>8}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")

    for T in cfg.seq_lens:
        x = torch.randn(cfg.batch_size, T, cfg.hidden_size, device=device, dtype=dtype)

        # DeltaNet
        try:
            deltanet = DeltaNetBench(cfg, device, dtype)
            dn_result = benchmark_forward_backward(deltanet, x, cfg.warmup_iters, cfg.bench_iters, f"deltanet_T{T}")
            dn_result["seq_len"] = T
            dn_result["kernel"] = "DeltaNet"
            results.append(dn_result)
            del deltanet
        except Exception as e:
            print(f"  {T:>8}  {'DeltaNet':>12}  FAILED: {e}")
            dn_result = None

        # Flash Attention
        try:
            flash = FlashAttnBench(cfg, device, dtype)
            fa_result = benchmark_forward_backward(flash, x, cfg.warmup_iters, cfg.bench_iters, f"flash_T{T}")
            fa_result["seq_len"] = T
            fa_result["kernel"] = "FlashAttn"
            results.append(fa_result)
            del flash
        except Exception as e:
            print(f"  {T:>8}  {'FlashAttn':>12}  FAILED: {e}")
            fa_result = None

        # Print comparison
        if dn_result and fa_result:
            speedup = dn_result["total_ms"] / fa_result["total_ms"]
            print(f"  {T:>8}  {'DeltaNet':>12}  {dn_result['fwd_ms']:>10.2f}  {dn_result['bwd_ms']:>10.2f}  {dn_result['total_ms']:>12.2f}  {dn_result['peak_mem_gb']:>10.2f}")
            winner = "◀ FASTER" if speedup < 1 else ""
            print(f"  {T:>8}  {'FlashAttn':>12}  {fa_result['fwd_ms']:>10.2f}  {fa_result['bwd_ms']:>10.2f}  {fa_result['total_ms']:>12.2f}  {fa_result['peak_mem_gb']:>10.2f}  {speedup:>7.2f}×")
            print()

        torch.cuda.empty_cache()

    # ── Part 2: Raw Kernel Only (no projections) ────────────────────────────
    print("\n" + "━" * 70)
    print("  PART 2: Raw Kernel Only (no projections, no output)")
    print("━" * 70)
    print(f"  {'Seq Len':>8}  {'Kernel':>12}  {'Fwd (ms)':>10}  {'Bwd (ms)':>10}  {'Total (ms)':>12}  {'Mem (GB)':>10}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")

    for T in cfg.seq_lens:
        H, D = cfg.num_heads, cfg.head_dim
        B = cfg.batch_size

        # Raw fla kernel
        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule

            def make_fla_inputs():
                return (
                    torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                    torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                    torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True),
                    torch.randn(B, T, H, device=device, dtype=torch.float32, requires_grad=True),
                    torch.rand(B, T, H, device=device, dtype=torch.float32, requires_grad=True),  # beta in [0,1]
                )

            def fla_fn(q, k, v, g, beta):
                o, _ = chunk_gated_delta_rule(q, k, v, g, beta, scale=1.0, output_final_state=False)
                return o

            fla_result = benchmark_kernel_only(
                make_fla_inputs, fla_fn,
                cfg.warmup_iters, cfg.bench_iters, f"fla_raw_T{T}"
            )
            fla_result["seq_len"] = T
            fla_result["kernel"] = "fla_raw"
            results.append(fla_result)
            print(f"  {T:>8}  {'fla_raw':>12}  {fla_result['fwd_ms']:>10.2f}  {fla_result['bwd_ms']:>10.2f}  {fla_result['total_ms']:>12.2f}  {fla_result['peak_mem_gb']:>10.2f}")
        except Exception as e:
            print(f"  {T:>8}  {'fla_raw':>12}  FAILED: {e}")

        # Raw Flash Attention (SDPA)
        try:
            def make_sdpa_inputs():
                return (
                    torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True),
                    torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True),
                    torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True),
                )

            def sdpa_fn(q, k, v):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

            sdpa_result = benchmark_kernel_only(
                make_sdpa_inputs, sdpa_fn,
                cfg.warmup_iters, cfg.bench_iters, f"sdpa_raw_T{T}"
            )
            sdpa_result["seq_len"] = T
            sdpa_result["kernel"] = "sdpa_raw"
            results.append(sdpa_result)
            print(f"  {T:>8}  {'sdpa_raw':>12}  {sdpa_result['fwd_ms']:>10.2f}  {sdpa_result['bwd_ms']:>10.2f}  {sdpa_result['total_ms']:>12.2f}  {sdpa_result['peak_mem_gb']:>10.2f}")
        except Exception as e:
            print(f"  {T:>8}  {'sdpa_raw':>12}  FAILED: {e}")

        print()
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY: Flash Attention Speedup over DeltaNet")
    print("=" * 80)
    print(f"  {'Seq Len':>8}  {'Layer Speedup':>15}  {'Kernel Speedup':>15}  {'Recommendation':>20}")
    print(f"  {'─'*8}  {'─'*15}  {'─'*15}  {'─'*20}")

    for T in cfg.seq_lens:
        dn_layer = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "DeltaNet"), None)
        fa_layer = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "FlashAttn"), None)
        fla_raw = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "fla_raw"), None)
        sdpa_raw = next((r for r in results if r.get("seq_len") == T and r.get("kernel") == "sdpa_raw"), None)

        layer_speedup = dn_layer["total_ms"] / fa_layer["total_ms"] if (dn_layer and fa_layer) else 0
        kernel_speedup = fla_raw["total_ms"] / sdpa_raw["total_ms"] if (fla_raw and sdpa_raw) else 0

        if layer_speedup > 1.2:
            rec = "Use Flash Attn ✅"
        elif layer_speedup < 0.8:
            rec = "Keep DeltaNet ✅"
        else:
            rec = "~Similar ⚖️"

        ls = f"{layer_speedup:.2f}×" if layer_speedup else "N/A"
        ks = f"{kernel_speedup:.2f}×" if kernel_speedup else "N/A"
        print(f"  {T:>8}  {ls:>15}  {ks:>15}  {rec:>20}")

    # ── Save Results ─────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/a1_deltanet_vs_flash.json"
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": "A1_deltanet_vs_flash",
            "config": {
                "batch_size": cfg.batch_size,
                "num_heads": cfg.num_heads,
                "head_dim": cfg.head_dim,
                "hidden_size": cfg.hidden_size,
                "dtype": cfg.dtype_str,
                "gpu": torch.cuda.get_device_name(),
                "warmup": cfg.warmup_iters,
                "iters": cfg.bench_iters,
            },
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A1: DeltaNet vs Flash Attention Benchmark")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"],
                        help="Data type (bf16 for A100/H100, fp16 for T4/Colab)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seq-lens", type=str, default="256,512,1024,2048,4096",
                        help="Comma-separated sequence lengths to test")
    args = parser.parse_args()

    cfg = BenchConfig(
        batch_size=args.batch_size,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        dtype_str=args.dtype,
        seq_lens=tuple(int(x) for x in args.seq_lens.split(",")),
    )

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.")
        exit(1)

    run_benchmark(cfg)
