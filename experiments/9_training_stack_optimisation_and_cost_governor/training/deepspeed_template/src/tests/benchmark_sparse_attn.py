"""
Benchmark: Triton Sparse Attention — Forward & Backward Timing
==============================================================

Compares:
  - PyTorch reference (autograd backward)
  - Triton optimized (custom forward + backward with tl.dot)

Run on a CUDA GPU:
    python tests/benchmark_sparse_attn.py
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kernels.triton_sparse_attn import (
    triton_sparse_attention,
    pytorch_sparse_attention,
    HAS_TRITON,
)


def make_bench_data(B, T, H, D, k_sel, device='cuda'):
    """Generate benchmark data with random sparse indices."""
    torch.manual_seed(42)

    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)

    # Random sparse indices (causal)
    indices = torch.zeros(B, H, T, k_sel, dtype=torch.int64, device=device)
    for b in range(B):
        for t in range(T):
            valid_range = t + 1
            if valid_range >= k_sel:
                idx = torch.randperm(valid_range, device=device)[:k_sel].sort().values
            else:
                idx = torch.arange(valid_range, device=device)
                idx = torch.cat([idx, torch.zeros(k_sel - valid_range, dtype=torch.long, device=device)])
            indices[b, :, t, :] = idx

    mask = torch.ones(B, H, T, k_sel, dtype=torch.float32, device=device)
    for t in range(T):
        valid_range = t + 1
        if valid_range < k_sel:
            mask[:, :, t, valid_range:] = 0.0

    scale = 1.0 / (D ** 0.5)
    return q, k, v, indices, mask, scale


def bench_forward_backward(fn_name, fn, q, k, v, indices, mask, scale, n_warmup=5, n_iters=20):
    """Time forward and backward passes separately."""
    grad_out = torch.randn_like(q)

    # Warmup
    for _ in range(n_warmup):
        q_ = q.detach().clone().requires_grad_(True)
        k_ = k.detach().clone().requires_grad_(True)
        v_ = v.detach().clone().requires_grad_(True)
        out = fn(q_, k_, v_, indices, mask, scale)
        torch.cuda.synchronize()
        out.backward(grad_out)
        torch.cuda.synchronize()

    # Timed: forward only
    fwd_times = []
    for _ in range(n_iters):
        q_ = q.detach().clone().requires_grad_(True)
        k_ = k.detach().clone().requires_grad_(True)
        v_ = v.detach().clone().requires_grad_(True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(q_, k_, v_, indices, mask, scale)
        torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000)

    # Timed: backward only (forward done, then time backward)
    bwd_times = []
    for _ in range(n_iters):
        q_ = q.detach().clone().requires_grad_(True)
        k_ = k.detach().clone().requires_grad_(True)
        v_ = v.detach().clone().requires_grad_(True)
        out = fn(q_, k_, v_, indices, mask, scale)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.backward(grad_out)
        torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1000)

    fwd_avg = sum(fwd_times) / len(fwd_times)
    bwd_avg = sum(bwd_times) / len(bwd_times)
    print(f"  {fn_name:25s}  fwd: {fwd_avg:8.2f} ms  bwd: {bwd_avg:8.2f} ms  total: {fwd_avg+bwd_avg:8.2f} ms")
    return fwd_avg, bwd_avg


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — cannot benchmark.")
        sys.exit(0)

    if not HAS_TRITON:
        print("⚠️  Triton not installed — cannot benchmark Triton kernels.")
        sys.exit(0)

    print("=" * 75)
    print("Triton Sparse Attention — Forward & Backward Benchmark")
    print("=" * 75)

    configs = [
        # (B, T, H, D, k_sel, label)
        (2, 512,  16, 256, 128,  "Production (small T)"),
        (2, 1024, 16, 256, 256,  "Production (mid T)"),
        (2, 2048, 16, 256, 512,  "Production (large T)"),
        (1, 4096, 16, 256, 512,  "Production (very large T)"),
        (2, 512,  4,  32,  64,   "Test dims (small D)"),
    ]

    for B, T, H, D, k_sel, label in configs:
        print(f"\n{'─' * 75}")
        print(f"Config: {label}")
        print(f"  B={B}, T={T}, H={H}, D={D}, k_sel={k_sel}")
        print(f"{'─' * 75}")

        q, k, v, indices, mask, scale = make_bench_data(B, T, H, D, k_sel)

        fwd_pt, bwd_pt = bench_forward_backward(
            "PyTorch reference", pytorch_sparse_attention,
            q, k, v, indices, mask, scale
        )

        fwd_tr, bwd_tr = bench_forward_backward(
            "Triton optimized", triton_sparse_attention,
            q, k, v, indices, mask, scale
        )

        fwd_speedup = fwd_pt / max(fwd_tr, 1e-6)
        bwd_speedup = bwd_pt / max(bwd_tr, 1e-6)
        total_speedup = (fwd_pt + bwd_pt) / max(fwd_tr + bwd_tr, 1e-6)
        print(f"  Speedup:  fwd: {fwd_speedup:.2f}×  bwd: {bwd_speedup:.2f}×  total: {total_speedup:.2f}×")

    print(f"\n{'=' * 75}")
    print("Benchmark complete.")
    print("=" * 75)
