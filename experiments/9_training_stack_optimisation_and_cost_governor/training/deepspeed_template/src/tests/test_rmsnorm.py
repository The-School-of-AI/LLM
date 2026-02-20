"""
RMSNorm Correctness & Benchmark Tests
======================================

3-way comparison:
  1. PyTorch RMSNorm     (pure PyTorch — baseline reference)
  2. Old Triton fwd-only (current kernel — backward via autograd)
  3. New Liger Triton     (fused forward + backward in Triton)

Tests:
  - Mathematical correctness: forward outputs + backward gradients
  - Speed benchmark: forward time, backward time, total time, peak memory
  - dtype coverage: float32 and bfloat16

Run on a CUDA GPU:
    python -m tests.test_rmsnorm
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.triton_rmsnorm import (
    HAS_TRITON,
    pytorch_rmsnorm,
    triton_rmsnorm,
    triton_rmsnorm_fwd_only,
)

# ═══════════════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════════════


def make_test_data(
    B=2, T=128, hidden=4096, dtype=torch.float32, device="cuda", seed=42
):
    """Generate random input and weight for RMSNorm testing."""
    torch.manual_seed(seed)
    x = torch.randn(B, T, hidden, device=device, dtype=dtype, requires_grad=True)
    w = torch.ones(hidden, device=device, dtype=dtype, requires_grad=True)
    # Add some variation to the weight so dW is not trivially uniform
    with torch.no_grad():
        w.add_(torch.randn_like(w) * 0.1)
    return x, w


def pytorch_rmsnorm_ref(x, weight, eps=1e-6):
    """
    PyTorch reference using Llama-style casting (matches Triton kernel).

    Llama-style: normalize in fp32, cast to input dtype, THEN multiply weight.
    This matches our Triton kernel's casting order exactly.
    """
    in_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    # Llama-style: cast normalized result back to input dtype BEFORE weight multiply
    return x_normed.to(in_dtype) * weight


# ═══════════════════════════════════════════════════════════════════════
# Correctness Tests
# ═══════════════════════════════════════════════════════════════════════


def test_forward_correctness():
    """Test that new Triton RMSNorm forward matches PyTorch reference."""
    print("\n1. Forward Correctness Tests")
    print("   " + "-" * 56)

    configs = [
        (2, 128, 256, torch.float32),
        (2, 128, 1024, torch.float32),
        (2, 128, 4096, torch.float32),
        (1, 512, 4096, torch.float32),
        (2, 128, 4096, torch.bfloat16),
        (1, 4096, 4096, torch.bfloat16),
    ]

    all_pass = True
    for B, T, hidden, dtype in configs:
        x, w = make_test_data(B, T, hidden, dtype=dtype)

        with torch.no_grad():
            out_ref = pytorch_rmsnorm_ref(x, w)
            out_triton = triton_rmsnorm(x, w, eps=1e-6)

        max_diff = (out_ref - out_triton).abs().max().item()
        mean_diff = (out_ref - out_triton).abs().mean().item()

        # bf16 can have single-element rounding outliers up to ~1.6e-2
        # (mean_diff is ~1e-8, so it's just isolated rounding, not systematic)
        threshold = 1e-4 if dtype == torch.float32 else 2e-2
        status = "✅" if max_diff < threshold else "❌"
        if max_diff >= threshold:
            all_pass = False

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(
            f"   {status} B={B}, T={T}, H={hidden}, {dtype_str}:  "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}"
        )

    return all_pass


def test_backward_correctness():
    """Test that new Triton RMSNorm backward gradients match PyTorch reference."""
    print("\n2. Backward Correctness Tests (dX and dW)")
    print("   " + "-" * 56)

    configs = [
        (2, 128, 256, torch.float32),
        (2, 128, 1024, torch.float32),
        (2, 128, 4096, torch.float32),
        (1, 512, 4096, torch.float32),
        (2, 128, 4096, torch.bfloat16),
        (1, 4096, 4096, torch.bfloat16),
    ]

    all_pass = True
    for B, T, hidden, dtype in configs:
        # PyTorch reference path
        x_ref, w_ref = make_test_data(B, T, hidden, dtype=dtype)
        out_ref = pytorch_rmsnorm_ref(x_ref, w_ref)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)
        dx_ref = x_ref.grad.clone()
        dw_ref = w_ref.grad.clone()

        # New Triton path
        x_tri, w_tri = make_test_data(B, T, hidden, dtype=dtype)
        out_tri = triton_rmsnorm(x_tri, w_tri, eps=1e-6)
        out_tri.backward(grad_out)
        dx_tri = x_tri.grad.clone()
        dw_tri = w_tri.grad.clone()

        # bf16 dW accumulates across B*T rows, so rounding compounds
        dx_threshold = 1e-3 if dtype == torch.float32 else 5e-2
        dw_threshold = 1e-3 if dtype == torch.float32 else 2.0

        dx_max = (dx_ref - dx_tri).abs().max().item()
        dw_max = (dw_ref - dw_tri).abs().max().item()

        dx_ok = dx_max < dx_threshold
        dw_ok = dw_max < dw_threshold
        status = "✅" if (dx_ok and dw_ok) else "❌"
        if not (dx_ok and dw_ok):
            all_pass = False

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(
            f"   {status} B={B}, T={T}, H={hidden}, {dtype_str}:  "
            f"dX_max={dx_max:.2e}, dW_max={dw_max:.2e}"
        )

    return all_pass


def test_old_vs_new_forward():
    """Test that new Triton forward matches old Triton forward."""
    print("\n3. New Triton vs Old Triton Forward Match")
    print("   " + "-" * 56)

    configs = [
        (2, 128, 4096, torch.float32),
        (2, 128, 4096, torch.bfloat16),
    ]

    all_pass = True
    for B, T, hidden, dtype in configs:
        x, w = make_test_data(B, T, hidden, dtype=dtype)

        with torch.no_grad():
            out_old = triton_rmsnorm_fwd_only(x, w, eps=1e-6)
            out_new = triton_rmsnorm(x, w, eps=1e-6)

        max_diff = (out_old - out_new).abs().max().item()
        threshold = 1e-5 if dtype == torch.float32 else 1e-3
        status = "✅" if max_diff < threshold else "❌"
        if max_diff >= threshold:
            all_pass = False

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(
            f"   {status} B={B}, T={T}, H={hidden}, {dtype_str}:  max_diff={max_diff:.2e}"
        )

    return all_pass


# ═══════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════


def benchmark_fn(fn, *args, warmup=10, repeats=50, backward=False, grad_output=None):
    """Benchmark a function, measuring time and peak memory."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        if backward:
            out.backward(grad_output, retain_graph=True)
            for a in args:
                if hasattr(a, "grad") and a.grad is not None:
                    a.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
        if backward:
            out.backward(grad_output, retain_graph=True)
            for a in args:
                if hasattr(a, "grad") and a.grad is not None:
                    a.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000  # ms

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    return elapsed, peak_mem


def run_benchmarks():
    """3-way benchmark: PyTorch vs Old Triton vs New Liger Triton."""
    print("\n4. Performance Benchmark")
    print("   " + "=" * 68)

    configs = [
        # (B, T, hidden, dtype_name)
        (2, 512, 4096, "bf16"),
        (2, 2048, 4096, "bf16"),
        (2, 4096, 4096, "bf16"),
        (4, 4096, 4096, "bf16"),
    ]

    for B, T, hidden, dtype_name in configs:
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        print(f"\n   Config: B={B}, T={T}, H={hidden}, {dtype_name}")
        print(
            f"   {'Method':<28} {'FWD (ms)':>10} {'BWD (ms)':>10} {'Total':>10} {'Peak MB':>10}"
        )
        print(f"   {'-'*68}")

        # Generate data
        x_pt, w_pt = make_test_data(B, T, hidden, dtype=dtype)
        x_old, w_old = make_test_data(B, T, hidden, dtype=dtype)
        x_new, w_new = make_test_data(B, T, hidden, dtype=dtype)

        grad_out = torch.randn(B, T, hidden, device="cuda", dtype=dtype)

        # ── PyTorch Reference ──
        fwd_time, _ = benchmark_fn(pytorch_rmsnorm_ref, x_pt, w_pt)
        total_time, peak_mem = benchmark_fn(
            pytorch_rmsnorm_ref, x_pt, w_pt, backward=True, grad_output=grad_out
        )
        bwd_time = total_time - fwd_time
        print(
            f"   {'PyTorch RMSNorm':<28} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}"
        )

        pt_fwd = fwd_time
        pt_bwd = bwd_time
        pt_total = total_time

        # ── Old Triton (forward-only, backward via autograd) ──
        fwd_time, _ = benchmark_fn(triton_rmsnorm_fwd_only, x_old, w_old, 1e-6)
        # Old kernel doesn't support backward — use pytorch_rmsnorm_ref for backward
        # to simulate the .forward() path that the model used
        total_time, peak_mem = benchmark_fn(
            pytorch_rmsnorm_ref, x_old, w_old, backward=True, grad_output=grad_out
        )
        # Replace only forward time with old triton forward
        bwd_time = total_time - fwd_time
        old_total = fwd_time + bwd_time
        print(
            f"   {'Old Triton (fwd-only)':<28} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {old_total:>9.3f}  {peak_mem:>9.1f}"
        )

        old_fwd = fwd_time
        old_bwd = bwd_time

        # ── New Liger Triton (fused fwd + bwd) ──
        fwd_time, _ = benchmark_fn(triton_rmsnorm, x_new, w_new, 1e-6)
        total_time, peak_mem = benchmark_fn(
            triton_rmsnorm, x_new, w_new, 1e-6, backward=True, grad_output=grad_out
        )
        bwd_time = total_time - fwd_time
        print(
            f"   {'New Liger Triton (fwd+bwd)':<28} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}"
        )

        new_fwd = fwd_time
        new_bwd = bwd_time
        new_total = total_time

        # ── Speedup summary ──
        print(f"   {'-'*68}")
        if old_fwd > 0 and old_bwd > 0:
            print(
                f"   {'Speedup vs Old Triton':<28} "
                f"{old_fwd/new_fwd:>9.2f}x "
                f"{old_bwd/new_bwd:>9.2f}x "
                f"{old_total/new_total:>9.2f}x"
            )
        if pt_fwd > 0 and pt_bwd > 0:
            print(
                f"   {'Speedup vs PyTorch':<28} "
                f"{pt_fwd/new_fwd:>9.2f}x "
                f"{pt_bwd/new_bwd:>9.2f}x "
                f"{pt_total/new_total:>9.2f}x"
            )


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — cannot run Triton kernel tests.")
        print("   These tests must be run on a GPU machine (e.g., Colab T4).")
        sys.exit(0)

    if not HAS_TRITON:
        print("⚠️  Triton not installed — cannot run Triton kernel tests.")
        print("   Install with: pip install triton")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"  RMSNorm Kernel Tests — {gpu_name}")
    print(f"  3-Way: PyTorch | Old Triton (fwd-only) | New Liger Triton (fwd+bwd)")
    print("=" * 70)

    results = []
    results.append(("Forward Correctness", test_forward_correctness()))
    results.append(("Backward Correctness", test_backward_correctness()))
    results.append(("Old vs New Forward Match", test_old_vs_new_forward()))

    run_benchmarks()

    # Final summary
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n  🎉 ALL CORRECTNESS TESTS PASSED")
    else:
        print("\n  ⚠️  SOME TESTS FAILED — check above for details")
    print("=" * 70)
