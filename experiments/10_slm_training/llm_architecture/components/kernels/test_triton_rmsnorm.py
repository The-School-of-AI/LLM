"""
Test and Benchmark Triton RMSNorm Kernel
=========================================

Verifies correctness and measures performance of fused RMSNorm.
"""

import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from components.kernels.triton_normalization import (
    HAS_TRITON,
    TritonRMSNorm,
    pytorch_rmsnorm,
    triton_rmsnorm,
)
from components.normalization.rms_norm import RMSNorm


def test_correctness():
    """Test that Triton kernel matches PyTorch implementation."""
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)

    if not HAS_TRITON:
        print("❌ Triton not available, skipping test")
        return False

    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping test")
        return False

    device = torch.device("cuda")
    hidden_size = 2048
    batch_size = 4
    seq_len = 512

    # Create test data
    torch.manual_seed(42)
    x = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    residual = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    eps = 1e-6

    # Test 1: Without residual
    print("\nTest 1: RMSNorm without residual")
    pytorch_out = pytorch_rmsnorm(x, weight, eps, residual=None)
    triton_out = triton_rmsnorm(x, weight, eps, residual=None)

    max_diff = (pytorch_out - triton_out).abs().max().item()
    mean_diff = (pytorch_out - triton_out).abs().mean().item()
    relative_diff = (
        ((pytorch_out - triton_out).abs() / (pytorch_out.abs() + 1e-6)).mean().item()
    )

    print(f"  Max diff:      {max_diff:.2e}")
    print(f"  Mean diff:     {mean_diff:.2e}")
    print(f"  Relative diff: {relative_diff:.2%}")

    # bfloat16 has ~3 decimal digits of precision, so we expect ~1e-2 to 1e-3 error
    tolerance = 0.1  # 10% tolerance for bfloat16
    if max_diff < tolerance:
        print("  ✅ PASSED")
        test1_passed = True
    else:
        print(f"  ❌ FAILED (tolerance: {tolerance})")
        test1_passed = False

    # Test 2: With residual
    print("\nTest 2: RMSNorm with residual")
    pytorch_out = pytorch_rmsnorm(x, weight, eps, residual=residual)
    triton_out = triton_rmsnorm(x, weight, eps, residual=residual)

    max_diff = (pytorch_out - triton_out).abs().max().item()
    mean_diff = (pytorch_out - triton_out).abs().mean().item()
    relative_diff = (
        ((pytorch_out - triton_out).abs() / (pytorch_out.abs() + 1e-6)).mean().item()
    )

    print(f"  Max diff:      {max_diff:.2e}")
    print(f"  Mean diff:     {mean_diff:.2e}")
    print(f"  Relative diff: {relative_diff:.2%}")

    tolerance = 0.15  # Slightly higher for residual (more ops)
    if max_diff < tolerance:
        print("  ✅ PASSED")
        test2_passed = True
    else:
        print(f"  ❌ FAILED (tolerance: {tolerance})")
        test2_passed = False

    # Test 3: Compare with original RMSNorm module
    print("\nTest 3: Compare with original RMSNorm")
    orig_norm = RMSNorm(hidden_size, eps=eps).to(device)
    orig_norm.weight.data = weight

    triton_norm = TritonRMSNorm(hidden_size, eps=eps).to(device)
    triton_norm.weight.data = weight

    orig_out = orig_norm(x)
    triton_out = triton_norm(x, residual=None)

    max_diff = (orig_out - triton_out).abs().max().item()
    mean_diff = (orig_out - triton_out).abs().mean().item()
    relative_diff = (
        ((orig_out - triton_out).abs() / (orig_out.abs() + 1e-6)).mean().item()
    )

    print(f"  Max diff:      {max_diff:.2e}")
    print(f"  Mean diff:     {mean_diff:.2e}")
    print(f"  Relative diff: {relative_diff:.2%}")

    tolerance = 0.1
    if max_diff < tolerance:
        print("  ✅ PASSED")
        test3_passed = True
    else:
        print(f"  ❌ FAILED (tolerance: {tolerance})")
        test3_passed = False

    all_passed = test1_passed and test2_passed and test3_passed
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests PASSED")
    else:
        print("❌ Some tests FAILED")
    print("=" * 60)

    return all_passed


def benchmark_performance():
    """Benchmark Triton kernel vs PyTorch."""
    print("\n" + "=" * 60)
    print("Benchmarking Performance")
    print("=" * 60)

    if not HAS_TRITON or not torch.cuda.is_available():
        print("❌ Triton or CUDA not available, skipping benchmark")
        return

    device = torch.device("cuda")
    hidden_size = 2048
    batch_size = 6
    seq_len = 2048

    # Create test data
    x = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    residual = torch.randn(
        batch_size, seq_len, hidden_size, device=device, dtype=torch.bfloat16
    )
    weight = torch.randn(hidden_size, device=device, dtype=torch.bfloat16)
    eps = 1e-6

    # Warmup
    for _ in range(10):
        _ = pytorch_rmsnorm(x, weight, eps, residual)
        _ = triton_rmsnorm(x, weight, eps, residual)

    torch.cuda.synchronize()

    # Benchmark PyTorch
    print("\nBenchmarking PyTorch RMSNorm...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    n_iters = 100
    start.record()
    for _ in range(n_iters):
        _ = pytorch_rmsnorm(x, weight, eps, residual)
    end.record()
    torch.cuda.synchronize()

    pytorch_time = start.elapsed_time(end) / n_iters
    print(f"  PyTorch: {pytorch_time:.3f} ms")

    # Benchmark Triton
    print("\nBenchmarking Triton RMSNorm...")
    start.record()
    for _ in range(n_iters):
        _ = triton_rmsnorm(x, weight, eps, residual)
    end.record()
    torch.cuda.synchronize()

    triton_time = start.elapsed_time(end) / n_iters
    print(f"  Triton:  {triton_time:.3f} ms")

    # Compute speedup
    speedup = pytorch_time / triton_time
    print(f"\n{'=' * 60}")
    print(f"Speedup: {speedup:.2f}x")
    if speedup > 1.0:
        print(f"✅ Triton is {speedup:.2f}x faster!")
    else:
        print(f"⚠️  PyTorch is {1/speedup:.2f}x faster")
    print("=" * 60)

    return speedup


def main():
    print("\n🔧 Testing Triton RMSNorm Kernel\n")

    # Run tests
    correctness_passed = test_correctness()

    if correctness_passed:
        # Run benchmarks
        speedup = benchmark_performance()

        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print("✅ Correctness: PASSED")
        if speedup:
            print(f"📊 Performance: {speedup:.2f}x speedup")
        print("=" * 60)
    else:
        print("\n❌ Correctness tests failed, skipping benchmark")


if __name__ == "__main__":
    main()
