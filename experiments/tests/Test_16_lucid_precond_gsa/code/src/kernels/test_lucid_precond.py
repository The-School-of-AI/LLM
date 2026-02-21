"""
LUCID Preconditioner — Correctness & Benchmark Tests
======================================================

3-way comparison:
  1. PyTorch Full-Matrix   (pure PyTorch — baseline reference)
  2. PyTorch Block-wise    (memory-efficient block solver)
  3. Triton Fused          (Triton RMS norm + cuBLAS TRSM + fused backward)

Tests:
  - Mathematical correctness: forward outputs + backward gradients
  - Speed benchmark: forward time, backward time, total time, peak memory
  - Reversibility: deterministic output under no_grad → enable_grad replay
  - dtype coverage: float32 and bfloat16

Run on a CUDA GPU:
    python test_lucid_precond.py
"""

import sys
import os
import time
import math
import torch
import torch.nn.functional as F

# Import LUCID preconditioner
from lucid_preconditioner import (
    pytorch_lucid_precondition,
    pytorch_lucid_precondition_blockwise,
    triton_lucid_precondition,
    lucid_precondition,
    _rms_normalize_keys,
    HAS_TRITON,
)


# ═══════════════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_test_data(B=2, T=32, H=4, D=64, dtype=torch.float32, device='cuda', seed=42):
    """Generate random keys and values for LUCID testing."""
    torch.manual_seed(seed)
    K = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    return K, V


# ═══════════════════════════════════════════════════════════════════════
# Correctness Tests
# ═══════════════════════════════════════════════════════════════════════

def test_unit_diagonal():
    """Test that P has unit diagonal after RMS normalization."""
    print("\n1. Unit Diagonal Property")
    print("   " + "-" * 56)

    configs = [
        (2, 32, 4, 64, torch.float32),
        (1, 64, 2, 128, torch.float32),
        (2, 32, 4, 64, torch.bfloat16),
    ]

    all_pass = True
    for B, T, H, D, dtype in configs:
        K, _ = make_test_data(B, T, H, D, dtype=dtype)
        K_flat = K.permute(0, 2, 1, 3).reshape(B * H, T, D)
        K_RN = _rms_normalize_keys(K_flat)
        sqrt_d = math.sqrt(D)

        diag_vals = (K_RN * K_RN).sum(dim=-1) / sqrt_d - sqrt_d  # should be ~0
        exp_diag = torch.exp(diag_vals)  # should be ~1
        max_err = (exp_diag - 1.0).abs().max().item()

        threshold = 1e-4 if dtype == torch.float32 else 1e-2
        status = "✅" if max_err < threshold else "❌"
        if max_err >= threshold:
            all_pass = False

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(f"   {status} B={B}, T={T}, H={H}, D={D}, {dtype_str}: "
              f"max|exp(diag)-1| = {max_err:.2e}")

    return all_pass


def test_forward_correctness():
    """Test that Triton forward matches PyTorch full-matrix reference."""
    print("\n2. Forward Correctness (Triton vs PyTorch Full-Matrix)")
    print("   " + "-" * 56)

    configs = [
        (2, 16, 2, 32, 8, torch.float32),
        (2, 32, 4, 64, 16, torch.float32),
        (1, 64, 2, 128, 32, torch.float32),
        (2, 32, 4, 64, 16, torch.bfloat16),
    ]

    all_pass = True
    for B, T, H, D, bs, dtype in configs:
        K, V = make_test_data(B, T, H, D, dtype=dtype)

        with torch.no_grad():
            Y_ref = pytorch_lucid_precondition(K, V)
            Y_block = pytorch_lucid_precondition_blockwise(K, V, block_size=bs)
            if HAS_TRITON:
                Y_triton = triton_lucid_precondition(K, V, block_size=bs)

        # Block vs Full
        diff_block = (Y_ref - Y_block).abs().max().item()
        threshold = 1e-4 if dtype == torch.float32 else 5e-2
        block_ok = diff_block < threshold

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        status = "✅" if block_ok else "❌"
        if not block_ok:
            all_pass = False
        print(f"   {status} Block vs Full  B={B},T={T},H={H},D={D},bs={bs},{dtype_str}: "
              f"max_diff={diff_block:.2e}")

        # Triton vs Full
        if HAS_TRITON:
            diff_triton = (Y_ref - Y_triton).abs().max().item()
            triton_ok = diff_triton < threshold
            status = "✅" if triton_ok else "❌"
            if not triton_ok:
                all_pass = False
            print(f"   {status} Triton vs Full B={B},T={T},H={H},D={D},bs={bs},{dtype_str}: "
                  f"max_diff={diff_triton:.2e}")

    return all_pass


def test_roundtrip():
    """Test P @ Y == V (the fundamental solve equation)."""
    print("\n3. Roundtrip P@Y == V")
    print("   " + "-" * 56)

    configs = [
        (2, 32, 4, 64, torch.float32),
        (1, 64, 2, 128, torch.float32),
    ]

    all_pass = True
    for B, T, H, D, dtype in configs:
        K, V = make_test_data(B, T, H, D, dtype=dtype)
        K_flat = K.permute(0, 2, 1, 3).reshape(B * H, T, D)
        V_flat = V.permute(0, 2, 1, 3).reshape(B * H, T, D)

        with torch.no_grad():
            K_RN = _rms_normalize_keys(K_flat)
            sqrt_d = math.sqrt(D)
            scores = torch.bmm(K_RN, K_RN.transpose(-2, -1)) / sqrt_d - sqrt_d
            causal = torch.tril(torch.ones(T, T, device='cuda', dtype=torch.bool))
            P = torch.exp(scores.masked_fill(~causal, float('-inf')))

            Y = pytorch_lucid_precondition(K, V)
            Y_flat = Y.permute(0, 2, 1, 3).reshape(B * H, T, D)

            reconstructed = torch.bmm(P, Y_flat)
            max_err = (reconstructed - V_flat).abs().max().item()

        status = "✅" if max_err < 1e-4 else "❌"
        if max_err >= 1e-4:
            all_pass = False
        print(f"   {status} B={B},T={T},H={H},D={D}: max|P@Y - V| = {max_err:.2e}")

    return all_pass


def test_backward_correctness():
    """Test that Triton backward gradients match PyTorch autograd reference."""
    print("\n4. Backward Correctness (dK and dV)")
    print("   " + "-" * 56)

    configs = [
        (1, 16, 2, 32, 8, torch.float32),
        (2, 16, 2, 32, 8, torch.float32),
        (1, 32, 2, 64, 16, torch.float32),
    ]

    all_pass = True
    for B, T, H, D, bs, dtype in configs:
        # PyTorch reference path (use full-matrix for clean autograd)
        K_ref, V_ref = make_test_data(B, T, H, D, dtype=dtype)
        Y_ref = pytorch_lucid_precondition(K_ref, V_ref)
        grad_out = torch.randn_like(Y_ref)
        Y_ref.backward(grad_out)
        dK_ref = K_ref.grad.clone()
        dV_ref = V_ref.grad.clone()

        # Triton path (fused fwd+bwd)
        K_tri, V_tri = make_test_data(B, T, H, D, dtype=dtype)
        Y_tri = lucid_precondition(K_tri, V_tri, block_size=bs, training=True)
        Y_tri.backward(grad_out)
        dK_tri = K_tri.grad.clone()
        dV_tri = V_tri.grad.clone()

        dV_max = (dV_ref - dV_tri).abs().max().item()
        dK_max = (dK_ref - dK_tri).abs().max().item()

        dV_threshold = 1e-3
        dK_threshold = 1e-2  # K gradient goes through more transforms

        dV_ok = dV_max < dV_threshold
        dK_ok = dK_max < dK_threshold

        status = "✅" if (dV_ok and dK_ok) else "❌"
        if not (dV_ok and dK_ok):
            all_pass = False

        print(f"   {status} B={B},T={T},H={H},D={D},bs={bs}: "
              f"dV_max={dV_max:.2e}, dK_max={dK_max:.2e}")

    return all_pass


def test_gradient_flow():
    """Test that gradients actually flow through the preconditioner."""
    print("\n5. Gradient Flow")
    print("   " + "-" * 56)

    all_pass = True
    for method_name, fn in [("lucid_precondition", lucid_precondition)]:
        K, V = make_test_data(2, 32, 4, 64)
        Y = fn(K, V, block_size=16, training=True)
        loss = Y.sum()
        loss.backward()

        k_grad_norm = K.grad.norm().item()
        v_grad_norm = V.grad.norm().item()

        ok = k_grad_norm > 0 and v_grad_norm > 0
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False

        print(f"   {status} {method_name}: K.grad norm={k_grad_norm:.4f}, "
              f"V.grad norm={v_grad_norm:.4f}")

    return all_pass


def test_reversibility():
    """
    Test that LUCID is safe for ReversibleMidpointStack.

    The preconditioner must produce IDENTICAL outputs in:
    1. Forward pass (torch.no_grad)
    2. Backward reconstruct (torch.enable_grad)

    This is guaranteed because LUCID is:
    - Deterministic (no randomness)
    - Stateless (no EMA, no running stats)
    - Pure function of K, V
    """
    print("\n6. Reversibility Check (no_grad vs enable_grad)")
    print("   " + "-" * 56)

    all_pass = True
    for B, T, H, D, bs in [(2, 32, 4, 64, 16), (1, 64, 2, 128, 32)]:
        K, V = make_test_data(B, T, H, D)

        # Simulate forward pass (no_grad, like reversible forward)
        with torch.no_grad():
            Y_fwd = lucid_precondition(K, V, block_size=bs, training=False)

        # Simulate backward reconstruct (enable_grad, like reversible backward)
        with torch.enable_grad():
            K_detach = K.detach()
            V_detach = V.detach()
            Y_reconstruct = lucid_precondition(K_detach, V_detach, block_size=bs, training=False)

        max_diff = (Y_fwd - Y_reconstruct).abs().max().item()
        ok = max_diff == 0.0  # Must be EXACTLY equal for reversibility
        status = "✅" if ok else "❌"
        if not ok:
            all_pass = False

        print(f"   {status} B={B},T={T},H={H},D={D},bs={bs}: "
              f"max_diff={max_diff:.2e} {'(exact match)' if ok else '(NOT exact!)'}")

    return all_pass


def test_edge_cases():
    """Test edge cases: T=1, T=block_size, T=block_size+1."""
    print("\n7. Edge Cases")
    print("   " + "-" * 56)

    all_pass = True
    bs = 8
    for T in [1, bs, bs + 1, 64, 65]:
        K, V = make_test_data(1, T, 2, 32)
        try:
            Y = lucid_precondition(K, V, block_size=bs, training=True)
            ok = Y.shape == K.shape
            status = "✅" if ok else "❌"
            if not ok:
                all_pass = False
            print(f"   {status} T={T}: shape={Y.shape}")
        except Exception as e:
            all_pass = False
            print(f"   ❌ T={T}: {e}")

    return all_pass


# ═══════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════

def benchmark_fn(fn, *args, warmup=5, repeats=20, backward=False, grad_output=None):
    """Benchmark a function, measuring time and peak memory."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
        if backward and grad_output is not None:
            out.backward(grad_output, retain_graph=True)
            for a in args:
                if hasattr(a, 'grad') and a.grad is not None:
                    a.grad = None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
        if backward and grad_output is not None:
            out.backward(grad_output, retain_graph=True)
            for a in args:
                if hasattr(a, 'grad') and a.grad is not None:
                    a.grad = None
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats * 1000  # ms

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return elapsed, peak_mem


def run_benchmarks():
    """3-way benchmark: PyTorch Full vs PyTorch Block vs Triton Fused."""
    print("\n8. Performance Benchmark")
    print("   " + "=" * 72)

    configs = [
        # (B, T, H, D, block_size, dtype_name)
        (2, 64, 4, 64, 32, "fp32"),
        (2, 128, 4, 64, 32, "fp32"),
        (2, 256, 4, 64, 64, "fp32"),
        (2, 128, 4, 64, 32, "bf16"),
        (2, 256, 4, 64, 64, "bf16"),
    ]

    for B, T, H, D, bs, dtype_name in configs:
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
        print(f"\n   Config: B={B}, T={T}, H={H}, D={D}, bs={bs}, {dtype_name}")
        print(f"   {'Method':<32} {'FWD (ms)':>10} {'BWD (ms)':>10} {'Total':>10} {'Peak MB':>10}")
        print(f"   {'-'*72}")

        grad_out = torch.randn(B, T, H, D, device='cuda', dtype=dtype)

        # ── PyTorch Full-Matrix ──
        K_pt, V_pt = make_test_data(B, T, H, D, dtype=dtype)
        try:
            fwd_time, _ = benchmark_fn(pytorch_lucid_precondition, K_pt, V_pt)
            total_time, peak_mem = benchmark_fn(
                pytorch_lucid_precondition, K_pt, V_pt,
                backward=True, grad_output=grad_out
            )
            bwd_time = total_time - fwd_time
            print(f"   {'PyTorch Full-Matrix':<32} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
            pt_fwd, pt_bwd, pt_total = fwd_time, bwd_time, total_time
        except Exception as e:
            print(f"   {'PyTorch Full-Matrix':<32} SKIP ({e})")
            pt_fwd, pt_bwd, pt_total = None, None, None

        # ── PyTorch Block-wise ──
        K_bw, V_bw = make_test_data(B, T, H, D, dtype=dtype)
        fwd_time, _ = benchmark_fn(
            lambda k, v: pytorch_lucid_precondition_blockwise(k, v, bs), K_bw, V_bw
        )
        # Block-wise backward uses full-matrix (via lucid_precondition autograd)
        total_time, peak_mem = benchmark_fn(
            lambda k, v: lucid_precondition(k, v, block_size=bs, training=True), K_bw, V_bw,
            backward=True, grad_output=grad_out
        )
        bwd_time = total_time - fwd_time
        print(f"   {'PyTorch Block-wise (fwd only)':<32} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
        bw_fwd = fwd_time

        # ── Triton Fused (fwd+bwd) ──
        if HAS_TRITON:
            K_tri, V_tri = make_test_data(B, T, H, D, dtype=dtype)
            fwd_time, _ = benchmark_fn(
                lambda k, v: triton_lucid_precondition(k, v, bs), K_tri, V_tri
            )
            total_time, peak_mem = benchmark_fn(
                lambda k, v: lucid_precondition(k, v, block_size=bs, training=True), K_tri, V_tri,
                backward=True, grad_output=grad_out
            )
            bwd_time = total_time - fwd_time
            print(f"   {'Triton Fused (fwd+bwd)':<32} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
            tri_fwd, tri_total = fwd_time, total_time

            # Speedup summary
            print(f"   {'-'*72}")
            if bw_fwd > 0:
                print(f"   {'Triton vs PyTorch Block (fwd)':<32} {bw_fwd/tri_fwd:>9.2f}x")
            if pt_total and pt_total > 0:
                print(f"   {'Triton vs PyTorch Full (total)':<32} {pt_total/tri_total:>9.2f}x")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — cannot run Triton kernel tests.")
        print("   These tests must be run on a GPU machine (e.g., Colab T4).")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 74)
    print(f"  LUCID Preconditioner Tests — {gpu_name}")
    print(f"  3-Way: PyTorch Full | PyTorch Block | Triton Fused (fwd+bwd)")
    print(f"  Triton available: {HAS_TRITON}")
    print("=" * 74)

    results = []
    results.append(("Unit Diagonal", test_unit_diagonal()))
    results.append(("Forward Correctness", test_forward_correctness()))
    results.append(("Roundtrip P@Y == V", test_roundtrip()))
    results.append(("Backward Correctness", test_backward_correctness()))
    results.append(("Gradient Flow", test_gradient_flow()))
    results.append(("Reversibility", test_reversibility()))
    results.append(("Edge Cases", test_edge_cases()))

    run_benchmarks()

    # Final summary
    print("\n" + "=" * 74)
    print("  RESULTS SUMMARY")
    print("=" * 74)
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
    print("=" * 74)
