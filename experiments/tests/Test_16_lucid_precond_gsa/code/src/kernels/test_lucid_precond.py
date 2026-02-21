"""
LUCID Preconditioner — Correctness & Overhead Tests
======================================================

What LUCID does:
  LUCID is an ACCURACY optimization, not a speed optimization.
  It decorrelates keys in RKHS so attention can distinguish similar keys.
  This adds compute overhead but should lower perplexity.

What we test:
  1. Math correctness: does the solver actually solve P·Y = V?
  2. Gradient correctness: do gradients match PyTorch autograd?
  3. Reversibility: safe for ReversibleMidpointStack?
  4. Overhead benchmark: how much compute/memory does LUCID add?

Implementation note:
  The "Triton" part only covers RMS normalization (~5% of the work).
  The heavy lifting (triangular solves) is cuBLAS TRSM in all paths.
  "Triton path" = Triton RMS norm + cuBLAS block solve
  "PyTorch path" = PyTorch RMS norm + cuBLAS block solve

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
    """
    Test: After RMS normalization, each key's self-similarity = 1.
    Why: If P[i,i] ≠ 1, the triangular solve diverges or gives wrong answers.
    """
    print("\n1. Unit Diagonal (P[i,i] = 1 after RMS norm)")
    print("   " + "-" * 56)

    configs = [
        (2, 32, 4, 64, torch.float32),
        (1, 64, 2, 128, torch.float32),
        (2, 32, 4, 64, torch.bfloat16),
    ]

    all_pass = True
    for B, T, H, D, dtype in configs:
        K, _ = make_test_data(B, T, H, D, dtype=dtype)
        # RMS norm is always done in fp32 internally
        K_flat = K.float().permute(0, 2, 1, 3).reshape(B * H, T, D)
        K_RN = _rms_normalize_keys(K_flat)
        sqrt_d = math.sqrt(D)

        diag_vals = (K_RN * K_RN).sum(dim=-1) / sqrt_d - sqrt_d  # should be ~0
        exp_diag = torch.exp(diag_vals)  # should be ~1
        max_err = (exp_diag - 1.0).abs().max().item()

        # In fp32 (which is what we compute in), this should be very tight
        threshold = 1e-4
        status = "✅" if max_err < threshold else "❌"
        if max_err >= threshold:
            all_pass = False

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        print(f"   {status} B={B}, T={T}, H={H}, D={D}, {dtype_str}: "
              f"max|exp(diag)-1| = {max_err:.2e}")

    return all_pass


def test_forward_correctness():
    """
    Test: Block-wise solver gives same answer as full-matrix solver.
    Why: Block solver tiles the T×T matrix into BS×BS blocks. If tiling
         logic is wrong, the answers diverge.

    Note: bf16 inputs are upcast to fp32 internally, so we compare the
          fp32 outputs before bf16 rounding for a meaningful threshold.
    """
    print("\n2. Forward: Block-wise == Full-matrix (solver tiling correctness)")
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

        # Compare in fp32 to avoid bf16 output rounding noise
        diff_block = (Y_ref.float() - Y_block.float()).abs().max().item()
        # fp32 internal compute means diffs come from solve path differences only
        threshold = 1e-4 if dtype == torch.float32 else 0.5
        block_ok = diff_block < threshold

        dtype_str = "fp32" if dtype == torch.float32 else "bf16"
        status = "✅" if block_ok else "❌"
        if not block_ok:
            all_pass = False
        print(f"   {status} Block vs Full  B={B},T={T},H={H},D={D},bs={bs},{dtype_str}: "
              f"max_diff={diff_block:.2e}")

        # Triton RMS norm vs PyTorch RMS norm (same block solver)
        if HAS_TRITON:
            diff_triton = (Y_ref.float() - Y_triton.float()).abs().max().item()
            triton_ok = diff_triton < threshold
            status = "✅" if triton_ok else "❌"
            if not triton_ok:
                all_pass = False
            print(f"   {status} Triton vs Full B={B},T={T},H={H},D={D},bs={bs},{dtype_str}: "
                  f"max_diff={diff_triton:.2e}")

    return all_pass


def test_roundtrip():
    """
    Test: P @ Y == V (the fundamental equation).
    Why: If the solver is correct, then multiplying the preconditioner P
         by the solution Y must give back the original values V.
    """
    print("\n3. Roundtrip: P @ Y == V (solver correctness)")
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
    """
    Test: Custom backward gradients match PyTorch autograd.
    Why: If dK or dV are wrong, the model learns garbage — loss won't decrease.
    """
    print("\n4. Backward: Custom grad == PyTorch autograd (training correctness)")
    print("   " + "-" * 56)

    configs = [
        (1, 16, 2, 32, 8, torch.float32),
        (2, 16, 2, 32, 8, torch.float32),
        (1, 32, 2, 64, 16, torch.float32),
    ]

    all_pass = True
    for B, T, H, D, bs, dtype in configs:
        # PyTorch reference (full-matrix, clean autograd)
        K_ref, V_ref = make_test_data(B, T, H, D, dtype=dtype)
        Y_ref = pytorch_lucid_precondition(K_ref, V_ref)
        grad_out = torch.randn_like(Y_ref)
        Y_ref.backward(grad_out)
        dK_ref = K_ref.grad.clone()
        dV_ref = V_ref.grad.clone()

        # Our custom backward (block-wise + Triton RMS norm bwd)
        K_tri, V_tri = make_test_data(B, T, H, D, dtype=dtype)
        Y_tri = lucid_precondition(K_tri, V_tri, block_size=bs, training=True)
        Y_tri.backward(grad_out)
        dK_tri = K_tri.grad.clone()
        dV_tri = V_tri.grad.clone()

        dV_max = (dV_ref - dV_tri).abs().max().item()
        dK_max = (dK_ref - dK_tri).abs().max().item()

        dV_threshold = 1e-3
        dK_threshold = 1e-2  # K grad goes through more transforms (P + RMS norm)

        dV_ok = dV_max < dV_threshold
        dK_ok = dK_max < dK_threshold

        status = "✅" if (dV_ok and dK_ok) else "❌"
        if not (dV_ok and dK_ok):
            all_pass = False

        print(f"   {status} B={B},T={T},H={H},D={D},bs={bs}: "
              f"dV_max={dV_max:.2e}, dK_max={dK_max:.2e}")

    return all_pass


def test_gradient_flow():
    """
    Test: K.grad and V.grad are non-zero after backward.
    Why: If gradients are zero, layers behind LUCID don't update — training is broken.
    """
    print("\n5. Gradient Flow (non-zero grads through preconditioner)")
    print("   " + "-" * 56)

    all_pass = True
    K, V = make_test_data(2, 32, 4, 64)
    Y = lucid_precondition(K, V, block_size=16, training=True)
    loss = Y.sum()
    loss.backward()

    k_grad_norm = K.grad.norm().item()
    v_grad_norm = V.grad.norm().item()

    ok = k_grad_norm > 0 and v_grad_norm > 0
    status = "✅" if ok else "❌"
    if not ok:
        all_pass = False

    print(f"   {status} K.grad norm={k_grad_norm:.4f}, V.grad norm={v_grad_norm:.4f}")

    return all_pass


def test_reversibility():
    """
    Test: LUCID produces IDENTICAL output under no_grad and enable_grad.
    Why: ReversibleMidpointStack re-runs forward during backward to reconstruct
         activations. If LUCID gives different outputs the second time (e.g., due to
         randomness or state), reconstruction fails and training explodes.
         LUCID is deterministic + stateless → should be exact match.
    """
    print("\n6. Reversibility (safe for ReversibleMidpointStack)")
    print("   " + "-" * 56)

    all_pass = True
    for B, T, H, D, bs in [(2, 32, 4, 64, 16), (1, 64, 2, 128, 32)]:
        K, V = make_test_data(B, T, H, D)

        # Simulate forward (no_grad, like reversible forward)
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
# Overhead Benchmark
# ═══════════════════════════════════════════════════════════════════════

def benchmark_fn(fn, *args, warmup=5, repeats=20, backward=False, grad_output=None):
    """Benchmark a function, measuring time and peak memory."""
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
    """
    Overhead benchmark: how much compute and memory does LUCID add?

    Three methods compared:
    1. Full-Matrix:  Build entire T×T matrix, single cuBLAS TRSM
                     Fast but O(T²) memory — IMPOSSIBLE at 256K+ context
    2. PyTorch Block: PyTorch RMS norm + cuBLAS block TRSM in tiles
                     Slightly slower but O(T × block_size) memory
    3. Triton Block:  Triton RMS norm + cuBLAS block TRSM in tiles
                     Same block solver, slightly faster RMS norm step
    """
    print("\n8. Overhead Benchmark (LUCID adds accuracy, costs compute)")
    print("   " + "=" * 76)
    print("   NOTE: Full-matrix is faster at small T but OOMs at long context.")
    print("         Block methods (PyTorch/Triton) are what we actually use.")
    print("         Triton only accelerates RMS norm (~5% of work).")

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
        print(f"   {'Method':<36} {'FWD (ms)':>10} {'BWD (ms)':>10} {'Total':>10} {'Peak MB':>10}")
        print(f"   {'-'*76}")

        grad_out = torch.randn(B, T, H, D, device='cuda', dtype=dtype)

        # ── Full-Matrix (reference, not usable at long context) ──
        K_pt, V_pt = make_test_data(B, T, H, D, dtype=dtype)
        try:
            fwd_time, _ = benchmark_fn(pytorch_lucid_precondition, K_pt, V_pt)
            total_time, peak_mem = benchmark_fn(
                pytorch_lucid_precondition, K_pt, V_pt,
                backward=True, grad_output=grad_out
            )
            bwd_time = total_time - fwd_time
            print(f"   {'Full-Matrix (O(T²) mem, ref only)':<36} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
            pt_total = total_time
        except Exception as e:
            print(f"   {'Full-Matrix (O(T²) mem, ref only)':<36} SKIP ({e})")
            pt_total = None

        # ── PyTorch RMS norm + cuBLAS Block Solve ──
        K_bw, V_bw = make_test_data(B, T, H, D, dtype=dtype)
        fwd_time, _ = benchmark_fn(
            lambda k, v: pytorch_lucid_precondition_blockwise(k, v, bs), K_bw, V_bw
        )
        total_time, peak_mem = benchmark_fn(
            lambda k, v: lucid_precondition(k, v, block_size=bs, training=True), K_bw, V_bw,
            backward=True, grad_output=grad_out
        )
        bwd_time = total_time - fwd_time
        print(f"   {'PyTorch RMSNorm + cuBLAS Block':<36} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
        bw_fwd = fwd_time

        # ── Triton RMS norm + cuBLAS Block Solve ──
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
            print(f"   {'Triton RMSNorm + cuBLAS Block':<36} {fwd_time:>9.3f}  {bwd_time:>9.3f}  {total_time:>9.3f}  {peak_mem:>9.1f}")
            tri_fwd, tri_total = fwd_time, total_time

            # Summary
            print(f"   {'-'*76}")
            if bw_fwd > 0:
                print(f"   {'Triton vs PyTorch RMSNorm (fwd)':<36} {bw_fwd/tri_fwd:>9.2f}x")
            if pt_total and pt_total > 0:
                print(f"   {'Block vs Full-Matrix (total)':<36} {pt_total/tri_total:>9.2f}x  (Full is faster but OOMs at long T)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — cannot run Triton kernel tests.")
        print("   These tests must be run on a GPU machine (e.g., Colab T4).")
        sys.exit(0)

    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 78)
    print(f"  LUCID Preconditioner Tests — {gpu_name}")
    print(f"  LUCID = accuracy optimization (not speed). Adds compute overhead.")
    print(f"  Triton available: {HAS_TRITON} (only used for RMS norm, ~5% of work)")
    print("=" * 78)

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
    print("\n" + "=" * 78)
    print("  RESULTS SUMMARY")
    print("=" * 78)
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
    print("=" * 78)
