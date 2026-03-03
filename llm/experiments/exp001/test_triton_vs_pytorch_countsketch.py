"""
Comparison test: Triton fused CountSketch vs PyTorch fallback.

Tests:
  1. Correctness — both paths produce identical sketches
  2. Benchmarks — wall-clock comparison at different scales

Run on a GPU machine:
    python3 test_triton_vs_pytorch_countsketch.py
"""
import sys
import time

import os
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _this_dir)

import torch

# ─── Check CUDA availability ─────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("⚠️  No CUDA device found. Running correctness test on CPU only.")
    print("   (Benchmarking requires GPU. Triton path will use PyTorch fallback.)\n")
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
    print(f"🔧 Using GPU: {torch.cuda.get_device_name(0)}\n")

# ─── Import both paths ───────────────────────────────────────────────────────
from exp.opus.triton_countsketch import HAS_TRITON, fused_sketch_scatter

print(f"Triton available: {HAS_TRITON}")


# ═══════════════════════════════════════════════════════════════════════════════
# PyTorch reference implementation (the code we're replacing)
# ═══════════════════════════════════════════════════════════════════════════════
def pytorch_sketch_scatter(
    grad_chunk: torch.Tensor,
    preconditioner: torch.Tensor | None,
    pair_sign: torch.Tensor,
    pair_hash: torch.Tensor,
    sketches: torch.Tensor,
) -> None:
    """Exact copy of the old inline PyTorch code from project_linear_batch."""
    B = grad_chunk.shape[0]
    if preconditioner is not None:
        grad_chunk = grad_chunk * preconditioner.unsqueeze(0)
    signed = (grad_chunk * pair_sign.unsqueeze(0)).reshape(B, -1)
    idx = pair_hash.reshape(1, -1).expand(B, -1)
    sketches.scatter_add_(1, idx, signed)


# ═══════════════════════════════════════════════════════════════════════════════
# Test harness
# ═══════════════════════════════════════════════════════════════════════════════
def run_test(B, R, C, M, has_precond, label):
    """Run one correctness + speed comparison."""
    torch.manual_seed(42)

    grad = torch.randn(B, R, C, device=DEVICE, dtype=torch.float32)
    precond = (
        torch.randn(R, C, device=DEVICE, dtype=torch.float32).abs() + 0.1
        if has_precond
        else None
    )
    pair_sign = (
        torch.randint(0, 2, (R, C), device=DEVICE, dtype=torch.float32) * 2 - 1
    )
    pair_hash = torch.randint(0, M, (R, C), device=DEVICE, dtype=torch.int64)

    # ── PyTorch reference ──
    sketch_ref = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
    pytorch_sketch_scatter(
        grad.clone(), precond, pair_sign, pair_hash, sketch_ref
    )

    # ── Fused (Triton or PyTorch fallback) ──
    sketch_fused = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
    fused_sketch_scatter(
        grad.clone(), precond, pair_sign, pair_hash, sketch_fused
    )

    # ── Correctness ──
    max_abs = (sketch_ref - sketch_fused).abs().max().item()
    max_val = sketch_ref.abs().max().item()
    rel_diff = max_abs / (max_val + 1e-8)

    passed = rel_diff < 1e-5
    status = "✅" if passed else "❌"
    print(
        f"  {status} {label:40s}  "
        f"max_abs={max_abs:.2e}  rel={rel_diff:.2e}  "
        f"shape=({B},{R},{C}) M={M} P={'yes' if has_precond else 'no'}"
    )

    # ── Benchmark (GPU only) ──
    if DEVICE.type == "cuda":
        warmup = 10
        iters = 100

        # Warmup
        for _ in range(warmup):
            s = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
            pytorch_sketch_scatter(grad.clone(), precond, pair_sign, pair_hash, s)
        torch.cuda.synchronize()

        # Time PyTorch
        t0 = time.perf_counter()
        for _ in range(iters):
            s = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
            pytorch_sketch_scatter(grad.clone(), precond, pair_sign, pair_hash, s)
        torch.cuda.synchronize()
        t_pytorch = (time.perf_counter() - t0) / iters * 1000

        # Warmup fused
        for _ in range(warmup):
            s = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
            fused_sketch_scatter(grad.clone(), precond, pair_sign, pair_hash, s)
        torch.cuda.synchronize()

        # Time fused
        t0 = time.perf_counter()
        for _ in range(iters):
            s = torch.zeros(B, M, device=DEVICE, dtype=torch.float32)
            fused_sketch_scatter(grad.clone(), precond, pair_sign, pair_hash, s)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / iters * 1000

        speedup = t_pytorch / t_fused if t_fused > 0 else float("inf")
        faster = "FUSED" if speedup > 1 else "PYTORCH"
        print(
            f"       ⏱  pytorch={t_pytorch:.3f}ms  fused={t_fused:.3f}ms  "
            f"speedup={speedup:.2f}×  ({faster} wins)"
        )

    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  Correctness + Benchmark: Fused vs PyTorch CountSketch Scatter")
print("=" * 80)

all_passed = True

# Test config scale (hidden=512, chunk=64)
print("\n── Test Config Scale (hidden=512) ──")
all_passed &= run_test(18, 64, 512, 512, True, "chunk=64, with precond")
all_passed &= run_test(18, 64, 512, 512, False, "chunk=64, no precond")
all_passed &= run_test(18, 512, 512, 512, True, "full (no chunking)")

# Production scale (hidden=4096, chunk=64)
print("\n── Production Scale (hidden=4096) ──")
all_passed &= run_test(40, 64, 4096, 8192, True, "chunk=64, with precond")
all_passed &= run_test(40, 64, 4096, 8192, False, "chunk=64, no precond")
all_passed &= run_test(40, 256, 4096, 8192, True, "chunk=256, with precond")

# Edge cases
print("\n── Edge Cases ──")
all_passed &= run_test(1, 16, 32, 64, True, "tiny (B=1)")
all_passed &= run_test(64, 64, 512, 8192, True, "large batch (B=64)")

print("\n" + "=" * 80)
if all_passed:
    print("  ✅ ALL TESTS PASSED")
else:
    print("  ❌ SOME TESTS FAILED")
    sys.exit(1)
print("=" * 80)
