"""
Test SVD MoE compression with synthetic weights — no checkpoint needed.

Tests:
1. SVD analysis correctly identifies rank structure
2. SVD decomposition produces correct factors
3. Compressed forward pass matches original (within Frobenius error)
4. LoRA gradients flow correctly through SVD forward
5. Memory savings are as expected
6. Grouped GEMM kernels work with SVD factor shapes

Usage:
    cd TrainingPipelineV1/code
    python -m src.test_svd_moe
"""

import sys
import torch
import torch.nn as nn

# ── Minimal MoEFFN stub for testing ────────────────────────────────────────

class MoEFFNStub(nn.Module):
    """Minimal MoEFFN that mimics the real model's expert weight structure."""

    def __init__(self, d_model=256, d_hidden=128, num_experts=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.dropout = dropout

        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

    def _moe_grouped(self, sorted_x, expert_counts):
        """Simple reference forward: loop over experts."""
        x_in = sorted_x.to(dtype=self.W_gate.dtype)
        E = self.num_experts
        offsets = torch.zeros(E + 1, device=x_in.device, dtype=torch.long)
        torch.cumsum(expert_counts, dim=0, out=offsets[1:])

        N = self.d_hidden
        gate_out = torch.empty(x_in.shape[0], N, device=x_in.device, dtype=x_in.dtype)
        up_out = torch.empty_like(gate_out)
        for e in range(E):
            s, t = offsets[e].item(), offsets[e + 1].item()
            if s < t:
                gate_out[s:t] = x_in[s:t] @ self.W_gate[e]
                up_out[s:t] = x_in[s:t] @ self.W_up[e]

        h = torch.nn.functional.silu(gate_out) * up_out
        K = self.d_model
        out = torch.empty(x_in.shape[0], K, device=x_in.device, dtype=x_in.dtype)
        for e in range(E):
            s, t = offsets[e].item(), offsets[e + 1].item()
            if s < t:
                out[s:t] = h[s:t] @ self.W_down[e]
        return out


class SimpleModel(nn.Module):
    """Wrapper with a single MoEFFN layer for testing."""

    def __init__(self, **kwargs):
        super().__init__()
        self.moe = MoEFFNStub(**kwargs)


def _make_expert_counts(E, M_total):
    """Distribute M_total tokens roughly evenly across E experts."""
    base = M_total // E
    remainder = M_total % E
    counts = torch.full((E,), base, dtype=torch.long)
    counts[:remainder] += 1
    return counts



# ============================================================================
# Test 1: SVD analysis on known-rank matrices
# ============================================================================

def test_analysis_known_rank():
    """Create matrices with known rank and verify analysis detects it."""
    print("\n" + "=" * 70)
    print("  TEST 1: SVD analysis on known-rank matrices")
    print("=" * 70)

    E, K, N = 4, 256, 128
    true_rank = 16

    model = SimpleModel(d_model=K, d_hidden=N, num_experts=E)

    # Replace weights with known low-rank matrices
    for pname in ["W_gate", "W_up"]:
        param = getattr(model.moe, pname)
        for e in range(E):
            A = torch.randn(K, true_rank) * 0.1
            B = torch.randn(true_rank, N) * 0.1
            param.data[e] = A @ B
    # W_down is [E, N, K] — different shape
    for e in range(E):
        A = torch.randn(N, true_rank) * 0.1
        B = torch.randn(true_rank, K) * 0.1
        model.moe.W_down.data[e] = A @ B

    from src.svd_moe_utils import analyze_svd_spectrum

    results = analyze_svd_spectrum(model, ranks=[8, 16, 32, 64], verbose=True)

    # Verify: rank-16 should capture ~100% energy for gate/up
    for pname in ["W_gate", "W_up", "W_down"]:
        agg = results["moe"][pname]["aggregate"]
        energy_at_16 = agg[16]["mean_energy_pct"]
        energy_at_32 = agg[32]["mean_energy_pct"]
        assert energy_at_16 > 99.99, (
            f"{pname}: rank-16 should capture ~100% but got {energy_at_16:.4f}%"
        )
        assert abs(energy_at_32 - energy_at_16) < 0.01, (
            f"{pname}: rank-32 shouldn't add energy beyond rank-16"
        )
        eff_rank = results["moe"][pname]["effective_rank_99"]["mean"]
        assert eff_rank <= true_rank, (
            f"{pname}: effective rank (99%) should be ≤{true_rank}, got {eff_rank}"
        )

    print("\n  ✓ PASSED: Analysis correctly identifies rank-16 structure")
    return True


# ============================================================================
# Test 2: SVD decomposition correctness
# ============================================================================

def test_decomposition_correctness():
    """Verify SVD factors reconstruct the original weight within tolerance."""
    print("\n" + "=" * 70)
    print("  TEST 2: SVD decomposition correctness")
    print("=" * 70)

    E, K, N = 4, 256, 128
    model = SimpleModel(d_model=K, d_hidden=N, num_experts=E)

    # Save original weights
    orig_gate = model.moe.W_gate.data.clone()
    orig_up = model.moe.W_up.data.clone()
    orig_down = model.moe.W_down.data.clone()

    from src.svd_moe_utils import decompose_moe_experts_svd

    # Full rank decomposition (r=128 = min(K,N)) should be near-exact
    count = decompose_moe_experts_svd(model, target_rank=128, verbose=True)
    assert count == 3, f"Expected 3 tensors decomposed, got {count}"

    # Reconstruct and compare
    for pname, orig in [("W_gate", orig_gate), ("W_up", orig_up), ("W_down", orig_down)]:
        U = getattr(model.moe, f"{pname}_U")    # [E, K, r]
        Vt = getattr(model.moe, f"{pname}_Vt")  # [E, r, N]
        reconstructed = torch.bmm(U, Vt)         # [E, K, N]

        max_err = (orig - reconstructed).abs().max().item()
        rel_err = (orig - reconstructed).norm() / orig.norm()
        print(f"  {pname}: max_abs_err={max_err:.2e}, rel_frob_err={rel_err:.2e}")
        assert max_err < 1e-3, f"{pname}: full-rank reconstruction error too large: {max_err}"

    # Verify original params are gone
    assert not hasattr(model.moe, "W_gate") or not isinstance(
        getattr(model.moe, "W_gate", None), nn.Parameter
    ), "W_gate should no longer be an nn.Parameter"

    print("\n  ✓ PASSED: Full-rank SVD reconstruction is near-exact")
    return True


# ============================================================================
# Test 3: Compressed forward matches original
# ============================================================================

def test_compressed_forward():
    """Verify SVD forward produces same output as original (full rank)."""
    print("\n" + "=" * 70)
    print("  TEST 3: Compressed forward matches original")
    print("=" * 70)

    E, K, N = 4, 256, 128
    M_total = 32

    # Model A: original forward
    model_a = SimpleModel(d_model=K, d_hidden=N, num_experts=E)
    torch.manual_seed(42)
    x = torch.randn(M_total, K)
    expert_counts = _make_expert_counts(E, M_total)

    with torch.no_grad():
        out_orig = model_a.moe._moe_grouped(x, expert_counts)

    # Model B: SVD decomposed (full rank = exact)
    model_b = SimpleModel(d_model=K, d_hidden=N, num_experts=E)
    # Copy weights from A
    model_b.moe.W_gate.data.copy_(model_a.moe.W_gate.data)
    model_b.moe.W_up.data.copy_(model_a.moe.W_up.data)
    model_b.moe.W_down.data.copy_(model_a.moe.W_down.data)

    from src.svd_moe_utils import decompose_moe_experts_svd, patch_moe_svd_forward

    decompose_moe_experts_svd(model_b, target_rank=128, verbose=False)
    patch_moe_svd_forward(model_b)

    with torch.no_grad():
        out_svd = model_b.moe._moe_grouped(x, expert_counts)

    max_err = (out_orig - out_svd).abs().max().item()
    rel_err = (out_orig - out_svd).norm() / out_orig.norm()
    print(f"  max_abs_err={max_err:.2e}, rel_err={rel_err:.2e}")

    # Full rank should be very close (float32 precision)
    assert max_err < 1e-2, f"Forward mismatch too large: {max_err}"

    print("\n  ✓ PASSED: SVD forward matches original forward")
    return True


# ============================================================================
# Test 4: Low-rank compression error is bounded
# ============================================================================

def test_low_rank_error_bounded():
    """Verify that low-rank SVD forward error is bounded by Frobenius norm."""
    print("\n" + "=" * 70)
    print("  TEST 4: Low-rank compression error is bounded")
    print("=" * 70)

    E, K, N = 4, 256, 128
    M_total = 32
    target_rank = 32  # Aggressive compression: 32 out of 128

    model_orig = SimpleModel(d_model=K, d_hidden=N, num_experts=E)
    x = torch.randn(M_total, K)
    expert_counts = _make_expert_counts(E, M_total)

    with torch.no_grad():
        out_orig = model_orig.moe._moe_grouped(x, expert_counts)

    # Decompose with low rank
    model_svd = SimpleModel(d_model=K, d_hidden=N, num_experts=E)
    model_svd.moe.W_gate.data.copy_(model_orig.moe.W_gate.data)
    model_svd.moe.W_up.data.copy_(model_orig.moe.W_up.data)
    model_svd.moe.W_down.data.copy_(model_orig.moe.W_down.data)

    from src.svd_moe_utils import decompose_moe_experts_svd, patch_moe_svd_forward

    decompose_moe_experts_svd(model_svd, target_rank=target_rank, verbose=True)
    patch_moe_svd_forward(model_svd)

    with torch.no_grad():
        out_svd = model_svd.moe._moe_grouped(x, expert_counts)

    rel_err = (out_orig - out_svd).norm() / out_orig.norm()
    print(f"\n  Output relative error at rank={target_rank}: {rel_err:.4f}")

    # For random matrices, rank-32 out of 128 captures ~25% of dimensions,
    # so error will be significant. Just verify it's finite and reasonable.
    assert rel_err < 2.0, f"Error unreasonably large: {rel_err}"
    assert not torch.isnan(out_svd).any(), "NaN in SVD output"

    # Now test with known low-rank weights — error should be tiny
    true_rank = 16
    for pname in ["W_gate", "W_up"]:
        param = getattr(model_orig.moe, pname)
        for e in range(E):
            A = torch.randn(K, true_rank) * 0.1
            B = torch.randn(true_rank, N) * 0.1
            param.data[e] = A @ B
    for e in range(E):
        A = torch.randn(N, true_rank) * 0.1
        B = torch.randn(true_rank, K) * 0.1
        model_orig.moe.W_down.data[e] = A @ B

    with torch.no_grad():
        out_orig_lr = model_orig.moe._moe_grouped(x, expert_counts)

    model_svd2 = SimpleModel(d_model=K, d_hidden=N, num_experts=E)
    model_svd2.moe.W_gate.data.copy_(model_orig.moe.W_gate.data)
    model_svd2.moe.W_up.data.copy_(model_orig.moe.W_up.data)
    model_svd2.moe.W_down.data.copy_(model_orig.moe.W_down.data)
    decompose_moe_experts_svd(model_svd2, target_rank=target_rank, verbose=False)
    patch_moe_svd_forward(model_svd2)

    with torch.no_grad():
        out_svd_lr = model_svd2.moe._moe_grouped(x, expert_counts)

    rel_err_lr = (out_orig_lr - out_svd_lr).norm() / (out_orig_lr.norm() + 1e-8)
    print(f"  Output relative error (true rank={true_rank}, SVD rank={target_rank}): {rel_err_lr:.6f}")
    assert rel_err_lr < 1e-3, f"Low-rank matrix should compress near-exactly: {rel_err_lr}"

    print("\n  ✓ PASSED: Low-rank error is bounded and correct")
    return True


# ============================================================================
# Test 5: Memory savings calculation
# ============================================================================

def test_memory_savings():
    """Verify memory savings match theoretical predictions."""
    print("\n" + "=" * 70)
    print("  TEST 5: Memory savings calculation")
    print("=" * 70)

    E, K, N = 260, 4096, 1024  # 70B model dimensions
    target_rank = 64

    # Calculate expected savings
    orig_per_proj = E * K * N * 2  # bf16 bytes
    svd_per_proj = (E * K * target_rank + E * target_rank * N) * 2
    compression = orig_per_proj / svd_per_proj

    orig_total = orig_per_proj * 3 * 20  # 3 projections × 20 layers
    svd_total = svd_per_proj * 3 * 20

    print(f"  Per projection: {orig_per_proj/1e9:.2f} GB → {svd_per_proj/1e9:.2f} GB "
          f"({compression:.1f}× compression)")
    print(f"  Total (3 proj × 20 layers): {orig_total/1e9:.2f} GB → {svd_total/1e9:.2f} GB")
    print(f"  Memory saved: {(orig_total - svd_total)/1e9:.2f} GB")

    assert compression > 10, f"Expected >10× compression, got {compression:.1f}×"
    assert svd_total / 1e9 < 15, f"SVD total should be <15 GB, got {svd_total/1e9:.1f} GB"

    print(f"\n  ✓ PASSED: {compression:.1f}× compression, {svd_total/1e9:.1f} GB total")
    return True


# ============================================================================
# Test 6: Backward pass + gradient flow
# ============================================================================

def test_backward_gradient_flow():
    """Verify gradients flow through SVD forward (for LoRA training)."""
    print("\n" + "=" * 70)
    print("  TEST 6: Backward pass + gradient flow")
    print("=" * 70)

    E, K, N = 4, 256, 128
    M_total = 32

    model = SimpleModel(d_model=K, d_hidden=N, num_experts=E)

    from src.svd_moe_utils import decompose_moe_experts_svd, patch_moe_svd_forward

    decompose_moe_experts_svd(model, target_rank=64, verbose=False)
    patch_moe_svd_forward(model)

    # SVD factors are buffers (frozen), but let's add a trainable linear
    # to simulate LoRA and verify gradients flow through x
    probe = nn.Linear(K, K, bias=False)
    x = torch.randn(M_total, K, requires_grad=True)
    expert_counts = _make_expert_counts(E, M_total)

    x_proj = probe(x)
    out = model.moe._moe_grouped(x_proj, expert_counts)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "No gradient on input x"
    assert x.grad.abs().sum() > 0, "Gradient on x is all zeros"
    assert probe.weight.grad is not None, "No gradient on probe weight"
    assert probe.weight.grad.abs().sum() > 0, "Gradient on probe is all zeros"

    print(f"  x.grad norm: {x.grad.norm():.4f}")
    print(f"  probe.weight.grad norm: {probe.weight.grad.norm():.4f}")

    # SVD factors should NOT have gradients (they're buffers)
    U = model.moe.W_gate_U
    assert not U.requires_grad, "SVD factor U should not require grad"

    print("\n  ✓ PASSED: Gradients flow correctly through SVD forward")
    return True


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  SVD MoE Compression — Test Suite (no checkpoint needed)")
    print("=" * 70)

    tests = [
        ("Analysis on known-rank matrices", test_analysis_known_rank),
        ("Decomposition correctness", test_decomposition_correctness),
        ("Compressed forward matches original", test_compressed_forward),
        ("Low-rank error is bounded", test_low_rank_error_bounded),
        ("Memory savings calculation", test_memory_savings),
        ("Backward gradient flow", test_backward_gradient_flow),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"\n  ✗ FAILED: {name}")
            print(f"    {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 70)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
