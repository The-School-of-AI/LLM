"""
LUCID Preconditioner — Correctness Tests
=========================================

Verifies:
1. PyTorch full-matrix: P @ Y == V roundtrip
2. Block-wise == full-matrix (same output)
3. Triton == PyTorch (if CUDA available)
4. Gradient flow: autograd works through the preconditioner
5. Edge cases: T=1, T=block_size, T=block_size+1

Run on GPU:
    python test_lucid_precond.py
"""

import torch
import math
import sys


def test_pytorch_roundtrip():
    """Verify P @ Y == V (solve then multiply back)."""
    from lucid_preconditioner import pytorch_lucid_precondition, _rms_normalize_keys

    B, T, H, D = 2, 16, 4, 64
    torch.manual_seed(42)
    K = torch.randn(B, T, H, D)
    V = torch.randn(B, T, H, D)

    if torch.cuda.is_available():
        K, V = K.cuda(), V.cuda()

    Y = pytorch_lucid_precondition(K, V)

    # Reconstruct P and verify P @ Y ≈ V
    K_flat = K.permute(0, 2, 1, 3).reshape(B * H, T, D)
    K_RN = _rms_normalize_keys(K_flat)
    sqrt_d = math.sqrt(D)
    scores = torch.bmm(K_RN, K_RN.transpose(-2, -1)) / sqrt_d - sqrt_d
    causal = torch.tril(torch.ones(T, T, device=K.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal, float('-inf'))
    P = torch.exp(scores)

    Y_flat = Y.permute(0, 2, 1, 3).reshape(B * H, T, D)
    V_flat = V.permute(0, 2, 1, 3).reshape(B * H, T, D)

    V_reconstructed = torch.bmm(P, Y_flat)
    err = (V_reconstructed - V_flat).abs().max().item()
    print(f"[roundtrip] max |P@Y - V| = {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")
    assert err < 1e-4, f"Roundtrip error too large: {err}"


def test_blockwise_matches_full():
    """Verify block-wise solver matches full-matrix solver."""
    from lucid_preconditioner import pytorch_lucid_precondition, pytorch_lucid_precondition_blockwise

    B, T, H, D = 2, 32, 4, 64
    torch.manual_seed(42)
    K = torch.randn(B, T, H, D)
    V = torch.randn(B, T, H, D)

    if torch.cuda.is_available():
        K, V = K.cuda(), V.cuda()

    Y_full = pytorch_lucid_precondition(K, V)

    for bs in [8, 16, 32]:
        Y_block = pytorch_lucid_precondition_blockwise(K, V, block_size=bs)
        err = (Y_full - Y_block).abs().max().item()
        print(f"[blockwise bs={bs}] max |full - block| = {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")
        assert err < 1e-4, f"Block-wise error too large for bs={bs}: {err}"


def test_triton_matches_pytorch():
    """Verify Triton path matches PyTorch path."""
    from lucid_preconditioner import (
        pytorch_lucid_precondition_blockwise,
        triton_lucid_precondition,
        HAS_TRITON,
    )

    if not torch.cuda.is_available():
        print("[triton] SKIP — no CUDA")
        return
    if not HAS_TRITON:
        print("[triton] SKIP — no Triton")
        return

    B, T, H, D = 2, 32, 4, 64
    torch.manual_seed(42)
    K = torch.randn(B, T, H, D, device="cuda")
    V = torch.randn(B, T, H, D, device="cuda")

    Y_pt = pytorch_lucid_precondition_blockwise(K, V, block_size=16)
    Y_tr = triton_lucid_precondition(K, V, block_size=16)

    err = (Y_pt - Y_tr).abs().max().item()
    print(f"[triton] max |pytorch - triton| = {err:.2e}  {'PASS' if err < 1e-4 else 'FAIL'}")
    assert err < 1e-4, f"Triton vs PyTorch error: {err}"


def test_gradient_flow():
    """Verify gradients flow through the preconditioner."""
    from lucid_preconditioner import lucid_precondition

    B, T, H, D = 2, 16, 4, 64
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    K = torch.randn(B, T, H, D, device=device, requires_grad=True)
    V = torch.randn(B, T, H, D, device=device, requires_grad=True)

    Y = lucid_precondition(K, V, block_size=8, training=True)
    loss = Y.sum()
    loss.backward()

    assert K.grad is not None, "No gradient for K"
    assert V.grad is not None, "No gradient for V"
    assert K.grad.abs().sum() > 0, "K gradient is all zeros"
    assert V.grad.abs().sum() > 0, "V gradient is all zeros"
    print(f"[gradient] K.grad norm={K.grad.norm():.4f}, V.grad norm={V.grad.norm():.4f}  PASS")


def test_edge_cases():
    """Test edge cases: T=1, T=block_size, T=block_size+1."""
    from lucid_preconditioner import lucid_precondition

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for T in [1, 8, 9, 64, 65]:
        K = torch.randn(1, T, 2, 32, device=device)
        V = torch.randn(1, T, 2, 32, device=device)
        Y = lucid_precondition(K, V, block_size=8, training=False)
        assert Y.shape == V.shape, f"Shape mismatch for T={T}"
        assert torch.isfinite(Y).all(), f"Non-finite output for T={T}"
        print(f"[edge T={T}] shape={Y.shape}  PASS")


def test_unit_diagonal():
    """Verify the preconditioner has unit diagonal (self-similarity = 1)."""
    from lucid_preconditioner import _rms_normalize_keys

    B, T, D = 2, 16, 64
    torch.manual_seed(42)
    K = torch.randn(B, T, D)
    if torch.cuda.is_available():
        K = K.cuda()

    K_RN = _rms_normalize_keys(K)
    sqrt_d = math.sqrt(D)

    # Self dot product should be d, so score = d/√d - √d = √d - √d = 0
    # exp(0) = 1 → unit diagonal
    self_dot = (K_RN * K_RN).sum(dim=-1)  # [B, T]
    scores = self_dot / sqrt_d - sqrt_d
    diag_vals = torch.exp(scores)

    err = (diag_vals - 1.0).abs().max().item()
    print(f"[unit_diag] max |exp(diag) - 1| = {err:.2e}  {'PASS' if err < 1e-5 else 'FAIL'}")
    assert err < 1e-5, f"Diagonal not unit: {err}"


def main():
    print("=" * 60)
    print("LUCID Preconditioner — Correctness Tests")
    print("=" * 60)

    tests = [
        ("Unit diagonal property", test_unit_diagonal),
        ("P@Y == V roundtrip", test_pytorch_roundtrip),
        ("Block-wise == full-matrix", test_blockwise_matches_full),
        ("Triton == PyTorch", test_triton_matches_pytorch),
        ("Gradient flow", test_gradient_flow),
        ("Edge cases", test_edge_cases),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name} ---")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
