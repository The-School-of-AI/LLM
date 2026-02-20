"""
Correctness test: Triton sparse attention backward vs PyTorch reference.

Verifies that TritonSparseAttnFn.backward produces the same gradients
as pytorch_sparse_attention's autograd.

Run on a CUDA GPU:
    python test_triton_sparse_attn_backward.py
"""

import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kernels.triton_sparse_attn import (
    HAS_TRITON,
    pytorch_sparse_attention,
    triton_sparse_attention,
)


def make_test_data(B=2, T=64, H=4, D=32, k_sel=16, device="cuda", seed=42):
    """Generate random Q, K, V and valid sparse indices/mask."""
    torch.manual_seed(seed)

    q = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.float32, requires_grad=True)

    # Random sparse indices: each query selects k_sel positions from [0, T)
    # indices: [B, H, T, k_sel], causal: each query can only attend to positions <= itself
    indices = torch.zeros(B, H, T, k_sel, dtype=torch.int64, device=device)
    for b in range(B):
        for t in range(T):
            valid_range = t + 1  # causal: 0..t
            if valid_range >= k_sel:
                idx = torch.randperm(valid_range, device=device)[:k_sel].sort().values
            else:
                # Fewer valid positions than k_sel — pad with 0
                idx = torch.arange(valid_range, device=device)
                idx = torch.cat(
                    [
                        idx,
                        torch.zeros(
                            k_sel - valid_range, dtype=torch.long, device=device
                        ),
                    ]
                )
            indices[b, :, t, :] = idx  # same indices for all heads (shared indexer)

    # Mask: 1.0 for valid, 0.0 for padding
    mask = torch.ones(B, H, T, k_sel, dtype=torch.float32, device=device)
    for t in range(T):
        valid_range = t + 1
        if valid_range < k_sel:
            mask[:, :, t, valid_range:] = 0.0

    scale = 1.0 / (D**0.5)
    return q, k, v, indices, mask, scale


def test_forward_match():
    """Test that Triton and PyTorch forward outputs match."""
    q, k, v, indices, mask, scale = make_test_data()

    with torch.no_grad():
        out_triton = triton_sparse_attention(q, k, v, indices, mask, scale)
        out_pytorch = pytorch_sparse_attention(q, k, v, indices, mask, scale)

    max_diff = (out_triton - out_pytorch).abs().max().item()
    mean_diff = (out_triton - out_pytorch).abs().mean().item()
    print(f"Forward:  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")
    assert max_diff < 1e-3, f"Forward mismatch: max_diff={max_diff}"
    print("  ✅ Forward match OK")


def test_backward_match():
    """Test that Triton and PyTorch backward gradients match."""
    # Generate data
    q_ref, k_ref, v_ref, indices, mask, scale = make_test_data()

    # Clone for Triton path (separate grad graphs)
    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    # ── PyTorch reference ──────────────────────────────────────
    out_ref = pytorch_sparse_attention(q_ref, k_ref, v_ref, indices, mask, scale)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    dq_ref = q_ref.grad.clone()
    dk_ref = k_ref.grad.clone()
    dv_ref = v_ref.grad.clone()

    # ── Triton ─────────────────────────────────────────────────
    out_tri = triton_sparse_attention(q_tri, k_tri, v_tri, indices, mask, scale)
    out_tri.backward(grad_out)

    dq_tri = q_tri.grad.clone()
    dk_tri = k_tri.grad.clone()
    dv_tri = v_tri.grad.clone()

    # ── Compare ────────────────────────────────────────────────
    for name, ref, tri in [
        ("dQ", dq_ref, dq_tri),
        ("dK", dk_ref, dk_tri),
        ("dV", dv_ref, dv_tri),
    ]:
        max_diff = (ref - tri).abs().max().item()
        mean_diff = (ref - tri).abs().mean().item()
        ref_norm = ref.abs().mean().item()
        rel_diff = mean_diff / max(ref_norm, 1e-8)
        print(
            f"  {name}:  max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, rel_diff={rel_diff:.2e}"
        )
        assert max_diff < 1e-2, f"{name} mismatch: max_diff={max_diff}"

    print("  ✅ Backward gradients match OK")


def test_backward_sizes():
    """Test with various sizes to catch edge cases."""
    configs = [
        (1, 32, 2, 16, 8),  # small
        (2, 64, 4, 32, 16),  # medium
        (1, 128, 8, 64, 32),  # larger heads/dim
        (2, 16, 2, 16, 4),  # k_sel << T
        (1, 8, 1, 8, 8),  # k_sel == T (dense-ish)
    ]
    for B, T, H, D, k_sel in configs:
        print(f"  Config: B={B}, T={T}, H={H}, D={D}, k_sel={k_sel}")
        q, k, v, indices, mask, scale = make_test_data(B, T, H, D, k_sel)

        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)

        out_ref = pytorch_sparse_attention(q, k, v, indices, mask, scale)
        grad_out = torch.randn_like(out_ref)
        out_ref.backward(grad_out)

        out_tri = triton_sparse_attention(q2, k2, v2, indices, mask, scale)
        out_tri.backward(grad_out)

        for name, ref_g, tri_g in [
            ("dQ", q.grad, q2.grad),
            ("dK", k.grad, k2.grad),
            ("dV", v.grad, v2.grad),
        ]:
            max_diff = (ref_g - tri_g).abs().max().item()
            assert (
                max_diff < 5e-2
            ), f"{name} mismatch at {(B,T,H,D,k_sel)}: max_diff={max_diff}"

        print(f"    ✅ OK")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available — cannot run Triton kernel tests.")
        print("   These tests must be run on a GPU machine.")
        sys.exit(0)

    if not HAS_TRITON:
        print("⚠️  Triton not installed — cannot run Triton kernel tests.")
        sys.exit(0)

    print("=" * 60)
    print("Triton Sparse Attention Backward — Correctness Tests")
    print("=" * 60)

    print("\n1. Forward match test:")
    test_forward_match()

    print("\n2. Backward gradient match test:")
    test_backward_match()

    print("\n3. Multi-size backward test:")
    test_backward_sizes()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)
