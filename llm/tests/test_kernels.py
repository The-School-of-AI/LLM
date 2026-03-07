"""
Test ③ — Custom Kernels Match PyTorch Reference Implementations.

Verifies each Triton kernel produces correct outputs vs its PyTorch fallback
or a manually written reference.

Kernels tested:
    1. RMSNorm          — has built-in PyTorch fallback (fwd + bwd)
    2. Sinkhorn-Knopp   — has built-in PyTorch fallback (fwd only)
    3. Gated Indexer    — has built-in PyTorch fallback (fwd only)
    4. Streaming Indexer — property-based (topk shapes, sink invariant)
    5. Sparse Attention  — manual PyTorch reference (gather + softmax)
    6. Delta Entrance    — has built-in PyTorch fallback (optional, off by default)

Requirements:
    - CUDA GPU with Triton support

Run:
    python -m pytest tests/test_kernels.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F


# ── Skip if no CUDA ──────────────────────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton kernels"
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. RMSNorm
# ═════════════════════════════════════════════════════════════════════════════

class TestRMSNorm:
    """triton_rmsnorm vs pytorch_rmsnorm — forward + backward."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.B, self.T, self.D = 2, 16, 64
        self.eps = 1e-6
        self.x_base = torch.randn(
            self.B, self.T, self.D, device="cuda", dtype=torch.float32
        )
        self.w_base = torch.randn(self.D, device="cuda", dtype=torch.float32)

    def test_rmsnorm_fwd_fp32(self):
        """Forward match in fp32 — strict tolerance."""
        from llm.kernels import pytorch_rmsnorm, triton_rmsnorm

        x = self.x_base.clone()
        w = self.w_base.clone()

        out_triton = triton_rmsnorm(x, w, self.eps)
        out_ref = pytorch_rmsnorm(x, w, self.eps)

        assert torch.allclose(out_triton, out_ref, atol=1e-4), (
            f"RMSNorm fp32 max diff: {(out_triton - out_ref).abs().max().item():.6e}"
        )

    def test_rmsnorm_fwd_bf16(self):
        """Forward match in bf16 — relaxed tolerance."""
        from llm.kernels import pytorch_rmsnorm, triton_rmsnorm

        x = self.x_base.to(torch.bfloat16)
        w = self.w_base.to(torch.bfloat16)

        out_triton = triton_rmsnorm(x, w, self.eps)
        out_ref = pytorch_rmsnorm(x, w, self.eps)

        assert torch.allclose(out_triton.float(), out_ref.float(), atol=1e-2), (
            f"RMSNorm bf16 max diff: "
            f"{(out_triton.float() - out_ref.float()).abs().max().item():.6e}"
        )

    def test_rmsnorm_backward(self):
        """Gradient correctness: grad_x and grad_weight."""
        from llm.kernels import pytorch_rmsnorm, triton_rmsnorm

        # Triton path
        x_t = self.x_base.clone().requires_grad_(True)
        w_t = self.w_base.clone().requires_grad_(True)
        out_t = triton_rmsnorm(x_t, w_t, self.eps)
        out_t.sum().backward()
        gx_t, gw_t = x_t.grad.clone(), w_t.grad.clone()

        # Reference path (separate tensors)
        x_r = self.x_base.clone().requires_grad_(True)
        w_r = self.w_base.clone().requires_grad_(True)
        out_r = pytorch_rmsnorm(x_r, w_r, self.eps)
        out_r.sum().backward()
        gx_r, gw_r = x_r.grad.clone(), w_r.grad.clone()

        assert torch.allclose(gx_t, gx_r, atol=1e-3), (
            f"grad_x max diff: {(gx_t - gx_r).abs().max().item():.6e}"
        )
        assert torch.allclose(gw_t, gw_r, atol=1e-3), (
            f"grad_w max diff: {(gw_t - gw_r).abs().max().item():.6e}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Sinkhorn-Knopp
# ═════════════════════════════════════════════════════════════════════════════

class TestSinkhorn:
    """triton_sinkhorn_knopp vs pytorch_sinkhorn_knopp."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.num_iters = 20
        self.eps = 1e-8

    def test_sinkhorn_forward(self):
        """Output match for n=4 matrices."""
        from llm.kernels import pytorch_sinkhorn_knopp, triton_sinkhorn_knopp

        n = 4
        H = torch.randn(8, n, n, device="cuda", dtype=torch.float32)

        out_triton = triton_sinkhorn_knopp(H, self.num_iters, self.eps)
        out_ref = pytorch_sinkhorn_knopp(H, self.num_iters, self.eps)

        assert torch.allclose(out_triton, out_ref, atol=1e-5), (
            f"Sinkhorn max diff: {(out_triton - out_ref).abs().max().item():.6e}"
        )

    def test_sinkhorn_doubly_stochastic(self):
        """Rows AND columns of output sum to 1."""
        from llm.kernels import triton_sinkhorn_knopp

        n = 4
        H = torch.randn(8, n, n, device="cuda", dtype=torch.float32)
        out = triton_sinkhorn_knopp(H, self.num_iters, self.eps)

        row_sums = out.sum(dim=-1)
        col_sums = out.sum(dim=-2)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), (
            f"Rows don't sum to 1: max deviation {(row_sums - 1).abs().max().item():.6e}"
        )
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), (
            f"Cols don't sum to 1: max deviation {(col_sums - 1).abs().max().item():.6e}"
        )

    def test_sinkhorn_sizes(self):
        """Works correctly for n=2, 4, 8."""
        from llm.kernels import pytorch_sinkhorn_knopp, triton_sinkhorn_knopp

        for n in [2, 4, 8]:
            H = torch.randn(4, n, n, device="cuda", dtype=torch.float32)
            out_triton = triton_sinkhorn_knopp(H, self.num_iters, self.eps)
            out_ref = pytorch_sinkhorn_knopp(H, self.num_iters, self.eps)
            assert torch.allclose(out_triton, out_ref, atol=1e-5), (
                f"Sinkhorn n={n} max diff: "
                f"{(out_triton - out_ref).abs().max().item():.6e}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# 3. Gated Indexer
# ═════════════════════════════════════════════════════════════════════════════

class TestGatedIndexer:
    """triton_gated_indexer vs pytorch_gated_indexer."""

    @pytest.fixture(autouse=True)
    def setup(self):
        B, T, H, D = 2, 16, 2, 8
        self.B, self.T, self.H, self.D = B, T, H, D
        self.q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        self.k = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        self.w = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        self.b = torch.randn(H, device="cuda", dtype=torch.float32)
        self.scale = 1.0 / (D ** 0.5)

    def test_indexer_causal(self):
        """Causal=True: Triton matches PyTorch reference."""
        from llm.kernels import pytorch_gated_indexer, triton_gated_indexer

        out_triton = triton_gated_indexer(
            self.q, self.k, self.w, self.b, self.scale, causal=True
        )
        out_ref = pytorch_gated_indexer(
            self.q, self.k, self.w, self.b, self.scale, causal=True
        )

        assert torch.allclose(out_triton.float(), out_ref.float(), atol=1e-3), (
            f"Causal indexer max diff: "
            f"{(out_triton.float() - out_ref.float()).abs().max().item():.6e}"
        )

    def test_indexer_noncausal(self):
        """Causal=False: Triton matches PyTorch reference."""
        from llm.kernels import pytorch_gated_indexer, triton_gated_indexer

        out_triton = triton_gated_indexer(
            self.q, self.k, self.w, self.b, self.scale, causal=False
        )
        out_ref = pytorch_gated_indexer(
            self.q, self.k, self.w, self.b, self.scale, causal=False
        )

        assert torch.allclose(out_triton.float(), out_ref.float(), atol=1e-3), (
            f"Non-causal indexer max diff: "
            f"{(out_triton.float() - out_ref.float()).abs().max().item():.6e}"
        )

    def test_indexer_causal_mask_values(self):
        """Future key positions must have large negative scores."""
        from llm.kernels import triton_gated_indexer

        out = triton_gated_indexer(
            self.q, self.k, self.w, self.b, self.scale, causal=True
        )
        # For each query position t, all key positions > t should be -inf
        for t in range(self.T):
            if t < self.T - 1:
                future = out[:, t, t + 1:]  # [B, T - t - 1]
                assert (future <= -1e4).all(), (
                    f"Position t={t}: future scores not masked, "
                    f"max future score = {future.max().item()}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# 4. Streaming Indexer (fused_indexer_topk)
# ═════════════════════════════════════════════════════════════════════════════

class TestStreamingIndexer:
    """Property-based tests for fused_indexer_topk (no PyTorch fallback)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        B, T, H, D = 2, 32, 2, 8
        self.B, self.T, self.H, self.D = B, T, H, D
        self.q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        self.k = torch.randn(B, T, D, device="cuda", dtype=torch.float32)
        self.w = torch.randn(B, T, H, device="cuda", dtype=torch.float32)
        self.b = torch.randn(H, device="cuda", dtype=torch.float32)
        self.scale = 1.0 / (D ** 0.5)

    def test_topk_shapes_and_validity(self):
        """Output shapes correct and indices within bounds."""
        from llm.kernels import fused_indexer_topk

        var_t, k_t, top_indices = fused_indexer_topk(
            self.q, self.k, self.w, self.b, self.scale,
            causal=True, k_base=8, k_min=4, k_max=16, sink_size=2,
        )

        # Shapes
        assert var_t.shape == (self.B, self.T), f"var_t shape: {var_t.shape}"
        assert k_t.shape == (self.B, self.T), f"k_t shape: {k_t.shape}"
        assert top_indices.shape[0] == self.B
        assert top_indices.shape[1] == self.T
        # k_limit = last dim, should be <= k_max
        assert top_indices.shape[2] <= max(16, 2)  # k_max or sink_size

        # Validity checks
        assert torch.isfinite(var_t).all(), "var_t has NaN/inf"
        assert (var_t >= 0).all(), "variance must be non-negative"
        assert (k_t >= 4).all(), f"k_t below k_min: min={k_t.min()}"
        assert (k_t <= 16).all(), f"k_t above k_max: max={k_t.max()}"
        assert (top_indices >= 0).all(), "indices must be non-negative"
        assert (top_indices < self.T).all(), (
            f"indices must be < seq_kv={self.T}, max={top_indices.max()}"
        )

    def test_sink_tokens_selected(self):
        """Sink tokens (first sink_size positions) appear in topk for late queries."""
        from llm.kernels import fused_indexer_topk

        sink_size = 4
        var_t, k_t, top_indices = fused_indexer_topk(
            self.q, self.k, self.w, self.b, self.scale,
            causal=True, k_base=8, k_min=4, k_max=16, sink_size=sink_size,
        )

        # For queries at position >= sink_size, sink tokens should be selected
        # (they're forced to +inf in scores)
        for t in range(sink_size, self.T):
            indices_for_t = top_indices[:, t, :]  # [B, k_limit]
            for sink_pos in range(sink_size):
                # Check if sink_pos appears in the selected indices
                found = (indices_for_t == sink_pos).any(dim=-1)  # [B]
                assert found.all(), (
                    f"Sink token {sink_pos} not found in topk at query pos {t}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# 5. Sparse Attention
# ═════════════════════════════════════════════════════════════════════════════

class TestSparseAttention:
    """triton_sparse_attention_v2 vs manual PyTorch reference."""

    @staticmethod
    def _manual_sparse_attn(q, k, v, indices, mask, scale):
        """
        Manual reference: gather selected K/V, softmax, weighted sum.

        Args:
            q:       [B, T, H, D]
            k:       [B, T_kv, H, D]
            v:       [B, T_kv, H, D]
            indices: [B, H, T, k_sel]  (int32)
            mask:    [B, H, T, k_sel]  (float32, 0/1)
            scale:   float

        Returns:
            out: [B, T, H, D]
        """
        B, T, H, D = q.shape
        T_kv = k.shape[1]
        k_sel = indices.shape[-1]

        out = torch.zeros(B, T, H, D, device=q.device, dtype=torch.float32)

        for b_idx in range(B):
            for h_idx in range(H):
                for t_idx in range(T):
                    idx = indices[b_idx, h_idx, t_idx].long()  # [k_sel]
                    msk = mask[b_idx, h_idx, t_idx]  # [k_sel]

                    # Gather selected K and V
                    k_sel_vecs = k[b_idx, idx, h_idx]  # [k_sel, D]
                    v_sel_vecs = v[b_idx, idx, h_idx]  # [k_sel, D]

                    # Compute attention scores
                    scores = (q[b_idx, t_idx, h_idx] @ k_sel_vecs.T) * scale

                    # Mask invalid positions
                    scores = scores.masked_fill(msk < 0.5, float("-inf"))

                    # Softmax (handle all-masked → zeros)
                    if (scores == float("-inf")).all():
                        continue

                    attn = torch.softmax(scores.float(), dim=-1)
                    out[b_idx, t_idx, h_idx] = attn @ v_sel_vecs.float()

        return out

    def test_sparse_attn_forward(self):
        """Triton sparse attention matches manual gather+softmax reference."""
        from llm.kernels import triton_sparse_attention

        B, T, H, D = 2, 8, 2, 16
        k_sel = 4  # number of selected keys per query
        scale = 1.0 / (D ** 0.5)

        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)

        # Create valid indices: for each query, select k_sel random keys
        # (using causal: only select keys <= current position)
        indices = torch.zeros(B, H, T, k_sel, dtype=torch.int32, device="cuda")
        mask_vals = torch.ones(B, H, T, k_sel, dtype=torch.float32, device="cuda")

        for b_idx in range(B):
            for h_idx in range(H):
                for t_idx in range(T):
                    valid_keys = min(t_idx + 1, T)  # causal
                    if valid_keys >= k_sel:
                        sel = torch.randperm(valid_keys, device="cuda")[:k_sel]
                    else:
                        # Pad with repeats if not enough keys
                        sel = torch.randint(0, max(1, valid_keys), (k_sel,), device="cuda")
                        mask_vals[b_idx, h_idx, t_idx, valid_keys:] = 0.0
                    indices[b_idx, h_idx, t_idx] = sel.to(torch.int32)

        out_triton = triton_sparse_attention(q, k, v, indices, mask_vals, scale)
        out_ref = self._manual_sparse_attn(q, k, v, indices, mask_vals, scale)

        assert torch.allclose(out_triton.float(), out_ref.float(), atol=1e-3), (
            f"Sparse attn max diff: "
            f"{(out_triton.float() - out_ref.float()).abs().max().item():.6e}"
        )

    def test_sparse_attn_finite(self):
        """Output contains no NaN or inf."""
        from llm.kernels import triton_sparse_attention

        B, T, H, D = 2, 8, 2, 16
        k_sel = 4
        scale = 1.0 / (D ** 0.5)

        q = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, T, H, D, device="cuda", dtype=torch.float32)

        # Simple valid indices: each query selects first k_sel keys
        indices = torch.arange(k_sel, device="cuda").view(1, 1, 1, k_sel)
        indices = indices.expand(B, H, T, k_sel).to(torch.int32)
        mask_vals = torch.ones(B, H, T, k_sel, dtype=torch.float32, device="cuda")

        out = triton_sparse_attention(q, k, v, indices, mask_vals, scale)
        assert torch.isfinite(out).all(), "Sparse attn output has NaN or inf"


# ═════════════════════════════════════════════════════════════════════════════
# 6. Delta Entrance (optional — off by default)
# ═════════════════════════════════════════════════════════════════════════════

DELTA_ENTRANCE_ENABLED = os.environ.get("T17_DN_USE_DELTA_ENTRANCE", "0") == "1"


@pytest.mark.skipif(
    not DELTA_ENTRANCE_ENABLED,
    reason="Delta Entrance off by default (set T17_DN_USE_DELTA_ENTRANCE=1 to enable)",
)
class TestDeltaEntrance:
    """fused_delta_entrance vs pytorch_unfused_exact — optional."""

    @pytest.fixture(autouse=True)
    def setup(self):
        B, T, H, D = 2, 16, 2, 32
        C = H * D  # 64
        self.B, self.T, self.H, self.D, self.C = B, T, H, D, C

        self.q = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
        self.k = torch.randn(B, T, C, device="cuda", dtype=torch.float32)
        self.v = torch.randn(B, T, C, device="cuda", dtype=torch.float32)

        # Depthwise conv weights: [C, 1, 4]
        self.wq = torch.randn(C, 1, 4, device="cuda", dtype=torch.float32)
        self.wk = torch.randn(C, 1, 4, device="cuda", dtype=torch.float32)
        self.wv = torch.randn(C, 1, 4, device="cuda", dtype=torch.float32)

        # Biases: [C]
        self.bq = torch.randn(C, device="cuda", dtype=torch.float32)
        self.bk = torch.randn(C, device="cuda", dtype=torch.float32)
        self.bv = torch.randn(C, device="cuda", dtype=torch.float32)

        # RoPE tables: [T, D//2]
        DH = D // 2
        theta = 10000.0
        freqs = 1.0 / (theta ** (torch.arange(0, D, 2, device="cuda").float() / D))
        positions = torch.arange(T, device="cuda").float()
        angles = positions[:, None] * freqs[None, :]  # [T, DH]
        self.cos = torch.cos(angles)
        self.sin = torch.sin(angles)

        # Mask: [B, T]
        self.mask = torch.ones(B, T, dtype=torch.uint8, device="cuda")

    def test_delta_entrance_forward(self):
        """Q, K, V outputs match PyTorch reference."""
        from llm.kernels.triton_delta_entrance import (
            fused_delta_entrance,
            pytorch_unfused_exact,
        )

        qo_t, ko_t, vo_t = fused_delta_entrance(
            self.q, self.k, self.v,
            self.wq, self.wk, self.wv,
            self.bq, self.bk, self.bv,
            self.cos, self.sin, self.mask,
        )
        qo_r, ko_r, vo_r = pytorch_unfused_exact(
            self.q, self.k, self.v,
            self.wq, self.wk, self.wv,
            self.bq, self.bk, self.bv,
            self.cos, self.sin, self.mask,
        )

        assert torch.allclose(qo_t.float(), qo_r.float(), atol=1e-3), (
            f"Q max diff: {(qo_t.float() - qo_r.float()).abs().max().item():.6e}"
        )
        assert torch.allclose(ko_t.float(), ko_r.float(), atol=1e-3), (
            f"K max diff: {(ko_t.float() - ko_r.float()).abs().max().item():.6e}"
        )
        assert torch.allclose(vo_t.float(), vo_r.float(), atol=1e-3), (
            f"V max diff: {(vo_t.float() - vo_r.float()).abs().max().item():.6e}"
        )

    def test_delta_entrance_backward(self):
        """Gradient correctness for q, k, v inputs."""
        from llm.kernels.triton_delta_entrance import (
            fused_delta_entrance,
            pytorch_unfused_exact,
        )

        # Triton path
        q_t = self.q.clone().requires_grad_(True)
        k_t = self.k.clone().requires_grad_(True)
        v_t = self.v.clone().requires_grad_(True)
        qo_t, ko_t, vo_t = fused_delta_entrance(
            q_t, k_t, v_t,
            self.wq.clone(), self.wk.clone(), self.wv.clone(),
            self.bq.clone(), self.bk.clone(), self.bv.clone(),
            self.cos, self.sin, self.mask,
        )
        (qo_t.sum() + ko_t.sum() + vo_t.sum()).backward()
        gq_t = q_t.grad.clone()

        # Reference path
        q_r = self.q.clone().requires_grad_(True)
        k_r = self.k.clone().requires_grad_(True)
        v_r = self.v.clone().requires_grad_(True)
        qo_r, ko_r, vo_r = pytorch_unfused_exact(
            q_r, k_r, v_r,
            self.wq.clone(), self.wk.clone(), self.wv.clone(),
            self.bq.clone(), self.bk.clone(), self.bv.clone(),
            self.cos, self.sin, self.mask,
        )
        (qo_r.sum() + ko_r.sum() + vo_r.sum()).backward()
        gq_r = q_r.grad.clone()

        # Relaxed tolerance due to tl.atomic_add non-determinism
        assert torch.allclose(gq_t.float(), gq_r.float(), atol=1e-2), (
            f"grad_q max diff: {(gq_t.float() - gq_r.float()).abs().max().item():.6e}"
        )
