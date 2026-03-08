"""
Fused QKV projection: one read of x, one kernel for q = x@W_q^T, k = x@W_k^T, v = x@W_v^T.

Used in GatedSparseAttention to reduce kernel launches and memory traffic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# Block sizes for tiled matmul (default: A100 / high shared-memory GPUs)
BLOCK_M = 64
BLOCK_D = 256

# Small blocks for 64 KB shared-memory GPUs (e.g. T4 on Colab)
BLOCK_M_SMALL = 16
BLOCK_D_SMALL = 128

# Use small blocks when device limit is below what our large-block kernel needs.
# Large kernel can need ~1.7 MB (Triton tl.dot staging); A100 default = 163 KB. T4 = 64 KB.
SHARED_MEM_LIMIT_SMALL_BLOCKS = 1703936
# Env override: TRITON_USE_SMALL_BLOCKS=1 forces small blocks (use on Colab if auto-detect fails).
_USE_SMALL_BLOCKS_ENV = __import__("os").environ.get("TRITON_USE_SMALL_BLOCKS", "").strip().lower() in ("1", "true", "yes")


def _get_block_sizes(device: torch.device) -> tuple[int, int]:
    """Use small blocks when device shared-memory limit < ~1.7 MB (e.g. T4 64 KB, A100 default 163 KB) or env TRITON_USE_SMALL_BLOCKS=1."""
    if _USE_SMALL_BLOCKS_ENV:
        return BLOCK_M_SMALL, BLOCK_D_SMALL
    if not device.type == "cuda":
        return BLOCK_M, BLOCK_D
    try:
        idx = device.index if device.index is not None else 0
        props = torch.cuda.get_device_properties(idx)
        limit = (
            getattr(props, "max_shared_memory_per_block", 0)
            or getattr(props, "shared_memory_per_block", 0)
            or 0
        )
        if limit > 0 and limit < SHARED_MEM_LIMIT_SMALL_BLOCKS:
            return BLOCK_M_SMALL, BLOCK_D_SMALL
    except Exception:
        pass
    return BLOCK_M, BLOCK_D


if HAS_TRITON:
    @triton.jit
    def _fused_qkv_fwd_kernel(
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        W_q_ptr,
        W_k_ptr,
        W_v_ptr,
        N,
        D,
        stride_x_m,
        stride_x_d,
        stride_q_m,
        stride_q_d,
        stride_w_d,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per block of M rows. Computes q, k, v from x and W_q, W_k, W_v."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        mask_m = row_offs < N

        for d_out_start in range(0, D, BLOCK_D):
            d_out_offs = d_out_start + tl.arange(0, BLOCK_D)
            d_out_mask = d_out_offs < D

            q_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            k_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            v_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D

                x_ptrs = x_ptr + row_offs[:, None] * stride_x_m + d_offs[None, :] * stride_x_d
                x_block = tl.load(x_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0)

                # W_* are (D, D) with W[d_out, d]; we need block (d, d_out) for x_block @ W^T
                w_q_ptrs = W_q_ptr + d_out_offs[None, :] * stride_w_d + d_offs[:, None]
                w_q_block = tl.load(w_q_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                q_acc += tl.dot(x_block, w_q_block)

                w_k_ptrs = W_k_ptr + d_out_offs[None, :] * stride_w_d + d_offs[:, None]
                w_k_block = tl.load(w_k_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                k_acc += tl.dot(x_block, w_k_block)

                w_v_ptrs = W_v_ptr + d_out_offs[None, :] * stride_w_d + d_offs[:, None]
                w_v_block = tl.load(w_v_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                v_acc += tl.dot(x_block, w_v_block)

            q_ptrs = q_ptr + row_offs[:, None] * stride_q_m + d_out_offs[None, :] * stride_q_d
            tl.store(q_ptrs, q_acc.to(q_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])
            k_ptrs = k_ptr + row_offs[:, None] * stride_q_m + d_out_offs[None, :] * stride_q_d
            tl.store(k_ptrs, k_acc.to(k_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])
            v_ptrs = v_ptr + row_offs[:, None] * stride_q_m + d_out_offs[None, :] * stride_q_d
            tl.store(v_ptrs, v_acc.to(v_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])


def fused_qkv_proj_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused QKV: q = x @ W_q^T, k = x @ W_k^T, v = x @ W_v^T with one read of x.
    x: [N, D], W_*: [D, D]. Returns q, k, v each [N, D].
    """
    if not (HAS_TRITON and x.is_cuda):
        q = F.linear(x, W_q)
        k = F.linear(x, W_k)
        v = F.linear(x, W_v)
        return q, k, v

    N, D = x.shape
    assert W_q.shape == (D, D) and W_k.shape == (D, D) and W_v.shape == (D, D)
    block_m, block_d = _get_block_sizes(x.device)
    x = x.contiguous()
    q = torch.empty_like(x)
    k = torch.empty_like(x)
    v = torch.empty_like(x)
    grid = (triton.cdiv(N, block_m),)
    _fused_qkv_fwd_kernel[grid](
        x, q, k, v,
        W_q, W_k, W_v,
        N=N,
        D=D,
        stride_x_m=x.stride(0),
        stride_x_d=x.stride(1),
        stride_q_m=q.stride(0),
        stride_q_d=q.stride(1),
        stride_w_d=W_q.stride(0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    return q, k, v


def has_fused_qkv_proj() -> bool:
    return bool(HAS_TRITON)


# ── Fused QKVG (DeltaNet: q, k, v, g from one read of x) ─────────────────────

if HAS_TRITON:
    @triton.jit
    def _fused_qkvg_fwd_kernel(
        x_ptr,
        q_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        W_q_ptr,
        W_k_ptr,
        W_v_ptr,
        W_g_ptr,
        N,
        D_in,
        D_out,
        stride_x_m,
        stride_x_d,
        stride_out_m,
        stride_out_d,
        stride_w_in,
        stride_w_out,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per block of M rows. q,k,v,g = x @ W_*^T. W_* are [D_out, D_in]."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        mask_m = row_offs < N

        for d_out_start in range(0, D_out, BLOCK_D):
            d_out_offs = d_out_start + tl.arange(0, BLOCK_D)
            d_out_mask = d_out_offs < D_out

            q_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            k_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            v_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            g_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

            for d_start in range(0, D_in, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D_in

                x_ptrs = x_ptr + row_offs[:, None] * stride_x_m + d_offs[None, :] * stride_x_d
                x_block = tl.load(x_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0)

                # W_* are [D_out, D_in]; we need block (d_in, d_out) for x_block @ W^T -> load W[d_out, d_in]
                w_q_ptrs = W_q_ptr + d_out_offs[None, :] * stride_w_out + d_offs[:, None] * stride_w_in
                w_q_block = tl.load(w_q_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                q_acc += tl.dot(x_block, w_q_block)

                w_k_ptrs = W_k_ptr + d_out_offs[None, :] * stride_w_out + d_offs[:, None] * stride_w_in
                w_k_block = tl.load(w_k_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                k_acc += tl.dot(x_block, w_k_block)

                w_v_ptrs = W_v_ptr + d_out_offs[None, :] * stride_w_out + d_offs[:, None] * stride_w_in
                w_v_block = tl.load(w_v_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                v_acc += tl.dot(x_block, w_v_block)

                w_g_ptrs = W_g_ptr + d_out_offs[None, :] * stride_w_out + d_offs[:, None] * stride_w_in
                w_g_block = tl.load(w_g_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                g_acc += tl.dot(x_block, w_g_block)

            q_ptrs = q_ptr + row_offs[:, None] * stride_out_m + d_out_offs[None, :] * stride_out_d
            tl.store(q_ptrs, q_acc.to(q_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])
            k_ptrs = k_ptr + row_offs[:, None] * stride_out_m + d_out_offs[None, :] * stride_out_d
            tl.store(k_ptrs, k_acc.to(k_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])
            v_ptrs = v_ptr + row_offs[:, None] * stride_out_m + d_out_offs[None, :] * stride_out_d
            tl.store(v_ptrs, v_acc.to(v_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])
            g_ptrs = g_ptr + row_offs[:, None] * stride_out_m + d_out_offs[None, :] * stride_out_d
            tl.store(g_ptrs, g_acc.to(g_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])


def fused_qkvg_proj_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused QKVG: q,k,v,g = x @ W_*^T with one read of x.
    x: [N, D_in], W_*: [D_out, D_in]. Returns q, k, v, g each [N, D_out].
    """
    if not (HAS_TRITON and x.is_cuda):
        q = F.linear(x, W_q)
        k = F.linear(x, W_k)
        v = F.linear(x, W_v)
        g = F.linear(x, W_g)
        return q, k, v, g

    N, D_in = x.shape
    D_out = W_q.shape[0]
    assert W_q.shape == (D_out, D_in) and W_k.shape == (D_out, D_in)
    assert W_v.shape == (D_out, D_in) and W_g.shape == (D_out, D_in)
    block_m, block_d = _get_block_sizes(x.device)
    x = x.contiguous()
    q = torch.empty((N, D_out), dtype=x.dtype, device=x.device)
    k = torch.empty((N, D_out), dtype=x.dtype, device=x.device)
    v = torch.empty((N, D_out), dtype=x.dtype, device=x.device)
    g = torch.empty((N, D_out), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(N, block_m),)
    _fused_qkvg_fwd_kernel[grid](
        x, q, k, v, g,
        W_q, W_k, W_v, W_g,
        N=N,
        D_in=D_in,
        D_out=D_out,
        stride_x_m=x.stride(0),
        stride_x_d=x.stride(1),
        stride_out_m=q.stride(0),
        stride_out_d=q.stride(1),
        stride_w_in=W_q.stride(1),
        stride_w_out=W_q.stride(0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    return q, k, v, g


def has_fused_qkvg_proj() -> bool:
    return bool(HAS_TRITON)


# ── Fused O + output gate for GSA: o_proj(o_sparse * sigmoid(W_go(x))) ───────

if HAS_TRITON:
    @triton.jit
    def _fused_o_gate_fwd_kernel(
        x_ptr,
        o_sparse_ptr,
        out_ptr,
        W_go_ptr,
        W_o_ptr,
        N,
        D,
        stride_m,
        stride_d,
        stride_w_d,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """out = (o_sparse * sigmoid(x @ W_go^T)) @ W_o^T. Tile over output D."""
        pid = tl.program_id(0)
        row_start = pid * BLOCK_M
        row_offs = row_start + tl.arange(0, BLOCK_M)
        mask_m = row_offs < N

        for d_out_start in range(0, D, BLOCK_D):
            d_out_offs = d_out_start + tl.arange(0, BLOCK_D)
            d_out_mask = d_out_offs < D

            gate_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D
                x_ptrs = x_ptr + row_offs[:, None] * stride_m + d_offs[None, :] * stride_d
                x_block = tl.load(x_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0)
                # W_go is (D, D) with W_go[d_out, d]; we need block (d, d_out) for x_block @ w_block
                w_ptrs = W_go_ptr + d_out_offs[None, :] * stride_w_d + d_offs[:, None]
                w_block = tl.load(w_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                gate_acc += tl.dot(x_block, w_block)

            gate = tl.sigmoid(gate_acc)
            o_ptrs = o_sparse_ptr + row_offs[:, None] * stride_m + d_out_offs[None, :] * stride_d
            o_block = tl.load(o_ptrs, mask=mask_m[:, None] & d_out_mask[None, :], other=0.0)
            scaled = gate * o_block.to(tl.float32)

            out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            for d_start in range(0, D, BLOCK_D):
                d_offs = d_start + tl.arange(0, BLOCK_D)
                d_mask = d_offs < D
                # W_o is (D, D) with W_o[d_out, d]; we need block (d, d_out) for scaled @ w_o_block
                w_o_ptrs = W_o_ptr + d_out_offs[None, :] * stride_w_d + d_offs[:, None]
                w_o_block = tl.load(w_o_ptrs, mask=d_mask[:, None] & d_out_mask[None, :], other=0.0)
                out_acc += tl.dot(scaled, w_o_block)

            out_ptrs = out_ptr + row_offs[:, None] * stride_m + d_out_offs[None, :] * stride_d
            tl.store(out_ptrs, out_acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & d_out_mask[None, :])


def fused_o_gate_proj_forward(
    x: torch.Tensor,
    o_sparse: torch.Tensor,
    W_go: torch.Tensor,
    W_o: torch.Tensor,
) -> torch.Tensor:
    """out = o_proj(o_sparse * sigmoid(W_go(x))). x, o_sparse: [N, D]; W_*: [D, D]."""
    if not (HAS_TRITON and x.is_cuda):
        g_o = torch.sigmoid(F.linear(x, W_go))
        return F.linear(o_sparse * g_o, W_o)

    N, D = x.shape
    assert o_sparse.shape == (N, D) and W_go.shape == (D, D) and W_o.shape == (D, D)
    block_m, block_d = _get_block_sizes(x.device)
    x = x.contiguous()
    o_sparse = o_sparse.contiguous()
    out = torch.empty_like(o_sparse)
    grid = (triton.cdiv(N, block_m),)
    _fused_o_gate_fwd_kernel[grid](
        x, o_sparse, out,
        W_go, W_o,
        N=N,
        D=D,
        stride_m=x.stride(0),
        stride_d=x.stride(1),
        stride_w_d=W_go.stride(0),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
    )
    return out


def has_fused_o_gate_proj() -> bool:
    return bool(HAS_TRITON)