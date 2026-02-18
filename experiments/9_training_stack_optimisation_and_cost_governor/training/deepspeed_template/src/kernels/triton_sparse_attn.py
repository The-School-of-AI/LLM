"""
Triton Sparse Attention Kernel
==============================

Computes attention only over selected token indices (from the GSA indexer),
achieving O(L*k) complexity instead of O(L^2).

Each program instance handles one query row for one (batch, head) pair.
Online softmax is used to accumulate the output in a single pass over
the k_selected keys, keeping register pressure low.

Includes both a Triton JIT kernel and a PyTorch chunked fallback.
"""

import torch
import torch.nn.functional as F
from typing import Optional

# Check for Triton availability
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, IDX_ptr, MASK_ptr,
        OUT_ptr, LSE_ptr,
        batch_size,
        seq_q, seq_kv, n_heads, d_head, k_selected,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        stride_ob, stride_oq, stride_oh, stride_od,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Sparse attention forward kernel.  One program computes one query row
        for one (batch, head) pair — no BLOCK_Q tiling needed, which keeps
        register pressure low and avoids the inner qi loop.

        Works directly on (B, T, H, D) tensors via stride-based access —
        no contiguous copies required in the wrapper.

        Grid: (batch_size * n_heads, seq_q)
        """
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh  % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        # load query vector
        q_row_ptr = (Q_ptr
                     + pid_b * stride_qb
                     + pid_q * stride_qq
                     + pid_h * stride_qh)
        q_i = tl.load(q_row_ptr + d_offs * stride_qd,
                       mask=d_offs < d_head, other=0.0)

        # online softmax accumulators
        m_i = tl.full((1,), float('-inf'), dtype=tl.float32)
        l_i = tl.full((1,), 0.0,          dtype=tl.float32)
        acc = tl.zeros((BLOCK_D,),         dtype=tl.float32)

        # indices/mask: (B, H, T, k_sel) — access with pid_b + pid_h
        idx_row_ptr  = IDX_ptr  + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row_ptr = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs

            idx_load_mask = k_block_offs < k_selected
            qi_indices = tl.load(idx_row_ptr + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row_ptr + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            qi_mask = qi_mask_val > 0.5

            # gather K, V via indirect load
            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_load_mask = qi_mask[:, None] & (d_offs[None, :] < d_head)
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

            # dot-product scores
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            valid = idx_load_mask & qi_mask
            scores = tl.where(valid, scores, float('-inf'))

            # online softmax update
            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.exp(m_i - m_new)
            beta  = tl.exp(scores - m_new)

            l_i = alpha * l_i + tl.sum(beta, axis=0)
            acc = alpha * acc + tl.sum(beta[:, None] * v_vals, axis=0)
            m_i = m_new

        # normalise
        acc = acc / l_i

        # store output
        out_row_ptr = (OUT_ptr
                       + pid_b * stride_ob
                       + pid_q * stride_oq
                       + pid_h * stride_oh)
        tl.store(out_row_ptr + d_offs * stride_od, acc,
                 mask=d_offs < d_head)

        # store LSE  [batch, n_heads, seq_q]
        lse_ptr = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + pid_q
        tl.store(lse_ptr + tl.arange(0, 1), m_i + tl.log(l_i))


def triton_sparse_attention(
    q: torch.Tensor,        # [B, T, H, D]
    k: torch.Tensor,        # [B, T, H, D]
    v: torch.Tensor,        # [B, T, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,     # [B, H, T, k_sel]
    scale: float,
) -> torch.Tensor:
    """
    Triton sparse attention wrapper.

    Zero-copy: passes strides directly from (B, T, H, D) inputs to the kernel.
    No permute/contiguous/reshape needed — the kernel uses stride-based access.
    Only allocates the output tensor and a small LSE buffer.
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_sparse_attention")

    B, T, H, D = q.shape
    k_sel = indices.size(-1)

    # Indices must be int64 for Triton pointer arithmetic
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    # Mask must be float32 for comparison
    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)

    # Output in same layout as input: (B, T, H, D)
    out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

    BLOCK_K = triton.next_power_of_2(min(64, k_sel))
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * H, T)

    try:
        _sparse_attn_fwd_kernel[grid](
            q, k, v, indices, mask,
            out, lse,
            B, T, T, H, D, k_sel,
            # q/k/v strides: (B, T, H, D) layout
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            # indices strides: (B, H, T, k_sel)
            indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
            # mask strides: (B, H, T, k_sel)
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            # output strides: (B, T, H, D)
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        )
    except Exception as e:
        import warnings
        warnings.warn(f"Triton sparse-attn kernel failed ({e}); falling back to PyTorch.")
        return pytorch_sparse_attention(q, k, v, indices, mask, scale)

    # Already in (B, T, H, D) layout, cast back to input dtype
    return out.to(q.dtype)


def pytorch_sparse_attention(
    q: torch.Tensor,        # [B, T, H, D]
    k: torch.Tensor,        # [B, T, H, D]
    v: torch.Tensor,        # [B, T, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,     # [B, H, T, k_sel]
    scale: float,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Memory-efficient PyTorch sparse attention fallback.

    Gathers the k_sel keys/values per query using advanced indexing,
    then runs a small chunked softmax — O(T*k) instead of O(T^2).
    Fully differentiable (autograd-friendly).
    """
    B, T, H, _ = q.shape

    k_bh = k.permute(0, 2, 1, 3)  # (B, H, T_kv, D)
    v_bh = v.permute(0, 2, 1, 3)  # (B, H, T_kv, D)
    q_bh = q.permute(0, 2, 1, 3)  # (B, H, T,    D)

    output = torch.empty_like(q_bh)  # (B, H, T, D)

    bh_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
    h_idx  = torch.arange(H, device=q.device).view(1, H, 1, 1)

    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)

        idx_chunk  = indices[:, :, i:end, :]
        mask_chunk = mask[:, :, i:end, :]
        q_chunk    = q_bh[:, :, i:end, :]

        # Convert mask to bool for masked_fill (handles both bool and float inputs)
        bool_mask = mask_chunk > 0.5 if mask_chunk.dtype != torch.bool else mask_chunk

        k_gathered = k_bh[bh_idx, h_idx, idx_chunk]
        v_gathered = v_bh[bh_idx, h_idx, idx_chunk]

        scores = torch.einsum('bhqd,bhqkd->bhqk', q_chunk, k_gathered) * scale
        scores = scores.masked_fill(~bool_mask, float('-inf'))

        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_w = attn_w.masked_fill(~bool_mask, 0.0)

        out_chunk = torch.einsum('bhqk,bhqkd->bhqd', attn_w, v_gathered)
        output[:, :, i:end, :] = out_chunk

    # (B, H, T, D) -> (B, T, H, D)
    return output.permute(0, 2, 1, 3)
