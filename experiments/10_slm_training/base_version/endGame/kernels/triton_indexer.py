"""
Triton Gated Lightning Indexer Kernel
=====================================

Computes importance scores for the GSA (Gated Sparse Attention) indexer
using a shared-K design that avoids materializing per-head (B, H, T, T)
intermediates.

Key memory optimization:
    Per-head design:  Q=[B,T,H,d], K=[B,T,H,d] -> scores=[B,H,T,T]  (~6 GB at T=4096)
    Shared-K design:  Q=[B,T,H,d], K=[B,T,d]   -> scores=[B,T,T]    (~134 MB at T=4096)

The Triton kernel processes one query row at a time (like triton_sparse_attn),
using element-wise multiply + tl.sum for dot products to avoid tl.dot which
requires fp16 inputs on T4/Volta (sm_75).

Based on the GSA paper implementation (arXiv:2601.15305v1).
"""

import torch
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
    def _gated_indexer_fwd_kernel(
        # Pointers to matrices
        Q_ptr, K_ptr, W_ptr, B_ptr, OUT_ptr,
        # Matrix dimensions
        batch_size, seq_q, seq_kv, n_heads, d_idx,
        # Strides
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_wb, stride_wq, stride_wh,
        stride_ob, stride_oq, stride_ok,
        # Scale factor
        scale,
        # Query offset for chunked causal masking
        q_offset,
        # Causal mask flag
        use_causal: tl.constexpr,
        # Meta parameters
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Triton kernel for computing gated indexer scores.

        For each query position, computes:
            score[q, k] = sum_h( sigmoid(w[q, h]) * sigmoid(dot(q[q, h], k[k]) * scale + b[h]) )

        Processes one query row at a time (like triton_sparse_attn) to avoid
        tl.dot which requires fp16 on T4/Volta. Uses element-wise multiply +
        tl.sum for the dot product instead.

        Args:
            q_offset: Global offset of query positions within the full sequence.
                      Used for correct causal masking when processing query chunks.
                      Causal mask: (q_offset + local_q_pos) >= k_pos
        """
        # Grid: (batch_size, seq_q, cdiv(seq_kv, BLOCK_K))
        pid_b = tl.program_id(0)  # Batch index
        pid_q = tl.program_id(1)  # Query position (local, one row per program)
        pid_k = tl.program_id(2)  # Key block index

        # Key block offsets
        k_start = pid_k * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)
        d_offs = tl.arange(0, BLOCK_D)

        # Initialize accumulator for this query row: [BLOCK_K]
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # Local and global query positions
        q_local = pid_q
        q_global = q_offset + pid_q

        if q_local < seq_q:
            # Load key block (shared across heads): [BLOCK_K, BLOCK_D]
            # K shape: [batch, seq_kv, d_idx]
            k_ptrs = K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kk + d_offs[None, :] * stride_kd
            k_mask = (k_offs[:, None] < seq_kv) & (d_offs[None, :] < d_idx)
            k_val = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            # k_val: [BLOCK_K, BLOCK_D]

            # Loop over indexer heads
            for h in range(n_heads):
                # Load importance weight for this query position and head (scalar)
                w_ptr = W_ptr + pid_b * stride_wb + q_local * stride_wq + h * stride_wh
                w_val = tl.load(w_ptr).to(tl.float32)
                w_sigmoid = tl.sigmoid(w_val)

                # Load bias for this head (scalar)
                b = tl.load(B_ptr + h).to(tl.float32)

                # Load query vector for this head: [BLOCK_D]
                # Q shape: [batch, seq_q, n_heads, d_idx]
                q_ptrs = Q_ptr + pid_b * stride_qb + q_local * stride_qq + h * stride_qh + d_offs * stride_qd
                q_d_mask = d_offs < d_idx
                q_val = tl.load(q_ptrs, mask=q_d_mask, other=0.0).to(tl.float32)
                # q_val: [BLOCK_D]

                # Dot product: q_val[d] * k_val[k, d] summed over d -> [BLOCK_K]
                scores = tl.sum(q_val[None, :] * k_val, axis=1) * scale
                # scores: [BLOCK_K]

                # Apply sigmoid with bias
                gated = tl.sigmoid(scores + b)

                # Accumulate weighted score
                acc += w_sigmoid * gated

        # Apply causal mask if needed (using global query position)
        if use_causal:
            causal_mask = q_global >= k_offs
            acc = tl.where(causal_mask, acc, float('-inf'))

        # Store output row
        out_ptrs = OUT_ptr + pid_b * stride_ob + q_local * stride_oq + k_offs * stride_ok
        out_mask = (q_local < seq_q) & (k_offs < seq_kv)
        tl.store(out_ptrs, acc, mask=out_mask)


def triton_gated_indexer(
    q: torch.Tensor,   # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,   # [batch, seq_kv, d_idx]
    w: torch.Tensor,   # [batch, seq_q, n_heads]
    b: torch.Tensor,   # [n_heads]
    scale: float = 1.0,
    causal: bool = True,
    q_offset: int = 0,
) -> torch.Tensor:
    """
    Compute gated indexer scores using Triton kernel.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_idx]
        k: Key tensor [batch, seq_kv, d_idx]  (shared across heads)
        w: Importance weights [batch, seq_q, n_heads]
        b: Per-head bias [n_heads]
        scale: Scaling factor (typically 1/sqrt(d_idx))
        causal: Whether to apply causal masking
        q_offset: Global offset of query positions for chunked causal masking.
                  When processing a chunk of queries [q_start:q_end], pass
                  q_offset=q_start so causal mask uses global positions.

    Returns:
        scores: [batch, seq_q, seq_kv]
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_gated_indexer")

    orig_dtype = q.dtype
    batch_size, seq_q, n_heads, d_idx = q.shape
    _, seq_kv, _ = k.shape

    # Cast inputs to float32 — kernel does all compute in fp32.
    # Passing bf16 pointers causes type-inference failures in Triton's JIT
    # on some GPU/Triton version combos.
    q_f32 = q.float().contiguous()
    k_f32 = k.float().contiguous()
    w_f32 = w.float().contiguous()
    b_f32 = b.float().contiguous()

    # Allocate output in float32 (kernel stores fp32 accumulator)
    out = torch.empty(batch_size, seq_q, seq_kv, device=q.device, dtype=torch.float32)

    # Block sizes
    BLOCK_K = min(64, triton.next_power_of_2(seq_kv))
    BLOCK_D = triton.next_power_of_2(d_idx)

    # Grid: one program per (batch, query_row, key_block)
    grid = (batch_size, seq_q, triton.cdiv(seq_kv, BLOCK_K))

    try:
        _gated_indexer_fwd_kernel[grid](
            q_f32, k_f32, w_f32, b_f32, out,
            batch_size, seq_q, seq_kv, n_heads, d_idx,
            q_f32.stride(0), q_f32.stride(1), q_f32.stride(2), q_f32.stride(3),
            k_f32.stride(0), k_f32.stride(1), k_f32.stride(2),
            w_f32.stride(0), w_f32.stride(1), w_f32.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            scale,
            q_offset,
            causal,
            BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        )
        # Cast back to original dtype
        out = out.to(orig_dtype)
    except Exception as e:
        import warnings
        warnings.warn(f"Triton indexer kernel failed with: {e}. Falling back to PyTorch.")
        out = pytorch_gated_indexer(q, k, w, b, scale, causal, q_offset)

    return out


def pytorch_gated_indexer(
    q: torch.Tensor,   # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,   # [batch, seq_kv, d_idx]
    w: torch.Tensor,   # [batch, seq_q, n_heads]
    b: torch.Tensor,   # [n_heads]
    scale: float = 1.0,
    causal: bool = True,
    q_offset: int = 0,
) -> torch.Tensor:
    """
    PyTorch fallback for gated indexer computation.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_idx]
        k: Key tensor [batch, seq_kv, d_idx]  (shared across heads)
        w: Importance weights [batch, seq_q, n_heads]
        b: Per-head bias [n_heads]
        scale: Scaling factor
        causal: Whether to apply causal masking
        q_offset: Global offset of query positions for chunked causal masking

    Returns:
        scores: [batch, seq_q, seq_kv]
    """
    batch_size, seq_q, n_heads, d_idx = q.shape
    seq_kv = k.shape[1]

    # Compute QK scores per head: [batch, n_heads, seq_q, seq_kv]
    raw_scores = torch.einsum('bqhd,bkd->bhqk', q, k) * scale

    # Add bias: [n_heads, 1, 1]
    bias_expanded = b.view(1, -1, 1, 1)

    # Apply sigmoid activation
    gated_scores = torch.sigmoid(raw_scores + bias_expanded)

    # Weight by query-dependent importance: [batch, seq_q, n_heads] -> [batch, n_heads, seq_q, 1]
    w_sigmoid = torch.sigmoid(w).permute(0, 2, 1).unsqueeze(-1)

    # Weighted sum across heads
    weighted_scores = gated_scores * w_sigmoid
    final_scores = weighted_scores.sum(dim=1)  # [batch, seq_q, seq_kv]

    # Apply causal mask (using global query positions = q_offset + local)
    if causal:
        query_positions = q_offset + torch.arange(seq_q, device=q.device)
        key_positions = torch.arange(seq_kv, device=q.device)
        causal_invalid = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        final_scores = final_scores.masked_fill(causal_invalid.unsqueeze(0), float('-inf'))

    return final_scores
