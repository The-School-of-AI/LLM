"""
Triton kernel for Sparse Attention computation.

This kernel computes attention only over selected token indices,
achieving O(L*k) complexity instead of O(L^2).

Based on the GSA paper implementation (arXiv:2601.15305v1).
"""

import torch
from typing import Optional, Tuple

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
    def _sparse_attention_fwd_kernel(
        # Input pointers
        Q_ptr, K_ptr, V_ptr, IDX_ptr, MASK_ptr,
        # Output pointers
        OUT_ptr, LSE_ptr,
        # Dimensions
        batch_size, seq_q, seq_kv, n_heads, d_head, k_selected,
        # Strides for Q, K, V: [batch, seq, n_heads, d_head]
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        # Strides for indices and mask: [batch, seq_q, k_selected]
        stride_ib, stride_iq, stride_ik,
        stride_mb, stride_mq, stride_mk,
        # Strides for output: [batch, seq_q, n_heads, d_head]
        stride_ob, stride_oq, stride_oh, stride_od,
        # Scale factor
        scale,
        # Meta parameters
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Triton kernel for sparse attention forward pass.

        For each query, only attends to k_selected keys based on indices.
        """
        # Get program IDs
        pid_b = tl.program_id(0)  # Batch index
        pid_h = tl.program_id(1)  # Head index
        pid_q = tl.program_id(2)  # Query block index

        # Compute query start position
        q_start = pid_q * BLOCK_Q

        # Create offset arrays
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        # Load query block
        # Q: [batch, seq_q, n_heads, d_head]
        q_ptrs = Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qq + pid_h * stride_qh + d_offs[None, :] * stride_qd
        q_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_head)
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Initialize accumulators
        m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)  # Max scores
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)  # Sum of exp
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)  # Output accumulator

        # Loop over key blocks
        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs

            # Load indices for this block
            # IDX: [batch, seq_q, k_selected]
            idx_ptrs = IDX_ptr + pid_b * stride_ib + q_offs[:, None] * stride_iq + k_block_offs[None, :] * stride_ik
            idx_mask = (q_offs[:, None] < seq_q) & (k_block_offs[None, :] < k_selected)
            indices = tl.load(idx_ptrs, mask=idx_mask, other=0)

            # Load mask for this block
            # MASK: [batch, seq_q, k_selected]
            mask_ptrs = MASK_ptr + pid_b * stride_mb + q_offs[:, None] * stride_mq + k_block_offs[None, :] * stride_mk
            valid_mask = tl.load(mask_ptrs, mask=idx_mask, other=False)

            # Gather K and V using indices
            # K, V: [batch, seq_kv, n_heads, d_head]
            # We need to gather for each query position
            for qi in range(BLOCK_Q):
                if q_offs[qi] >= seq_q:
                    continue

                # Get indices for this query
                qi_indices = indices[qi, :]  # [BLOCK_K]
                qi_mask = valid_mask[qi, :]   # [BLOCK_K]

                # Load K values for selected indices
                k_vals = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
                v_vals = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

                for ki in range(BLOCK_K):
                    if k_block_offs[ki] >= k_selected:
                        continue
                    if not qi_mask[ki]:
                        continue

                    kv_idx = qi_indices[ki]

                    # Load K
                    k_ptr = K_ptr + pid_b * stride_kb + kv_idx * stride_kk + pid_h * stride_kh
                    k_val = tl.load(k_ptr + d_offs * stride_kd, mask=d_offs < d_head, other=0.0)
                    k_vals[ki, :] = k_val

                    # Load V
                    v_ptr = V_ptr + pid_b * stride_vb + kv_idx * stride_vk + pid_h * stride_vh
                    v_val = tl.load(v_ptr + d_offs * stride_vd, mask=d_offs < d_head, other=0.0)
                    v_vals[ki, :] = v_val

                # Compute attention scores for this query
                # q[qi]: [BLOCK_D], k_vals: [BLOCK_K, BLOCK_D]
                q_i = q[qi, :]  # [BLOCK_D]
                scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale  # [BLOCK_K]

                # Mask invalid positions
                scores = tl.where(qi_mask & (k_block_offs < k_selected), scores, float('-inf'))

                # Online softmax update
                m_i_new = tl.maximum(m_i[qi], tl.max(scores, axis=0))
                alpha = tl.exp(m_i[qi] - m_i_new)
                beta = tl.exp(scores - m_i_new)

                l_i_new = alpha * l_i[qi] + tl.sum(beta, axis=0)

                # Update accumulator
                acc[qi, :] = alpha * acc[qi, :] + tl.sum(beta[:, None] * v_vals, axis=0)

                m_i[qi] = m_i_new
                l_i[qi] = l_i_new

        # Normalize output
        acc = acc / l_i[:, None]

        # Store output
        out_ptrs = OUT_ptr + pid_b * stride_ob + q_offs[:, None] * stride_oq + pid_h * stride_oh + d_offs[None, :] * stride_od
        out_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_head)
        tl.store(out_ptrs, acc, mask=out_mask)

        # Store log-sum-exp for backward
        lse_ptrs = LSE_ptr + pid_b * seq_q * n_heads + pid_h * seq_q + q_offs
        lse_mask = q_offs < seq_q
        tl.store(lse_ptrs, m_i + tl.log(l_i), mask=lse_mask)


def triton_sparse_attention(
    q: torch.Tensor,         # [batch, seq_q, n_heads, d_head]
    k: torch.Tensor,         # [batch, seq_kv, n_heads, d_head]
    v: torch.Tensor,         # [batch, seq_kv, n_heads, d_head]
    indices: torch.Tensor,   # [batch, seq_q, k_selected]
    mask: torch.Tensor,      # [batch, seq_q, k_selected]
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sparse attention using Triton kernel.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_head]
        k: Key tensor [batch, seq_kv, n_heads, d_head]
        v: Value tensor [batch, seq_kv, n_heads, d_head]
        indices: Selected token indices [batch, seq_q, k_selected]
        mask: Valid selection mask [batch, seq_q, k_selected]
        scale: Attention scale factor

    Returns:
        output: [batch, seq_q, n_heads, d_head]
        lse: Log-sum-exp for backward [batch, n_heads, seq_q]
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_sparse_attention")

    batch_size, seq_q, n_heads, d_head = q.shape
    _, seq_kv, _, _ = k.shape
    _, _, k_selected = indices.shape

    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    # Allocate outputs
    out = torch.empty_like(q)
    lse = torch.empty(batch_size, n_heads, seq_q, device=q.device, dtype=torch.float32)

    # Block sizes
    BLOCK_Q = min(32, seq_q)
    BLOCK_K = min(32, k_selected)
    BLOCK_D = triton.next_power_of_2(d_head)

    # Grid
    grid = (batch_size, n_heads, triton.cdiv(seq_q, BLOCK_Q))

    # Ensure indices are int64 for Triton
    indices = indices.to(torch.int64)

    try:
        _sparse_attention_fwd_kernel[grid](
            q, k, v, indices, mask,
            out, lse,
            batch_size, seq_q, seq_kv, n_heads, d_head, k_selected,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            indices.stride(0), indices.stride(1), indices.stride(2),
            mask.stride(0), mask.stride(1), mask.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
        )
    except Exception as e:
        # Fall back to PyTorch implementation on kernel errors
        import warnings
        warnings.warn(f"Triton kernel failed with: {e}. Falling back to PyTorch.")
        out, lse = pytorch_sparse_attention(q, k, v, indices, mask, scale)

    return out, lse


def pytorch_sparse_attention(
    q: torch.Tensor,         # [batch, seq_q, n_heads, d_head]
    k: torch.Tensor,         # [batch, seq_kv, n_heads, d_head]
    v: torch.Tensor,         # [batch, seq_kv, n_heads, d_head]
    indices: torch.Tensor,   # [batch, seq_q, k_selected]
    mask: torch.Tensor,      # [batch, seq_q, k_selected]
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of sparse attention (memory-efficient version).

    This is the fallback when Triton is not available or fails.
    Uses efficient gather without expanding to full [seq_q, seq_kv] dimension.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_head]
        k: Key tensor [batch, seq_kv, n_heads, d_head]
        v: Value tensor [batch, seq_kv, n_heads, d_head]
        indices: Selected token indices [batch, seq_q, k_selected]
        mask: Valid selection mask [batch, seq_q, k_selected]
        scale: Attention scale factor

    Returns:
        output: [batch, seq_q, n_heads, d_head]
        lse: Log-sum-exp [batch, n_heads, seq_q]
    """
    import torch.nn.functional as F

    batch_size, seq_q, n_heads, d_head = q.shape
    _, seq_kv, _, _ = k.shape
    _, _, k_selected = indices.shape

    if scale is None:
        scale = 1.0 / (d_head ** 0.5)

    # Memory-efficient gather: avoid expanding to [batch, seq_q, seq_kv, ...]
    # Reshape K, V: [batch, seq_kv, n_heads * d_head]
    k_flat = k.view(batch_size, seq_kv, n_heads * d_head)
    v_flat = v.view(batch_size, seq_kv, n_heads * d_head)

    # Flatten indices: [batch, seq_q * k_selected]
    indices_flat = indices.view(batch_size, seq_q * k_selected)

    # Expand indices for gather: [batch, seq_q * k_selected, n_heads * d_head]
    indices_expanded = indices_flat.unsqueeze(-1).expand(-1, -1, n_heads * d_head)

    # Gather K and V: [batch, seq_q * k_selected, n_heads * d_head]
    k_gathered_flat = torch.gather(k_flat, dim=1, index=indices_expanded)
    v_gathered_flat = torch.gather(v_flat, dim=1, index=indices_expanded)

    # Reshape: [batch, seq_q, k_selected, n_heads, d_head]
    k_gathered = k_gathered_flat.view(batch_size, seq_q, k_selected, n_heads, d_head)
    v_gathered = v_gathered_flat.view(batch_size, seq_q, k_selected, n_heads, d_head)

    # Permute for attention: [batch, seq_q, n_heads, k_selected, d_head]
    k_gathered = k_gathered.permute(0, 1, 3, 2, 4)
    v_gathered = v_gathered.permute(0, 1, 3, 2, 4)

    # Compute attention scores: [batch, seq_q, n_heads, k_selected]
    scores = torch.einsum('bqhd,bqhkd->bqhk', q, k_gathered) * scale

    # Apply mask: [batch, seq_q, k_selected] -> [batch, seq_q, 1, k_selected]
    mask_expanded = mask.unsqueeze(2)
    scores = scores.masked_fill(~mask_expanded, float('-inf'))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    attn_weights = attn_weights.masked_fill(~mask_expanded, 0.0).to(q.dtype)

    # Weighted sum: [batch, seq_q, n_heads, d_head]
    out = torch.einsum('bqhk,bqhkd->bqhd', attn_weights, v_gathered)

    # Compute LSE for backward
    lse = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)  # [batch, n_heads, seq_q]

    return out, lse
