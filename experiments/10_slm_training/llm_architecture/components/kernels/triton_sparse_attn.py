"""
Triton kernel for Sparse Attention computation.

This kernel computes attention only over selected token indices,
achieving O(L*k) complexity instead of O(L^2).

Based on the GSA paper implementation (arXiv:2601.15305v1).
"""

from typing import Optional, Tuple

import torch

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
        Q_ptr,
        K_ptr,
        V_ptr,
        IDX_ptr,
        MASK_ptr,
        # Output pointers
        OUT_ptr,
        LSE_ptr,
        # Dimensions
        batch_size,
        seq_q,
        seq_kv,
        n_heads,
        d_head,
        k_selected,
        # Strides for Q, K, V: [batch, seq, n_heads, d_head]
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vh,
        stride_vd,
        # Strides for indices and mask: [batch, seq_q, k_selected]
        stride_ib,
        stride_iq,
        stride_ik,
        stride_mb,
        stride_mq,
        stride_mk,
        # Strides for output: [batch, seq_q, n_heads, d_head]
        stride_ob,
        stride_oq,
        stride_oh,
        stride_od,
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
        q_start + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        # Loop over queries in the block
        for qi in range(BLOCK_Q):
            # Calculate scalar offsets for this specific query row
            current_q_idx = pid_q * BLOCK_Q + qi

            # Check valid query row
            valid_q = current_q_idx < seq_q

            # Initialize accumulators for this query
            m_i = tl.full((1,), float("-inf"), dtype=tl.float32)  # Scalar max score
            l_i = tl.full((1,), 0.0, dtype=tl.float32)  # Scalar sum of exp
            acc = tl.zeros(
                (BLOCK_D,), dtype=tl.float32
            )  # Output accumulator for this query

            # Load Query Vector
            # Q ptr: [batch, seq, n_heads, d_head]
            q_row_ptr = (
                Q_ptr
                + pid_b * stride_qb
                + pid_h * stride_qh
                + current_q_idx * stride_qq
            )
            q_i = tl.load(
                q_row_ptr + d_offs * stride_qd,
                mask=(d_offs < d_head) & valid_q,
                other=0.0,
            )

            # Loop over key blocks
            for k_block in range(0, k_selected, BLOCK_K):
                k_block_offs = k_block + k_offs  # [BLOCK_K]

                # Offsets for indices row in IDX tensor
                # IDX: [batch, seq_q, k_selected]
                idx_row_ptr = IDX_ptr + pid_b * stride_ib + current_q_idx * stride_iq

                # Mask for indices load: valid k positions in the block AND valid query
                idx_load_mask = (k_block_offs < k_selected) & valid_q

                # Load indices for this query: [BLOCK_K]
                qi_indices = tl.load(
                    idx_row_ptr + k_block_offs * stride_ik, mask=idx_load_mask, other=0
                )

                # Load valid_mask for this query
                mask_row_ptr = MASK_ptr + pid_b * stride_mb + current_q_idx * stride_mq
                # Ensure qi_mask is loaded as boolean (or converts to it)
                qi_mask_val = tl.load(
                    mask_row_ptr + k_block_offs * stride_mk,
                    mask=idx_load_mask,
                    other=0.0,
                )
                qi_mask = (
                    qi_mask_val > 0.5
                )  # Assuming mask is float/int, convert to bool explicitly for safety

                # --- Vectorized Gather Logic (Indirect Load) ---

                # Calculate pointers for all K and V in this block simultaneously
                # K_ptr base for this batch/head
                k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
                v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

                # We need to broadcast qi_indices to [BLOCK_K, BLOCK_D] via stride_kk
                # and d_offs to [BLOCK_K, BLOCK_D] via stride_kd
                # qi_indices is [BLOCK_K]

                # Pointers for K: [BLOCK_K, BLOCK_D]
                k_ptrs = (
                    k_base
                    + qi_indices[:, None] * stride_kk
                    + d_offs[None, :] * stride_kd
                )
                v_ptrs = (
                    v_base
                    + qi_indices[:, None] * stride_vk
                    + d_offs[None, :] * stride_vd
                )

                # Mask for loading K/V
                # We interpret qi_mask as the validity of the index.
                # If idx_load_mask (k < k_selected) is False, qi_mask is False (from 'other=False/0').
                # We also assume indices < seq_kv (checked during index generation or safe via other=0 fallback).
                # Actually, if qi_mask is False, we don't care about the load result, but we must ensure pointers are safe.
                # Since qi_indices uses 'other=0', pointers point to index 0, which is valid.

                kv_load_mask = qi_mask[:, None] & (d_offs[None, :] < d_head)

                # Load K and V
                k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
                v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

                # --- Attention Computation ---

                # Compute attention scores for this query
                # Re-load query vector for this specific row from memory to avoid unsupported register slicing
                # Q ptr: [batch, seq, n_heads, d_head]
                # We need Q[pid_b, current_q_idx, pid_h, :]
                q_row_ptr = (
                    Q_ptr
                    + pid_b * stride_qb
                    + pid_h * stride_qh
                    + current_q_idx * stride_qq
                )
                q_i = tl.load(
                    q_row_ptr + d_offs * stride_qd, mask=d_offs < d_head, other=0.0
                )

                # Calculation: [1, D] * [K, D] -> [K] (sum over D)
                scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale  # [BLOCK_K]

                # Mask invalid positions
                # Scores corresponding to invalid keys or masked keys should be -inf
                valid_k_mask = (k_block_offs < k_selected) & qi_mask & valid_q
                scores = tl.where(valid_k_mask, scores, float("-inf"))
                # If valid_q is false, scores are all -inf (or ignored).
                # To avoid NaNs/issues, strict masking is good.

                # Combine all validity checks for the score mask
                # We need to reconstruct the element-wise mask since we didn't store it fully
                # Actually, qi_mask (loaded vector) contains the logic for (2) and (3) mostly?
                # No, qi_mask is just the GSA bool mask.

                valid_k_mask = (k_block_offs < k_selected) & qi_mask & valid_q
                scores = tl.where(valid_k_mask, scores, float("-inf"))

                # Online softmax update
                # If valid_q is false, scores are all -inf (or ignored).
                # To avoid NaNs/issues, strict masking is good.

                block_max = tl.max(
                    scores, axis=0
                )  # [BLOCK_K] -> scalar? No, axis=0 reduced K dim.
                # scores is [BLOCK_K]. max(axis=0) -> scalar.

                m_i_new = tl.maximum(m_i, block_max)
                alpha = tl.exp(m_i - m_i_new)
                beta = tl.exp(scores - m_i_new)

                l_i_new = alpha * l_i + tl.sum(beta, axis=0)

                # Update accumulator
                # acc is [BLOCK_D]
                # beta is [BLOCK_K]. v_vals is [BLOCK_K, BLOCK_D]
                # beta[:, None] -> [BLOCK_K, 1]
                # sum -> [BLOCK_D]
                acc = alpha * acc + tl.sum(beta[:, None] * v_vals, axis=0)

                m_i = m_i_new
                l_i = l_i_new

            # Normalize output (ensure valid shapes)
            # acc: [BLOCK_D]
            # l_i: [1]
            acc = acc / l_i

            # Store output
            # OUT: [batch, seq_q, n_heads, d_head]
            # We must ensure the pointer math produces a block of pointers [BLOCK_D]
            # out_row_ptr is scalar. d_offs is [BLOCK_D].
            out_row_ptr = (
                OUT_ptr
                + pid_b * stride_ob
                + current_q_idx * stride_oq
                + pid_h * stride_oh
            )

            # Explicitly recreate offsets to guarantee block type
            offs_d = tl.arange(0, BLOCK_D)
            out_ptrs = out_row_ptr + offs_d * stride_od

            # Store [BLOCK_D] value to [BLOCK_D] pointers
            tl.store(out_ptrs, acc, mask=(offs_d < d_head) & valid_q)

            # Store log-sum-exp for backward
            # LSE: [batch, n_heads, seq_q]
            lse_ptr_base = (
                LSE_ptr + pid_b * seq_q * n_heads + pid_h * seq_q + current_q_idx
            )
            # lse_ptr_base is scalar. m_i and l_i are [1] blocks.
            # We construct a block pointer of size [1] to match the value.
            lse_ptrs = lse_ptr_base + tl.arange(0, 1)
            tl.store(lse_ptrs, m_i + tl.log(l_i), mask=valid_q)


def triton_sparse_attention(
    q: torch.Tensor,  # [batch, seq_q, n_heads, d_head]
    k: torch.Tensor,  # [batch, seq_kv, n_heads, d_head]
    v: torch.Tensor,  # [batch, seq_kv, n_heads, d_head]
    indices: torch.Tensor,  # [batch, seq_q, k_selected]
    mask: torch.Tensor,  # [batch, seq_q, k_selected]
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
        scale = 1.0 / (d_head**0.5)

    # Allocate outputs
    out = torch.empty_like(q)
    lse = torch.empty(batch_size, n_heads, seq_q, device=q.device, dtype=torch.float32)

    # Block sizes
    BLOCK_Q = min(
        16, seq_q
    )  # Reduced block size to reduce register pressure/complexity
    BLOCK_K = min(32, k_selected)
    BLOCK_D = triton.next_power_of_2(d_head)

    # Grid
    grid = (batch_size, n_heads, triton.cdiv(seq_q, BLOCK_Q))

    # === Ensure tensor types and contiguity for Triton JIT ===
    # This prevents "value cannot be converted to type at::Ha" errors
    # that occur when Triton's JIT compiler has trouble with certain tensor types

    # Ensure contiguity for all tensors
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Ensure indices are int64 for Triton
    indices = indices.contiguous().to(torch.int64)

    # Convert boolean mask to float32 for Triton compatibility
    # Boolean tensors cause JIT compilation issues in some PyTorch/Triton versions
    if mask.dtype == torch.bool:
        mask = mask.to(torch.float32).contiguous()
    else:
        mask = mask.contiguous()

    try:
        _sparse_attention_fwd_kernel[grid](
            q,
            k,
            v,
            indices,
            mask,
            out,
            lse,
            # Dimensions
            batch_size,
            seq_q,
            seq_kv,
            n_heads,
            d_head,
            k_selected,
            # Strides for Q, K, V
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            # Strides for indices and mask
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            mask.stride(0),
            mask.stride(1),
            mask.stride(2),
            # Strides for output
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            scale,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
        )
    except Exception as e:
        # Fall back to PyTorch implementation on kernel errors
        import warnings

        warnings.warn(f"Triton kernel failed with: {e}. Falling back to PyTorch.")
        # Convert mask back to bool for PyTorch implementation
        mask_bool = mask > 0.5 if mask.dtype != torch.bool else mask
        out, lse = pytorch_sparse_attention(q, k, v, indices, mask_bool, scale)

    return out, lse


def pytorch_sparse_attention(
    q: torch.Tensor,  # [batch, seq_q, n_heads, d_head]
    k: torch.Tensor,  # [batch, seq_kv, n_heads, d_head]
    v: torch.Tensor,  # [batch, seq_kv, n_heads, d_head]
    indices: torch.Tensor,  # [batch, seq_q, k_selected]
    mask: torch.Tensor,  # [batch, seq_q, k_selected]
    scale: Optional[float] = None,
    chunk_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch implementation of sparse attention (memory-efficient version using chunking).
    Uses advanced indexing to avoid OOM from torch.gather expansion.
    """
    import torch.nn.functional as F

    batch_size, seq_q, n_heads, d_head = q.shape
    _, seq_kv, _, _ = k.shape
    _, _, k_selected = indices.shape

    if scale is None:
        scale = 1.0 / (d_head**0.5)

    output = torch.empty_like(q)
    lse_list = []

    # Prepare batch indices for advanced indexing: [batch, 1, 1]
    batch_idx = torch.arange(batch_size, device=q.device).view(batch_size, 1, 1)

    # Process in chunks
    for i in range(0, seq_q, chunk_size):
        end = min(i + chunk_size, seq_q)
        q_chunk = q[:, i:end]  # [B, chunk, H, D]
        indices_chunk = indices[:, i:end]  # [B, chunk, K_sel]
        mask_chunk = mask[:, i:end]  # [B, chunk, K_sel]

        # === 1. Gather K/V for this chunk using Advanced Indexing ===
        # Avoids expanding to [B, chunk, K_sel, H*D] which causes OOM.

        # indices_chunk: [B, chunk, K_sel] (values < seq_kv)
        # k: [B, seq_kv, H, D]
        # We want [B, chunk, K_sel, H, D]

        # k[batch_idx, indices_chunk] -> [B, chunk, K_sel, H, D]
        # This works because batch_idx broadcasts to [B, chunk, K_sel]
        # and we select the 'seq_kv' dimension using indices_chunk.
        # The remaining dims (H, D) are kept.

        k_gathered = k[batch_idx, indices_chunk]
        v_gathered = v[batch_idx, indices_chunk]

        # Permute for attention: [B, chunk, K_sel, H, D] -> [B, chunk, H, K_sel, D] (Wait, einsum prefers H first?)
        # Scores einsum: q[bqhd], k[bqhkd] -> bqhk.
        # Let's align k to [B, chunk, H, K_sel, D]
        k_gathered = k_gathered.transpose(2, 3)
        v_gathered = v_gathered.transpose(2, 3)

        # === 2. Compute Attention ===
        # Scores: [batch, chunk, n_heads, k_selected]
        # q_chunk: [B, chunk, H, D] -> [B, chunk, H, 1, D]
        scores = torch.einsum("bqhd,bqhkd->bqhk", q_chunk, k_gathered) * scale

        # Mask
        mask_expanded = mask_chunk.unsqueeze(2)  # [B, chunk, 1, K_sel]
        scores = scores.masked_fill(~mask_expanded, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.masked_fill(~mask_expanded, 0.0)

        # Output: [batch, chunk, n_heads, d_head]
        out_chunk = torch.einsum("bqhk,bqhkd->bqhd", attn_weights, v_gathered)
        output[:, i:end] = out_chunk

        # LSE: [batch, chunk, n_heads] -> [batch, n_heads, chunk]
        lse_chunk = torch.logsumexp(scores, dim=-1).permute(0, 2, 1)
        lse_list.append(lse_chunk)

    lse = torch.cat(lse_list, dim=2)

    return output, lse
