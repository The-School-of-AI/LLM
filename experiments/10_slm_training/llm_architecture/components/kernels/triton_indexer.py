"""
Triton kernel for Gated Lightning Indexer.

This kernel computes indexer scores efficiently on GPU using Triton.
Based on the GSA paper implementation (arXiv:2601.15305v1).
"""

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
    def _sigmoid(x):
        """Manual sigmoid: 1/(1+exp(-x)), universally supported in Triton."""
        return 1.0 / (1.0 + tl.exp(-x))

    @triton.jit
    def _gated_indexer_fwd_kernel(
        # Pointers to matrices
        Q_ptr,
        K_ptr,
        W_ptr,
        B_ptr,
        OUT_ptr,
        # Matrix dimensions
        batch_size,
        seq_q,
        seq_kv,
        n_heads,
        d_idx,
        # Strides
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_wb,
        stride_wq,
        stride_wh,
        stride_ob,
        stride_oq,
        stride_ok,
        # Scale factor
        scale,
        # Causal mask flag
        use_causal: tl.constexpr,
        # Meta parameters
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Triton kernel for computing gated indexer scores.

        For each query position, computes:
            score[q, k] = sum_h( sigmoid(w[q, h]) * sigmoid(dot(q[q, h], k[k]) * scale + b[h]) )
        """
        # Get program IDs
        pid_b = tl.program_id(0)  # Batch index
        pid_q = tl.program_id(1)  # Query block index
        pid_k = tl.program_id(2)  # Key block index

        # Compute block start positions
        q_start = pid_q * BLOCK_Q
        k_start = pid_k * BLOCK_K

        # Create offset arrays
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        k_offs = k_start + tl.arange(0, BLOCK_K)
        d_offs = tl.arange(0, BLOCK_D)

        # Initialize output accumulator
        acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

        # Loop over indexer heads
        for h in range(n_heads):
            # Load query block for this head
            # Q shape: [batch, seq_q, n_heads, d_idx]
            q_ptrs = (
                Q_ptr
                + pid_b * stride_qb
                + q_offs[:, None] * stride_qq
                + h * stride_qh
                + d_offs[None, :] * stride_qd
            )
            q_mask = (q_offs[:, None] < seq_q) & (d_offs[None, :] < d_idx)
            q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

            # Load key block (shared across heads)
            # K shape: [batch, seq_kv, d_idx]
            k_ptrs = (
                K_ptr
                + pid_b * stride_kb
                + k_offs[:, None] * stride_kk
                + d_offs[None, :] * stride_kd
            )
            k_mask = (k_offs[:, None] < seq_kv) & (d_offs[None, :] < d_idx)
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Load importance weights
            # W shape: [batch, seq_q, n_heads]
            w_ptrs = W_ptr + pid_b * stride_wb + q_offs * stride_wq + h * stride_wh
            w_mask = q_offs < seq_q
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            w_sigmoid = _sigmoid(w)

            # Load bias for this head
            b = tl.load(B_ptr + h).to(tl.float32)

            # Compute dot product: [BLOCK_Q, BLOCK_K]
            dot = tl.dot(q, tl.trans(k)) * scale

            # Apply sigmoid activation with bias
            gated = _sigmoid(dot + b)

            # Accumulate weighted scores
            acc += w_sigmoid[:, None] * gated

        # Apply causal mask if needed
        if use_causal:
            causal_mask = q_offs[:, None] >= k_offs[None, :]
            acc = tl.where(causal_mask, acc, float("-inf"))

        # Store output
        out_ptrs = (
            OUT_ptr
            + pid_b * stride_ob
            + q_offs[:, None] * stride_oq
            + k_offs[None, :] * stride_ok
        )
        out_mask = (q_offs[:, None] < seq_q) & (k_offs[None, :] < seq_kv)
        tl.store(out_ptrs, acc, mask=out_mask)


# Track whether we've warned about kernel failure
_warned_indexer_failure = False


def triton_gated_indexer(
    q: torch.Tensor,  # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,  # [batch, seq_kv, d_idx]
    w: torch.Tensor,  # [batch, seq_q, n_heads]
    b: torch.Tensor,  # [n_heads]
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute gated indexer scores using Triton kernel.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_idx]
        k: Key tensor [batch, seq_kv, d_idx]
        w: Importance weights [batch, seq_q, n_heads]
        b: Per-head bias [n_heads]
        scale: Scaling factor (typically 1/sqrt(d_idx))
        causal: Whether to apply causal masking

    Returns:
        scores: [batch, seq_q, seq_kv]
    """
    global _warned_indexer_failure

    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_gated_indexer")

    batch_size, seq_q, n_heads, d_idx = q.shape
    _, seq_kv, _ = k.shape

    # Ensure contiguous for correct strides
    q = q.contiguous()
    k = k.contiguous()
    w = w.contiguous()

    # Allocate output
    out = torch.empty(batch_size, seq_q, seq_kv, device=q.device, dtype=torch.float32)

    # Block sizes — must be powers of 2 for tl.dot
    BLOCK_Q = min(64, triton.next_power_of_2(seq_q))
    BLOCK_K = min(64, triton.next_power_of_2(seq_kv))
    BLOCK_D = triton.next_power_of_2(d_idx)

    # Grid
    grid = (batch_size, triton.cdiv(seq_q, BLOCK_Q), triton.cdiv(seq_kv, BLOCK_K))

    try:
        # Launch kernel
        _gated_indexer_fwd_kernel[grid](
            q,
            k,
            w,
            b,
            out,
            batch_size,
            seq_q,
            seq_kv,
            n_heads,
            d_idx,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            w.stride(0),
            w.stride(1),
            w.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            scale,
            causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
        )
    except Exception as e:
        # Fall back to PyTorch implementation (warn once only)
        if not _warned_indexer_failure:
            import warnings

            warnings.warn(
                f"Triton indexer kernel failed: {e}. Falling back to PyTorch."
            )
            _warned_indexer_failure = True
        out = pytorch_gated_indexer(q, k, w, b, scale, causal)

    return out


def pytorch_gated_indexer(
    q: torch.Tensor,  # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,  # [batch, seq_kv, d_idx]
    w: torch.Tensor,  # [batch, seq_q, n_heads]
    b: torch.Tensor,  # [n_heads]
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """
    PyTorch fallback for gated indexer computation.

    Args:
        q: Query tensor [batch, seq_q, n_heads, d_idx]
        k: Key tensor [batch, seq_kv, d_idx]
        w: Importance weights [batch, seq_q, n_heads]
        b: Per-head bias [n_heads]
        scale: Scaling factor
        causal: Whether to apply causal masking

    Returns:
        scores: [batch, seq_q, seq_kv]
    """
    batch_size, seq_q, n_heads, d_idx = q.shape
    seq_kv = k.shape[1]

    # Compute QK scores per head: [batch, n_heads, seq_q, seq_kv]
    raw_scores = torch.einsum("bqhd,bkd->bhqk", q, k) * scale

    # Add bias: [n_heads, 1, 1]
    bias_expanded = b.view(1, -1, 1, 1)

    # Apply sigmoid activation
    gated_scores = torch.sigmoid(raw_scores + bias_expanded)

    # Weight by query-dependent importance: [batch, seq_q, n_heads] -> [batch, n_heads, seq_q, 1]
    w_sigmoid = torch.sigmoid(w).permute(0, 2, 1).unsqueeze(-1)

    # Weighted sum across heads
    weighted_scores = gated_scores * w_sigmoid
    final_scores = weighted_scores.sum(dim=1)  # [batch, seq_q, seq_kv]

    # Apply causal mask
    if causal:
        query_positions = torch.arange(seq_q, device=q.device)
        key_positions = torch.arange(seq_kv, device=q.device)
        causal_invalid = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        final_scores = final_scores.masked_fill(
            causal_invalid.unsqueeze(0), float("-inf")
        )

    return final_scores
