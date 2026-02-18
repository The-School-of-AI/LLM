"""
Streaming Indexer: Memory-Efficient Chunked Variance + TopK
============================================================

Avoids materializing the full (B, T, T) importance score tensor by
processing C query rows at a time in a SINGLE pass:

    For each chunk of C queries:
      1. Compute (B, C, T) importance scores
      2. Compute variance from this chunk
      3. Compute adaptive k_t from variance
      4. Apply torch.topk to select top indices
      5. Free the chunk

Memory comparison at T=256K, B=16:
    Current:   (B, T, T) = 4 TB  (impossible)
    Streaming: (B, C, T) = 64 MB per chunk + (B, T, k_limit) output

Based on the GSA paper implementation (arXiv:2601.15305v1).
"""

import torch
from typing import Tuple, Optional

# Import the base indexer kernels
from .triton_indexer import (
    HAS_TRITON,
    triton_gated_indexer,
    pytorch_gated_indexer,
)


# ============================================================
# Single-Pass Chunked Indexer: Variance + TopK in one loop
# ============================================================

def fused_indexer_topk(
    q: torch.Tensor,       # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,       # [batch, seq_kv, d_idx]
    w: torch.Tensor,       # [batch, seq_q, n_heads]
    b: torch.Tensor,       # [n_heads]
    scale: float,
    causal: bool = True,
    k_base: int = 512,
    k_min: int = 32,
    k_max: int = 4096,
    variance_ema: Optional[torch.Tensor] = None,
    is_training: bool = False,
    sink_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Memory-efficient fused indexer + topk for GSA.

    Single-pass chunked architecture: processes C query rows at a time,
    computing variance and topk from the same (B, C, T) score chunk.
    Never materializes the full (B, T, T) importance score tensor.

    Args:
        q: Query [batch, seq_q, n_heads, d_idx]
        k: Key [batch, seq_kv, d_idx] (shared across heads)
        w: Importance weights [batch, seq_q, n_heads]
        b: Per-head bias [n_heads]
        scale: Scaling factor (1/sqrt(d_idx))
        causal: Whether to apply causal masking
        k_base: Base sparsity level
        k_min: Minimum k per query
        k_max: Maximum k per query
        variance_ema: Running EMA of variance mean (scalar buffer, updated in-place)
        is_training: Whether this is the forward pass (for EMA update)
        sink_size: Number of initial tokens forced into every selection

    Returns:
        var_t: [batch, seq_q] — per-query variance (float32)
        k_t: [batch, seq_q] — per-query adaptive k (long)
        top_indices: [batch, seq_q, k_limit] — topk key indices (int32)
    """
    batch_size, seq_q, n_heads, d_idx = q.shape
    seq_kv = k.shape[1]
    device = q.device

    # ---- Phase 1: Compute variance per chunk (single pass) ----
    # We need variance first to compute k_t, then k_limit = max(k_t),
    # then topk. But computing topk in the same pass as variance requires
    # knowing k_limit upfront. Solution: two-phase single-pass.
    #
    # Phase 1: Chunked variance → var_t, k_t, k_limit
    # Phase 2: Chunked topk using same kernel, now knowing k_limit
    #
    # At small T (where C >= T), both phases are a single kernel call each.
    # At large T, the score computation is the bottleneck — but we must
    # know k_limit before we can allocate the output tensor.
    #
    # OPTIMIZATION: If k_max is small enough, use k_max as k_limit directly
    # to avoid the variance pass entirely when it would be a separate loop.
    # This trades slightly larger output tensor for halving compute.

    C = _auto_chunk_size(batch_size, seq_kv)
    use_triton = HAS_TRITON and q.is_cuda

    # Check if we can do true single-pass (use k_max as k_limit)
    # This avoids computing variance separately when k_max is reasonable.
    # At T=256K with k_max=4096: output = B*T*k_max*4 bytes
    # B=1: 4GB, B=16: 64GB — too much. Use dynamic k_limit.
    #
    # Heuristic: single-pass if output would be < 2GB, else two-phase.
    output_bytes_at_kmax = batch_size * seq_q * min(k_max, seq_kv) * 4
    single_pass = output_bytes_at_kmax < 2 * 1024 * 1024 * 1024  # 2GB

    if single_pass:
        # ---- TRUE SINGLE PASS: variance + topk from same chunk ----
        k_limit = min(seq_kv, max(k_max, sink_size))
        var_t = torch.empty(batch_size, seq_q, device=device, dtype=torch.float32)
        top_indices = torch.empty(batch_size, seq_q, k_limit, device=device, dtype=torch.int32)

        for q_start in range(0, seq_q, C):
            q_end = min(q_start + C, seq_q)
            q_chunk = q[:, q_start:q_end]
            w_chunk = w[:, q_start:q_end]

            # Compute (B, C_actual, seq_kv) importance scores
            scores = _compute_scores(q_chunk, k, w_chunk, b, scale, causal, q_start, use_triton)

            # Variance from this chunk (replace -inf with 0)
            scores_for_var = scores.masked_fill(scores == float('-inf'), 0.0)
            var_t[:, q_start:q_end] = scores_for_var.var(dim=-1, unbiased=False)
            del scores_for_var

            # Attention sinks
            if seq_kv > sink_size:
                scores = scores.clone()
                scores[:, :, :sink_size] = float('inf')

            # TopK
            _, chunk_idx = scores.topk(k_limit, dim=-1)
            top_indices[:, q_start:q_end, :] = chunk_idx.to(torch.int32)

            del scores, chunk_idx

        # Compute adaptive k_t from variance
        if variance_ema is not None:
            avg_V = variance_ema.clamp(min=1e-6)
        else:
            avg_V = var_t.mean().clamp(min=1e-6)
        k_t = (k_base * var_t / avg_V).floor().clamp(min=k_min, max=k_max).long()

        # Update EMA
        if is_training and variance_ema is not None:
            variance_ema.mul_(0.99).add_(var_t.mean().detach(), alpha=0.01)

    else:
        # ---- TWO-PHASE: variance first to get dynamic k_limit ----

        # Phase 1: Chunked variance
        var_t = torch.empty(batch_size, seq_q, device=device, dtype=torch.float32)
        for q_start in range(0, seq_q, C):
            q_end = min(q_start + C, seq_q)
            scores = _compute_scores(
                q[:, q_start:q_end], k, w[:, q_start:q_end],
                b, scale, causal, q_start, use_triton)
            scores_for_var = scores.masked_fill(scores == float('-inf'), 0.0)
            var_t[:, q_start:q_end] = scores_for_var.var(dim=-1, unbiased=False)
            del scores, scores_for_var

        # Compute adaptive k_t
        if variance_ema is not None:
            avg_V = variance_ema.clamp(min=1e-6)
        else:
            avg_V = var_t.mean().clamp(min=1e-6)
        k_t = (k_base * var_t / avg_V).floor().clamp(min=k_min, max=k_max).long()

        # Update EMA
        if is_training and variance_ema is not None:
            variance_ema.mul_(0.99).add_(var_t.mean().detach(), alpha=0.01)

        # Dynamic k_limit
        k_limit = min(seq_kv, max(int(k_t.max().item()), sink_size, 1))

        # Phase 2: Chunked topk
        top_indices = torch.empty(batch_size, seq_q, k_limit, device=device, dtype=torch.int32)
        for q_start in range(0, seq_q, C):
            q_end = min(q_start + C, seq_q)
            scores = _compute_scores(
                q[:, q_start:q_end], k, w[:, q_start:q_end],
                b, scale, causal, q_start, use_triton)

            if seq_kv > sink_size:
                scores = scores.clone()
                scores[:, :, :sink_size] = float('inf')

            _, chunk_idx = scores.topk(k_limit, dim=-1)
            top_indices[:, q_start:q_end, :] = chunk_idx.to(torch.int32)
            del scores, chunk_idx

    return var_t, k_t, top_indices


def _compute_scores(q_chunk, k, w_chunk, b, scale, causal, q_offset, use_triton):
    """Compute importance scores for a chunk of queries."""
    if use_triton:
        return triton_gated_indexer(q_chunk, k, w_chunk, b, scale, causal, q_offset)
    else:
        return pytorch_gated_indexer(q_chunk, k, w_chunk, b, scale, causal, q_offset)


# Keep streaming_indexer_variance for backward compat / standalone use
def streaming_indexer_variance(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """
    Compute per-query variance of importance scores without materializing
    the full (B, T, T) score matrix. Chunked PyTorch implementation.

    Args:
        q: [batch, seq_q, n_heads, d_idx]
        k: [batch, seq_kv, d_idx]
        w: [batch, seq_q, n_heads]
        b: [n_heads]
        scale: Scaling factor
        causal: Whether to apply causal masking

    Returns:
        var_t: [batch, seq_q] — population variance per query row
    """
    batch_size, seq_q, n_heads, d_idx = q.shape
    seq_kv = k.shape[1]
    device = q.device
    use_triton = HAS_TRITON and q.is_cuda

    var_out = torch.empty(batch_size, seq_q, device=device, dtype=torch.float32)
    C = _auto_chunk_size(batch_size, seq_kv)

    for q_start in range(0, seq_q, C):
        q_end = min(q_start + C, seq_q)
        scores = _compute_scores(
            q[:, q_start:q_end], k, w[:, q_start:q_end],
            b, scale, causal, q_start, use_triton)
        scores_for_var = scores.masked_fill(scores == float('-inf'), 0.0)
        var_out[:, q_start:q_end] = scores_for_var.var(dim=-1, unbiased=False)
        del scores, scores_for_var

    return var_out


# ============================================================
# Utilities
# ============================================================

def _auto_chunk_size(batch_size: int, seq_kv: int,
                     target_bytes: int = 512 * 1024 * 1024) -> int:
    """
    Auto-select chunk size C so that (B, C, seq_kv) x 4 bytes < target_bytes.

    Returns C rounded down to the nearest power of 2 for kernel efficiency,
    capped at seq_kv (no chunking needed if C >= seq_kv).
    """
    bytes_per_row = batch_size * seq_kv * 4  # float32
    if bytes_per_row == 0:
        return 1
    max_C = target_bytes // bytes_per_row
    # Round down to power of 2
    C = 1
    while C * 2 <= max_C:
        C *= 2
    return max(1, min(C, seq_kv))
