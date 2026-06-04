"""
Fused Weighted Scatter-Add for MoE output gathering.

Replaces: sorted_out.mul_(weights.unsqueeze(-1)) + routed_out.index_add_(...)
With: single fused kernel that does weighted scatter in one pass.

Called 8x per forward pass (once per MoE layer).
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# Forward kernel: routed_out[idx[i]] += sorted_out[i] * weights[i]
# Uses fp32 atomic add for correctness on SM 8.9
# ============================================================================


@triton.jit
def _weighted_scatter_add_kernel(
    Sorted_ptr,
    Weights_ptr,
    Indices_ptr,
    Out_ptr,
    M,  # number of sorted rows
    D: tl.constexpr,  # hidden dimension
    stride_sd,  # sorted_out stride(0)
    stride_od,  # out stride(0)
    BLOCK_D: tl.constexpr,
):
    """Each program handles one sorted row, scattering it to the output."""
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_m >= M:
        return

    # Load scalar weight and target index
    w = tl.load(Weights_ptr + pid_m).to(tl.float32)
    idx = tl.load(Indices_ptr + pid_m)

    # Load sorted_out row chunk
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    val = tl.load(Sorted_ptr + pid_m * stride_sd + offs_d, mask=mask, other=0.0).to(
        tl.float32
    )
    weighted_val = val * w

    # Atomic add to output row (multiple sorted rows may target same token)
    out_ptrs = Out_ptr + idx * stride_od + offs_d
    tl.atomic_add(out_ptrs, weighted_val, mask=mask)


# ============================================================================
# Alternative: non-atomic approach using output-side processing
# For each output token, gather its (at most top_k) contributions
# ============================================================================


@triton.jit
def _weighted_scatter_output_side_kernel(
    Sorted_ptr,
    Weights_ptr,
    SortedTokenIdx_ptr,
    Out_ptr,
    # Reverse mapping: for each output token, which sorted rows contribute
    RevStart_ptr,
    RevEnd_ptr,  # [N] — range of sorted indices for each output token
    N,  # number of output tokens
    D: tl.constexpr,
    stride_sd,
    stride_od,
    BLOCK_D: tl.constexpr,
):
    """Process output-side: for each output token, sum its contributions."""
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_n >= N:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs_d < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Each output token has at most top_k=2 contributions
    rev_start = tl.load(RevStart_ptr + pid_n)
    rev_end = tl.load(RevEnd_ptr + pid_n)

    for i in range(rev_start, rev_end):
        w = tl.load(Weights_ptr + i).to(tl.float32)
        val = tl.load(Sorted_ptr + i * stride_sd + offs_d, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += val * w

    tl.store(
        Out_ptr + pid_n * stride_od + offs_d,
        acc.to(Out_ptr.dtype.element_ty),
        mask=mask,
    )


# ============================================================================
# Python wrapper
# ============================================================================


def fused_weighted_scatter_add(sorted_out, weights, token_indices, N, use_atomic=False):
    """
    Fused weighted scatter-add.

    Args:
        sorted_out:    [M, D] — expert output, sorted by expert
        weights:       [M] — per-assignment weights (from gate, already normalized)
        token_indices: [M] — maps each sorted row to its output token position
        N:             int — number of output tokens (B*T)
        use_atomic:    bool — use atomic scatter (True) or output-side gather (False)

    Returns: [N, D] — accumulated weighted outputs
    """
    M, D = sorted_out.shape
    device = sorted_out.device
    dtype = sorted_out.dtype

    if use_atomic:
        # Atomic approach: process input-side, scatter with atomics
        out = torch.zeros(N, D, device=device, dtype=torch.float32)

        if M > 0:
            BLOCK_D = min(triton.next_power_of_2(D), 4096)
            grid = (M, triton.cdiv(D, BLOCK_D))
            _weighted_scatter_add_kernel[grid](
                sorted_out,
                weights.float(),
                token_indices,
                out,
                M,
                D,
                sorted_out.stride(0),
                out.stride(0),
                BLOCK_D=BLOCK_D,
            )

        return out.to(dtype)
    else:
        # Output-side approach: build reverse mapping, then gather
        # Build reverse mapping: for each output token, find its sorted row range
        # This requires token_indices to be sorted by token (they're sorted by expert!)
        # So we need an argsort by token_indices first
        sort_by_token = token_indices.argsort(stable=True)
        sorted_token_idx = token_indices[sort_by_token]

        # Find start/end for each output token
        rev_start = torch.zeros(N, dtype=torch.int64, device=device)
        rev_end = torch.zeros(N, dtype=torch.int64, device=device)

        if M > 0:
            # Use searchsorted for efficient range finding
            positions = torch.arange(N, device=device)
            rev_start = torch.searchsorted(sorted_token_idx, positions)
            rev_end = torch.searchsorted(sorted_token_idx, positions, right=True)

        out = torch.zeros(N, D, device=device, dtype=dtype)

        if M > 0:
            # Reorder sorted_out and weights to match token order
            reordered_sorted = sorted_out[sort_by_token].contiguous()
            reordered_weights = weights[sort_by_token].contiguous()

            BLOCK_D = min(triton.next_power_of_2(D), 4096)
            grid = (N, triton.cdiv(D, BLOCK_D))
            _weighted_scatter_output_side_kernel[grid](
                reordered_sorted,
                reordered_weights,
                sorted_token_idx,
                out,
                rev_start,
                rev_end,
                N,
                D,
                reordered_sorted.stride(0),
                out.stride(0),
                BLOCK_D=BLOCK_D,
            )

        return out


# ============================================================================
# Reference implementation
# ============================================================================


def pytorch_weighted_scatter(sorted_out, weights, token_indices, N):
    """Reference: mul + index_add."""
    D = sorted_out.shape[1]
    weighted = sorted_out * weights.unsqueeze(-1).to(sorted_out.dtype)
    out = torch.zeros(N, D, device=sorted_out.device, dtype=sorted_out.dtype)
    out.index_add_(0, token_indices, weighted)
    return out
