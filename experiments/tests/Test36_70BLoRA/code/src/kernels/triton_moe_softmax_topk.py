"""
K5: Fused Softmax-TopK-Index for MoE Gate.

Fuses the router's topk → softmax → null-mask → renormalize into a single Triton kernel.
Replaces 4-5 separate kernel launches that are launch-overhead dominated.

Algorithm:
    1. Load full row of logits [S] (S=520 total slots) into SRAM
    2. Iterative argmax to find top-k (k=8): for each of k iterations,
       find max, record index+value, mask with -inf
    3. Softmax over k selected values (numerically stable: max-subtract)
    4. Null masking: zero weights where index >= num_real_experts
    5. Renormalize: divide by sum of real weights (clamp min 1e-6)

Reference: woct0rdho/transformers-qwen3-moe-fused, vLLM topk_softmax_kernels.cu
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_softmax_topk_kernel(
    Logits_ptr,       # [N_rows, S] input logits
    Out_idx_ptr,      # [N_rows, K] output: topk indices
    Out_weight_ptr,   # [N_rows, K] output: renormalized weights
    stride_logit_row,
    stride_idx_row,
    stride_weight_row,
    S: tl.constexpr,            # total number of slots (e.g. 520)
    K: tl.constexpr,            # top-k (e.g. 8)
    num_real_experts: tl.constexpr,  # null threshold
    BLOCK_S: tl.constexpr,      # >= S, power of 2 (e.g. 1024)
):
    row_id = tl.program_id(0)

    # Load full row of logits into SRAM
    logit_base = Logits_ptr + row_id * stride_logit_row
    offs = tl.arange(0, BLOCK_S)
    mask = offs < S
    logits = tl.load(logit_base + offs, mask=mask, other=float("-inf")).to(tl.float32)

    # --- Iterative argmax for top-k ---
    # We accumulate topk values and indices in separate registers.
    # Since K is a constexpr, Triton can unroll this loop.
    # We store results in a [BLOCK_K] register block where BLOCK_K >= K.

    # Use tl.arange to create index arrays for storing topk results
    topk_vals = tl.full([K], float("-inf"), dtype=tl.float32)
    topk_idxs = tl.full([K], 0, dtype=tl.int32)

    for i in tl.static_range(K):
        # Find max value and its index
        max_val = tl.max(logits, axis=0)
        # Create mask for the max position(s) — take first occurrence
        is_max = (logits == max_val) & mask
        # Get index of max — argmax over the block
        idx_range = tl.arange(0, BLOCK_S)
        # Set non-max positions to a large index so min gives us the first max
        candidate_idx = tl.where(is_max, idx_range, BLOCK_S)
        max_idx = tl.min(candidate_idx, axis=0)

        # Store into topk arrays using i as offset
        topk_vals = tl.where(tl.arange(0, K) == i, max_val, topk_vals)
        topk_idxs = tl.where(tl.arange(0, K) == i, max_idx.to(tl.int32), topk_idxs)

        # Mask out this position for next iteration
        logits = tl.where(idx_range == max_idx, float("-inf"), logits)

    # --- Softmax over K selected values ---
    max_topk = tl.max(topk_vals, axis=0)
    exp_vals = tl.exp(topk_vals - max_topk)
    sum_exp = tl.sum(exp_vals, axis=0)
    weights = exp_vals / sum_exp

    # --- Null masking ---
    k_offs = tl.arange(0, K)
    is_null = topk_idxs >= num_real_experts
    weights = tl.where(is_null, 0.0, weights)

    # --- Renormalize ---
    real_sum = tl.sum(weights, axis=0)
    real_sum = tl.where(real_sum < 1e-6, 1e-6, real_sum)
    weights = weights / real_sum

    # --- Store results ---
    idx_base = Out_idx_ptr + row_id * stride_idx_row
    weight_base = Out_weight_ptr + row_id * stride_weight_row

    tl.store(idx_base + k_offs, topk_idxs, mask=k_offs < K)
    tl.store(weight_base + k_offs, weights.to(Out_weight_ptr.dtype.element_ty), mask=k_offs < K)


def fused_softmax_topk(
    logits: torch.Tensor,
    k: int,
    num_real_experts: int,
) -> tuple:
    """
    Fused softmax-topk-null_mask-renormalize for MoE gate.

    Equivalent to:
        topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
        topk_weight = F.softmax(topk_logits, dim=-1)
        is_null = topk_idx >= num_real_experts
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

    Args:
        logits: [*, S] — router logits (flattened to 2D internally)
        k: int — number of experts to select
        num_real_experts: int — indices >= this are null slots

    Returns:
        topk_idx: [*, K] int64 — selected expert indices
        topk_weight: [*, K] float — renormalized weights (null slots = 0, renormalized)
        is_null: [*, K] bool — True where topk_idx >= num_real_experts
    """
    orig_shape = logits.shape
    S = orig_shape[-1]
    logits_2d = logits.reshape(-1, S).contiguous()
    N_rows = logits_2d.shape[0]

    # Outputs
    topk_idx = torch.empty(N_rows, k, device=logits.device, dtype=torch.int32)
    topk_weight = torch.empty(N_rows, k, device=logits.device, dtype=logits.dtype)

    # Block size: next power of 2 >= S
    BLOCK_S = triton.next_power_of_2(S)

    _fused_softmax_topk_kernel[(N_rows,)](
        logits_2d,
        topk_idx,
        topk_weight,
        logits_2d.stride(0),
        topk_idx.stride(0),
        topk_weight.stride(0),
        S=S,
        K=k,
        num_real_experts=num_real_experts,
        BLOCK_S=BLOCK_S,
    )

    # Reshape to match input shape
    out_shape = orig_shape[:-1] + (k,)
    topk_idx = topk_idx.to(torch.int64).reshape(out_shape)
    topk_weight = topk_weight.reshape(out_shape)
    is_null = topk_idx >= num_real_experts

    return topk_idx, topk_weight, is_null


def pytorch_softmax_topk(logits, k, num_real_experts):
    """Reference implementation for correctness testing."""
    topk_logits, topk_idx = torch.topk(logits, k, dim=-1)
    topk_weight = torch.nn.functional.softmax(topk_logits, dim=-1)
    is_null = topk_idx >= num_real_experts
    real_weights = topk_weight * (~is_null).float()
    weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    topk_weight = real_weights / weight_sum
    return topk_idx, topk_weight, is_null
