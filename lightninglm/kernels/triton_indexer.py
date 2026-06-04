"""
Triton Gated Lightning Indexer Kernel (Optimized for L4)
========================================================

Optimized version: replaces per-head element-wise multiply + tl.sum loop
with a batched tl.dot across all heads using tensor cores.

Key change:
    OLD: for h in range(n_heads): scores = tl.sum(q * k, axis=1)  # 16 iterations
    NEW: all_scores = tl.dot(k_val, q_all)  # single tensor-core op [BLOCK_K, BLOCK_D] @ [BLOCK_D, N_HEADS]
"""

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def kernel_region(name: str):
        yield


if HAS_TRITON:

    @triton.jit
    def _gated_indexer_fwd_kernel(
        Q_ptr,
        K_ptr,
        W_ptr,
        B_ptr,
        OUT_ptr,
        batch_size,
        seq_q,
        seq_kv,
        n_heads,
        d_idx,
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
        scale,
        q_offset,
        use_causal: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        N_HEADS: tl.constexpr,
    ):
        """
        Optimized gated indexer using tl.dot for batched dot product across heads.

        Instead of looping over heads with element-wise multiply + tl.sum,
        loads all head queries as [BLOCK_D, N_HEADS] and computes a single
        tl.dot: [BLOCK_K, BLOCK_D] @ [BLOCK_D, N_HEADS] -> [BLOCK_K, N_HEADS]
        using tensor cores (bf16 inputs, fp32 accumulator).
        """
        pid_b = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_k = tl.program_id(2)

        k_start = pid_k * BLOCK_K
        k_offs = k_start + tl.arange(0, BLOCK_K)
        d_offs = tl.arange(0, BLOCK_D)
        h_offs = tl.arange(0, N_HEADS)

        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

        q_local = pid_q
        q_global = q_offset + pid_q

        if q_local < seq_q:
            # Load key block in native dtype (bf16) for tensor-core dot
            # K shape: [batch, seq_kv, d_idx]
            k_ptrs = (
                K_ptr
                + pid_b * stride_kb
                + k_offs[:, None] * stride_kk
                + d_offs[None, :] * stride_kd
            )
            k_mask = (k_offs[:, None] < seq_kv) & (d_offs[None, :] < d_idx)
            k_val = tl.load(k_ptrs, mask=k_mask, other=0.0)
            # k_val: [BLOCK_K, BLOCK_D] in bf16

            # Load ALL query heads at once as [BLOCK_D, N_HEADS] (transposed for tl.dot)
            # Q shape: [batch, seq_q, n_heads, d_idx]
            q_all_ptrs = (
                Q_ptr
                + pid_b * stride_qb
                + q_local * stride_qq
                + h_offs[None, :] * stride_qh
                + d_offs[:, None] * stride_qd
            )
            q_all_mask = (d_offs[:, None] < d_idx) & (h_offs[None, :] < n_heads)
            q_all = tl.load(q_all_ptrs, mask=q_all_mask, other=0.0)
            # q_all: [BLOCK_D, N_HEADS] in bf16

            # Batched dot product using tensor cores
            # [BLOCK_K, BLOCK_D] @ [BLOCK_D, N_HEADS] -> [BLOCK_K, N_HEADS] (fp32 accum)
            all_scores = tl.dot(k_val, q_all) * scale
            # all_scores: [BLOCK_K, N_HEADS] in fp32

            # Load all biases: [N_HEADS]
            b_all = tl.load(B_ptr + h_offs, mask=h_offs < n_heads, other=-100.0).to(
                tl.float32
            )

            # Load all importance weights: [N_HEADS]
            w_ptrs = (
                W_ptr + pid_b * stride_wb + q_local * stride_wq + h_offs * stride_wh
            )
            w_all = tl.load(w_ptrs, mask=h_offs < n_heads, other=-100.0).to(tl.float32)
            w_sigmoid_all = tl.sigmoid(w_all)  # [N_HEADS]

            # Apply sigmoid with bias: [BLOCK_K, N_HEADS]
            gated = tl.sigmoid(all_scores + b_all[None, :])

            # Weighted sum across heads: [BLOCK_K, N_HEADS] * [1, N_HEADS] -> sum -> [BLOCK_K]
            acc = tl.sum(gated * w_sigmoid_all[None, :], axis=1)

        # Apply causal mask
        if use_causal:
            causal_mask = q_global >= k_offs
            acc = tl.where(causal_mask, acc, float("-inf"))

        # Store output
        out_ptrs = (
            OUT_ptr + pid_b * stride_ob + q_local * stride_oq + k_offs * stride_ok
        )
        out_mask = (q_local < seq_q) & (k_offs < seq_kv)
        tl.store(out_ptrs, acc, mask=out_mask)


def triton_gated_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    causal: bool = True,
    q_offset: int = 0,
) -> torch.Tensor:
    with kernel_region("indexer_total"):
        if not HAS_TRITON:
            raise ImportError("Triton is required for triton_gated_indexer")

        orig_dtype = q.dtype
        batch_size, seq_q, n_heads, d_idx = q.shape
        _, seq_kv, _ = k.shape

        with kernel_region("indexer_contiguous"):
            q_in = q.contiguous()
            k_in = k.contiguous()
            w_in = w.contiguous()
            b_in = b.contiguous()

        with kernel_region("indexer_alloc"):
            out = torch.empty(
                batch_size, seq_q, seq_kv, device=q.device, dtype=torch.float32
            )

        BLOCK_K = min(128, triton.next_power_of_2(seq_kv))
        BLOCK_D = triton.next_power_of_2(d_idx)
        N_HEADS = max(16, triton.next_power_of_2(n_heads))

        grid = (batch_size, seq_q, triton.cdiv(seq_kv, BLOCK_K))

        try:
            with kernel_region("indexer_kernel"):
                _gated_indexer_fwd_kernel[grid](
                    q_in,
                    k_in,
                    w_in,
                    b_in,
                    out,
                    batch_size,
                    seq_q,
                    seq_kv,
                    n_heads,
                    d_idx,
                    q_in.stride(0),
                    q_in.stride(1),
                    q_in.stride(2),
                    q_in.stride(3),
                    k_in.stride(0),
                    k_in.stride(1),
                    k_in.stride(2),
                    w_in.stride(0),
                    w_in.stride(1),
                    w_in.stride(2),
                    out.stride(0),
                    out.stride(1),
                    out.stride(2),
                    scale,
                    q_offset,
                    causal,
                    BLOCK_K=BLOCK_K,
                    BLOCK_D=BLOCK_D,
                    N_HEADS=N_HEADS,
                    num_warps=2,
                )

            with kernel_region("indexer_convert"):
                out = out.to(orig_dtype)
        except Exception as e:
            strict = os.environ.get("REQUIRE_LONGCTX_KERNELS", "0") == "1"
            if strict:
                raise RuntimeError(
                    f"Triton indexer kernel failed in strict mode: {e}"
                ) from e
            import warnings

            warnings.warn(
                f"Triton indexer kernel failed with: {e}. Falling back to PyTorch."
            )
            with kernel_region("indexer_fallback"):
                out = pytorch_gated_indexer(q, k, w, b, scale, causal, q_offset)

        return out


def pytorch_gated_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    causal: bool = True,
    q_offset: int = 0,
) -> torch.Tensor:
    batch_size, seq_q, n_heads, d_idx = q.shape
    seq_kv = k.shape[1]
    raw_scores = torch.einsum("bqhd,bkd->bhqk", q, k) * scale
    bias_expanded = b.view(1, -1, 1, 1)
    gated_scores = torch.sigmoid(raw_scores + bias_expanded)
    w_sigmoid = torch.sigmoid(w).permute(0, 2, 1).unsqueeze(-1)
    weighted_scores = gated_scores * w_sigmoid
    final_scores = weighted_scores.sum(dim=1)
    if causal:
        query_positions = q_offset + torch.arange(seq_q, device=q.device)
        key_positions = torch.arange(seq_kv, device=q.device)
        causal_invalid = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        final_scores = final_scores.masked_fill(
            causal_invalid.unsqueeze(0), float("-inf")
        )
    return final_scores
