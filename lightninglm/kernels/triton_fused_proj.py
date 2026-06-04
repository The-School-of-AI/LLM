"""
Fused Multi-Head Linear Projection kernels.

Replaces multiple nn.Linear calls on the same input with a single GEMM
by concatenating weight matrices. Variants:
  - fused_multi_proj: generic N-way split projection
  - fused_multi_proj_sigmoid: same but applies sigmoid to specified output splits

Simple approach: use F.linear on concatenated weights, then split.
Custom Triton GEMM with fused epilogue for beta_gk variant (sigmoid + bias).
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================================
# Simple fused projection (concat weights + F.linear + split)
# ============================================================================


class FusedMultiProjFn(torch.autograd.Function):
    """Fused multi-projection: single GEMM then split. Supports backward."""

    @staticmethod
    def forward(ctx, x, weight, splits, bias=None, sigmoid_mask=None):
        """
        Args:
            x: [*, D_in] input tensor
            weight: [D_out_total, D_in] concatenated weight matrix
            splits: tuple of output sizes (must sum to D_out_total)
            bias: optional [D_out_total] bias
            sigmoid_mask: optional tuple of bools, True = apply sigmoid to that split
        Returns:
            tuple of output tensors, each [*, split_size]
        """
        # Single GEMM
        y = F.linear(x, weight, bias)  # [*, D_out_total]

        # Apply sigmoid to specified splits if needed
        if sigmoid_mask is not None:
            outputs = y.split(list(splits), dim=-1)
            results = []
            for out, do_sigmoid in zip(outputs, sigmoid_mask):
                if do_sigmoid:
                    results.append(torch.sigmoid(out))
                else:
                    results.append(out)
            # For backward, save the sigmoid outputs for gradient computation
            ctx.save_for_backward(
                x,
                weight,
                *[r if s else out for r, s, out in zip(results, sigmoid_mask, outputs)],
            )
            ctx.splits = splits
            ctx.sigmoid_mask = sigmoid_mask
            ctx.has_bias = bias is not None
            return tuple(results)
        else:
            ctx.save_for_backward(x, weight)
            ctx.splits = splits
            ctx.sigmoid_mask = None
            ctx.has_bias = bias is not None
            outputs = y.split(list(splits), dim=-1)
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.sigmoid_mask is not None:
            saved = ctx.saved_tensors
            x = saved[0]
            weight = saved[1]
            split_tensors = saved[2:]

            # Reconstruct grad through sigmoid
            grad_parts = []
            idx = 0
            for grad_out, do_sigmoid in zip(grad_outputs, ctx.sigmoid_mask):
                if do_sigmoid:
                    sig_out = split_tensors[idx]
                    # dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x))
                    # But we saved sigmoid(x) not x, so: grad * sig * (1-sig)
                    grad_parts.append(grad_out * sig_out * (1.0 - sig_out))
                else:
                    grad_parts.append(grad_out)
                idx += 1

            grad_y = torch.cat(grad_parts, dim=-1)
        else:
            x, weight = ctx.saved_tensors
            grad_y = torch.cat(list(grad_outputs), dim=-1)

        # dX = grad_y @ weight
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        grad_y_2d = grad_y.reshape(-1, grad_y.shape[-1])

        grad_x = grad_y_2d @ weight  # [M, D_in]
        grad_x = grad_x.reshape(orig_shape)

        # dW = grad_y^T @ x
        grad_weight = grad_y_2d.t() @ x_2d  # [D_out_total, D_in]

        # dBias
        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_y_2d.sum(dim=0)

        return grad_x, grad_weight, None, grad_bias, None


def fused_multi_proj(x, weight, splits, bias=None, sigmoid_mask=None):
    """
    Fused multi-projection: single GEMM on concatenated weights, then split.

    Args:
        x: [B, T, D_in] or [*, D_in]
        weight: [D_out_total, D_in] concatenated weight
        splits: tuple of ints (output dimension sizes)
        bias: optional [D_out_total]
        sigmoid_mask: optional tuple of bools (apply sigmoid to which outputs)

    Returns:
        tuple of tensors, one per split
    """
    return FusedMultiProjFn.apply(x, weight, splits, bias, sigmoid_mask)


# ============================================================================
# Triton GEMM kernel with fused sigmoid+bias epilogue (for beta_gk variant)
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _fused_proj_sigmoid_bias_kernel(
    X_ptr,
    W_ptr,
    Bias_ptr,
    Y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    N_sigmoid: tl.constexpr,  # first N_sigmoid output cols get sigmoid
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """GEMM Y = X @ W^T + bias, with sigmoid on first N_sigmoid output columns."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers
    X = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    W = W_ptr + rn[None, :] * stride_wn + rk[:, None] * stride_wk

    # Accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        x_vals = tl.load(
            X, mask=(rm[:, None] < M) & (rk[None, :] + k_start < K), other=0.0
        )
        w_vals = tl.load(
            W, mask=(rn[None, :] < N) & (rk[:, None] + k_start < K), other=0.0
        )
        acc += tl.dot(x_vals, w_vals)
        X += BLOCK_K * stride_xk
        W += BLOCK_K * stride_wk

    # Add bias
    bias_vals = tl.load(Bias_ptr + rn, mask=rn < N, other=0.0)
    acc += bias_vals[None, :]

    # Apply sigmoid to first N_sigmoid columns
    col_offset = pid_n * BLOCK_N
    needs_sigmoid = (col_offset + tl.arange(0, BLOCK_N)) < N_sigmoid
    acc = tl.where(needs_sigmoid[None, :], tl.sigmoid(acc), acc)

    # Store
    Y = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(Y, acc.to(Y_ptr.dtype.element_ty), mask=mask)


def fused_beta_gk_proj_triton(x, weight, bias, n_sigmoid):
    """
    Fused beta+gk projection with Triton: GEMM + bias + sigmoid on first n_sigmoid cols.

    Args:
        x: [B, T, D_in]
        weight: [N_out, D_in]  (N_out = n_sigmoid + n_rest)
        bias: [N_out]
        n_sigmoid: int, number of first output columns to apply sigmoid to

    Returns:
        (beta, gk) where beta has sigmoid applied
    """
    orig_shape = x.shape[:-1]
    D_in = x.shape[-1]
    N_out = weight.shape[0]
    M = x.reshape(-1, D_in).shape[0]

    x_2d = x.reshape(M, D_in).contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    y = torch.empty(M, N_out, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N_out, meta["BLOCK_N"]),
    )

    _fused_proj_sigmoid_bias_kernel[grid](
        x_2d,
        weight,
        bias,
        y,
        M,
        N_out,
        D_in,
        x_2d.stride(0),
        x_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        y.stride(0),
        y.stride(1),
        N_sigmoid=n_sigmoid,
    )

    y = y.reshape(*orig_shape, N_out)
    beta = y[..., :n_sigmoid]
    gk = y[..., n_sigmoid:]
    return beta, gk


# ============================================================================
# Convenience wrappers matching model usage patterns
# ============================================================================


def fused_qkvg_proj(x, weight):
    """DeltaNet 4-way projection: q, k, v, g each [B, T, 4096]."""
    return fused_multi_proj(x, weight, (4096, 4096, 4096, 4096))


def fused_qkv_proj(x, weight):
    """GSA 3-way projection: q, k, v each [B, T, 4096]."""
    return fused_multi_proj(x, weight, (4096, 4096, 4096))


def fused_dual_proj_sigmoid(x, weight):
    """GSA dual gate: g_v, g_o each [B, T, 4096] with sigmoid."""
    return fused_multi_proj(x, weight, (4096, 4096), sigmoid_mask=(True, True))


# ============================================================================
# PyTorch reference implementations (for correctness checking)
# ============================================================================


def pytorch_fused_qkvg(x, W_q, W_k, W_v, W_g):
    return F.linear(x, W_q), F.linear(x, W_k), F.linear(x, W_v), F.linear(x, W_g)


def pytorch_fused_qkv(x, W_q, W_k, W_v):
    return F.linear(x, W_q), F.linear(x, W_k), F.linear(x, W_v)


def pytorch_fused_beta_gk(x, W_b, b_b, W_gk, b_gk):
    beta = torch.sigmoid(F.linear(x, W_b, b_b))
    gk = F.linear(x, W_gk, b_gk)
    return beta, gk


def pytorch_fused_dual_sigmoid(x, W_gv, W_go):
    return torch.sigmoid(F.linear(x, W_gv)), torch.sigmoid(F.linear(x, W_go))
