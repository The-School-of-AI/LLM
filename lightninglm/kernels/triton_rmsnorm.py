"""
Triton Fused RMSNorm Kernel — Forward AND Backward
====================================================

Fused kernels for RMSNorm with forward + backward in Triton.
Based on Liger-Kernel (LinkedIn, Apache-2.0) and Unsloth implementations.

Key improvement over previous version:
- OLD: Forward-only Triton kernel, backward via PyTorch autograd (3-4 kernels)
- NEW: Both forward AND backward are fused Triton kernels (1 kernel each)

The forward kernel saves RSTD (reciprocal standard deviation) per row — a tiny
tensor of shape [n_rows] — which the backward kernel reuses to avoid recomputing
variance. This saves 4 operations (*, sum, /, sqrt) per row in backward.

Backward math:
    dX = rstd * (dY*W - (1/N) * rstd^2 * dot(dY*W, X) * X)
    dW = sum_over_rows(dY * X * rstd)

Attribution:
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel (Apache-2.0)
- Unsloth: https://github.com/unslothai/unsloth (Apache-2.0)
"""

import math
from typing import Optional

import torch
import torch.nn as nn

# Check for Triton availability
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

# Import profiling helpers
try:
    from ..profiler import kernel_region
except ImportError:
    # Fallback: no-op context manager
    from contextlib import contextmanager

    @contextmanager
    def kernel_region(name: str):
        yield


# ═══════════════════════════════════════════════════════════════════════
# Triton Forward Kernel
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _rmsnorm_fwd_kernel(
        # Pointers
        Y_ptr,  # Output tensor
        X_ptr,  # Input tensor
        W_ptr,  # RMSNorm weight
        RSTD_ptr,  # Saved reciprocal std (for backward)
        # Dimensions
        n_cols,
        # Hyperparameters
        eps,
        # Strides
        stride_x_row,
        stride_y_row,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RMSNorm forward: computes output AND saves RSTD for backward.

        out = (x / RMS(x)) * w
        rstd = 1 / sqrt(mean(x^2) + eps)

        Computation done in fp32 for numerical stability (Llama-style).
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input row
        x_base = X_ptr + row_idx * stride_x_row
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        X_row_dtype = X_row.dtype

        # Upcast to fp32 for variance computation (Llama-style)
        X_row_f32 = X_row.to(tl.float32)

        # Compute RMSNorm
        mean_square = tl.sum(X_row_f32 * X_row_f32, axis=0) / n_cols
        rstd = 1.0 / tl.sqrt(mean_square + eps)

        # Save RSTD for backward (tiny: 1 scalar per row)
        tl.store(RSTD_ptr + row_idx, rstd)

        # Normalize
        normed = X_row_f32 * rstd

        # Cast back to input dtype, then apply weight (Llama-style casting)
        normed = normed.to(X_row_dtype)
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=1.0)
        output = normed * W_row

        # Store output
        y_base = Y_ptr + row_idx * stride_y_row
        tl.store(y_base + col_offsets, output, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Triton Backward Kernel
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _rmsnorm_bwd_kernel(
        # Pointers
        dY_ptr,  # Gradient of output
        dX_ptr,  # Gradient of input (output)
        X_ptr,  # Saved input (from forward)
        W_ptr,  # Weight
        RSTD_ptr,  # Saved reciprocal std
        dW_ptr,  # Partial gradient of weight (output, per-block)
        # Dimensions
        n_rows,
        n_cols,
        # Strides
        stride_dy_row,
        stride_dx_row,
        stride_x_row,
        # Control
        rows_per_program,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused RMSNorm backward kernel.

        Computes dX and partial dW in a single kernel launch.
        Uses saved RSTD from forward to avoid recomputing variance.

        Math:
            m = dY * W
            dX = rstd * (m - (1/N) * rstd^2 * dot(m, X) * X)
            dW += dY * (X * rstd)  (accumulated across rows)
        """
        row_block_id = tl.program_id(0).to(tl.int64)
        row_start = row_block_id * rows_per_program
        row_end = min((row_block_id + 1) * rows_per_program, n_rows)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Accumulator for dW (summed across rows assigned to this block)
        dW_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        # Load weight once (shared across all rows)
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

        for row_idx in range(row_start, row_end):
            # Load dY and X for this row
            dy_base = dY_ptr + row_idx * stride_dy_row
            x_base = X_ptr + row_idx * stride_x_row

            dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
            X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)

            # Load cached RSTD (1 scalar)
            rstd = tl.load(RSTD_ptr + row_idx)

            # Upcast X to fp32 for computation
            X_row_f32 = X_row.to(tl.float32)

            # m = dY * W (Llama-style: multiply in input dtype, then upcast)
            m = (dY_row * W_row).to(tl.float32)

            # dX = rstd * (m - (1/N) * rstd^2 * dot(m, X) * X)
            dot_mx = tl.sum(m * X_row_f32, axis=0)
            dX_row = rstd * m + rstd * (
                -(1.0 / n_cols) * rstd * rstd * dot_mx * X_row_f32
            )

            # Store dX (cast back to input dtype)
            dx_base = dX_ptr + row_idx * stride_dx_row
            tl.store(dx_base + col_offsets, dX_row.to(X_row.dtype), mask=mask)

            # Accumulate dW: dY * (X * rstd), cast to input dtype first (Llama-style)
            dW_acc += dY_row * (X_row_f32 * rstd).to(X_row.dtype)

        # Store partial dW for this block (will be summed across blocks in wrapper)
        dw_base = dW_ptr + row_block_id * n_cols
        tl.store(dw_base + col_offsets, dW_acc, mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# torch.autograd.Function wrapper
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    def _calculate_settings(n_cols):
        """Calculate BLOCK_SIZE and num_warps for a given hidden dim."""
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 8192:
            num_warps = 16
        return BLOCK_SIZE, num_warps

    class LigerRMSNormFunction(torch.autograd.Function):
        """
        Fused RMSNorm with Triton forward + backward.

        Forward:  Computes RMSNorm and saves RSTD (1 scalar per row)
        Backward: Uses saved RSTD to compute dX and dW in a single kernel

        Attribution: Based on Liger-Kernel (Apache-2.0)
        """

        @staticmethod
        def forward(ctx, X, W, eps):
            """
            Args:
                X: Input tensor [..., hidden_size]
                W: Weight parameter [hidden_size]
                eps: Epsilon for numerical stability

            Returns:
                Normalized tensor, same dtype as X
            """
            with kernel_region("rmsnorm_fwd_total"):
                orig_shape = X.shape
                n_cols = X.shape[-1]

                with kernel_region("rmsnorm_fwd_reshape"):
                    X_2d = X.contiguous().reshape(-1, n_cols)

                n_rows = X_2d.shape[0]

                BLOCK_SIZE, num_warps = _calculate_settings(n_cols)

                # Safety: fall back to PyTorch if dim is too large
                if BLOCK_SIZE > 65536:
                    return pytorch_rmsnorm(X, W, eps)

                with kernel_region("rmsnorm_fwd_alloc"):
                    Y = torch.empty_like(X_2d)
                    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

                with kernel_region("rmsnorm_fwd_kernel"):
                    _rmsnorm_fwd_kernel[(n_rows,)](
                        Y,
                        X_2d,
                        W,
                        RSTD,
                        n_cols,
                        eps,
                        X_2d.stride(0),
                        Y.stride(0),
                        BLOCK_SIZE=BLOCK_SIZE,
                        num_warps=num_warps,
                    )

                with kernel_region("rmsnorm_fwd_reshape_out"):
                    Y = Y.reshape(orig_shape)

                # Save for backward
                ctx.save_for_backward(X_2d, W, RSTD)
                ctx.BLOCK_SIZE = BLOCK_SIZE
                ctx.num_warps = num_warps
                ctx.n_rows = n_rows
                ctx.n_cols = n_cols
                ctx.orig_shape = orig_shape

                return Y

        @staticmethod
        def backward(ctx, dY):
            with kernel_region("rmsnorm_bwd_total"):
                X_2d, W, RSTD = ctx.saved_tensors
                BLOCK_SIZE = ctx.BLOCK_SIZE
                num_warps = ctx.num_warps
                n_rows = ctx.n_rows
                n_cols = ctx.n_cols

                with kernel_region("rmsnorm_bwd_reshape"):
                    dY_2d = dY.contiguous().reshape(-1, n_cols)

                # Allocate outputs
                with kernel_region("rmsnorm_bwd_alloc"):
                    dX = torch.empty_like(X_2d)

                    # Number of SMs for dW accumulation across row blocks
                    sm_count = torch.cuda.get_device_properties(
                        X_2d.device
                    ).multi_processor_count
                    _dW = torch.empty(
                        sm_count, n_cols, dtype=torch.float32, device=W.device
                    )

                rows_per_program = math.ceil(n_rows / sm_count)
                grid = (sm_count,)

                with kernel_region("rmsnorm_bwd_kernel"):
                    _rmsnorm_bwd_kernel[grid](
                        dY_2d,
                        dX,
                        X_2d,
                        W,
                        RSTD,
                        _dW,
                        n_rows,
                        n_cols,
                        dY_2d.stride(0),
                        dX.stride(0),
                        X_2d.stride(0),
                        rows_per_program,
                        BLOCK_SIZE=BLOCK_SIZE,
                        num_warps=num_warps,
                    )

                # Sum partial dW across SM blocks → final dW
                with kernel_region("rmsnorm_bwd_dw_reduce"):
                    dW = _dW.sum(dim=0).to(W.dtype)

                return dX.reshape(ctx.orig_shape), dW, None


# ═══════════════════════════════════════════════════════════════════════
# Public API (drop-in replacement, now with backward support)
# ═══════════════════════════════════════════════════════════════════════


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply RMSNorm using fused Triton kernels (forward + backward).

    IMPORTANT: Unlike the old version, this now works WITH gradients enabled.
    The LigerRMSNormFunction handles both forward and backward passes in Triton.

    Args:
        x: Input tensor [..., hidden_size]
        weight: Weight parameter [hidden_size]
        eps: Epsilon for numerical stability
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor of same shape as x
    """
    with kernel_region("rmsnorm_total"):
        if not HAS_TRITON:
            raise ImportError("Triton is required for triton_rmsnorm")

        # Handle optional residual (add before norm)
        if residual is not None:
            with kernel_region("rmsnorm_residual_add"):
                x = x + residual

        with kernel_region("rmsnorm_apply"):
            return LigerRMSNormFunction.apply(x, weight, eps)


def pytorch_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    PyTorch fallback for RMSNorm.

    Args:
        x: Input tensor [..., hidden_size]
        weight: Weight parameter [hidden_size]
        eps: Epsilon for numerical stability
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor of same shape as x
    """
    if residual is not None:
        x = x + residual

    in_dtype = x.dtype
    x_f = x.float()
    variance = x_f.pow(2).mean(-1, keepdim=True)
    x_normed = x_f * torch.rsqrt(variance + eps)
    return (x_normed * weight.float()).to(in_dtype)


# ═══════════════════════════════════════════════════════════════════════
# Legacy forward-only kernel (kept for benchmark comparison)
# ═══════════════════════════════════════════════════════════════════════


def triton_rmsnorm_fwd_only(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    OLD forward-only Triton RMSNorm (no fused backward).

    Kept for benchmark comparison — demonstrates the improvement
    of the new LigerRMSNormFunction over this approach.

    WARNING: Backward falls back to PyTorch autograd (3-4 separate kernels).
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required")

    if residual is not None:
        x = x + residual

    orig_shape = x.shape
    x_2d = x.contiguous().reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    out = torch.empty_like(x_2d)
    BLOCK_SIZE, num_warps = _calculate_settings(n_cols)

    if BLOCK_SIZE > 65536:
        return pytorch_rmsnorm(x, weight, eps)

    # Dummy RSTD (not saved for backward since we don't have one)
    rstd_dummy = torch.empty(n_rows, dtype=torch.float32, device=x.device)

    _rmsnorm_fwd_kernel[(n_rows,)](
        out,
        x_2d,
        weight,
        rstd_dummy,
        n_cols,
        eps,
        x_2d.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out.reshape(orig_shape)


class TritonRMSNorm(nn.Module):
    """
    RMSNorm using fused Triton kernels with automatic fallback.

    Features:
    - Forward AND backward fused in Triton (via LigerRMSNormFunction)
    - 50% less memory bandwidth
    - 3-4x fewer kernel launches vs PyTorch
    - Works with torch.is_grad_enabled() = True (training!)
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.use_triton = HAS_TRITON and torch.cuda.is_available()

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_triton and x.is_cuda:
            try:
                return triton_rmsnorm(x, self.weight, self.eps, residual)
            except Exception as e:
                import warnings

                warnings.warn(f"Triton RMSNorm failed: {e}. Using PyTorch fallback.")
                return pytorch_rmsnorm(x, self.weight, self.eps, residual)
        else:
            return pytorch_rmsnorm(x, self.weight, self.eps, residual)

    def extra_repr(self) -> str:
        return f"{self.hidden_size}, eps={self.eps}, triton={self.use_triton}"
