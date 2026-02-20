"""
Triton Fused RMSNorm Kernel
============================

Fused kernels for RMSNorm with optional residual connections.
Based on Unsloth and FlashAttention implementations.

Performance improvements over PyTorch:
- Fuses variance computation + rsqrt + mul + residual add
- Reduces kernel launches from 3-4 to 1
- Cuts memory bandwidth by ~50%
"""

import torch
import torch.nn as nn
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
    def rmsnorm_forward_kernel(
        # Pointers
        x_ptr,          # Input tensor
        residual_ptr,   # Residual to add (optional)
        weight_ptr,     # RMSNorm weight
        out_ptr,        # Output tensor
        # Dimensions
        n_rows,         # Batch * seq_len
        n_cols,         # Hidden size
        # Hyperparameters
        eps,
        # Strides
        stride_x_row,
        stride_out_row,
        # Meta-parameters
        BLOCK_SIZE: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
    ):
        """
        Fused RMSNorm kernel with optional residual addition.

        Computes: out = (x + residual) * rsqrt(mean((x + residual)^2) + eps) * weight
        """
        row_idx = tl.program_id(0)

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load input row
        x_ptr_row = x_ptr + row_idx * stride_x_row
        x = tl.load(x_ptr_row + col_offsets, mask=mask, other=0.0).to(tl.float32)

        # Add residual if present
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)
            x = x + residual

        # Compute RMSNorm
        x_squared = x * x
        variance_sum = tl.sum(tl.where(mask, x_squared, 0.0), axis=0)
        variance = variance_sum / n_cols
        rstd = tl.rsqrt(variance + eps)
        normed = x * rstd

        # Apply weight
        weight = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)
        output = normed * weight

        # Store output
        out_ptr_row = out_ptr + row_idx * stride_out_row
        tl.store(out_ptr_row + col_offsets, output, mask=mask)


def triton_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply RMSNorm using Triton kernel.

    Args:
        x: Input tensor [..., hidden_size]
        weight: Weight parameter [hidden_size]
        eps: Epsilon for numerical stability
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor of same shape as x
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_rmsnorm")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE < n_cols:
        BLOCK_SIZE = BLOCK_SIZE * 2
    # FIX: BLOCK_SIZE must be >= n_cols for correctness. The old cap of 4096
    # silently dropped columns beyond 4096, producing uninitialized output
    # (caused NaN at batch_size > ~4 due to stale CUDA memory).
    # Triton supports BLOCK_SIZE up to 65536 on modern GPUs.
    BLOCK_SIZE = max(BLOCK_SIZE, 128)

    # Safety: fall back to PyTorch if dim is too large for Triton's tl.arange
    if BLOCK_SIZE > 65536:
        return pytorch_rmsnorm(x, weight, eps, residual)

    has_residual = residual is not None
    if has_residual:
        residual_2d = residual.reshape(-1, residual.shape[-1])
        residual_ptr = residual_2d
    else:
        residual_ptr = x_2d  # Dummy pointer (won't be used)

    grid = (n_rows,)

    rmsnorm_forward_kernel[grid](
        x_2d,
        residual_ptr,
        weight,
        out,
        n_rows,
        n_cols,
        eps,
        x_2d.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_RESIDUAL=has_residual,
    )

    return out.reshape(orig_shape)


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

    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


class TritonRMSNorm(nn.Module):
    """
    RMSNorm using Triton kernel with automatic fallback.

    Features:
    - Fused residual addition
    - 50% less memory bandwidth
    - 3-4x fewer kernel launches
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
        return f'{self.hidden_size}, eps={self.eps}, triton={self.use_triton}'