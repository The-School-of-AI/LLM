"""
Triton Fused RMSNorm + SiLU + Gate Kernel
==========================================

Fuses the FusedRMSNormSwishGate operation into a single kernel launch:
    output = g * silu(rmsnorm(x))

Standard PyTorch path launches 3+ kernels:
1. RMSNorm (variance + rsqrt + mul by weight)
2. SiLU activation
3. Elementwise multiply with gate

This kernel does everything in a single pass:
- Load x, load g, load weight
- Compute RMSNorm in-register
- Apply SiLU
- Multiply by g
- Store result

Eliminates 2 intermediate tensors, saving ~2x memory bandwidth.
For DeltaNet with 6 layers * num_heads calls, this is significant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def _fused_norm_silu_gate_kernel(
        # Pointers
        x_ptr,  # Input tensor (to be normed)
        g_ptr,  # Gate tensor
        weight_ptr,  # RMSNorm weight
        out_ptr,  # Output tensor
        # Dimensions
        n_rows,
        n_cols,
        # Hyperparameters
        eps,
        # Strides
        stride_x_row,
        stride_g_row,
        stride_out_row,
        # Meta
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fused kernel: output = g * silu(rmsnorm(x))

        Each program handles one row.
        """
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        # Load x row
        x = tl.load(
            x_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        # Load gate row
        g = tl.load(
            g_ptr + row_idx * stride_g_row + col_offsets, mask=mask, other=0.0
        ).to(tl.float32)

        # Load weight
        w = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0).to(tl.float32)

        # RMSNorm: x_normed = x * rsqrt(mean(x^2) + eps) * weight
        x_sq = x * x
        variance = tl.sum(tl.where(mask, x_sq, 0.0), axis=0) / n_cols
        rstd = tl.rsqrt(variance + eps)
        x_normed = x * rstd * w

        # SiLU: silu(x) = x * sigmoid(x)
        # Manual sigmoid for Triton compatibility: 1/(1+exp(-x))
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x_normed))
        silu_x = x_normed * sigmoid_x

        # Gate: output = g * silu(x_normed)
        output = g * silu_x

        # Store
        tl.store(out_ptr + row_idx * stride_out_row + col_offsets, output, mask=mask)


def triton_fused_norm_silu_gate(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm + SiLU + Gate via Triton.

    Computes: output = g * silu(rmsnorm(x, weight))

    Args:
        x: Input tensor [..., D] (to be normalized)
        g: Gate tensor [..., D]
        weight: RMSNorm weight [D]
        eps: Numerical stability

    Returns:
        Output tensor [..., D]
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_fused_norm_silu_gate")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    g_2d = g.reshape(-1, g.shape[-1])
    n_rows, n_cols = x_2d.shape

    out = torch.empty_like(x_2d)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(max(BLOCK_SIZE, 128), 4096)

    grid = (n_rows,)

    _fused_norm_silu_gate_kernel[grid](
        x_2d,
        g_2d,
        weight,
        out,
        n_rows,
        n_cols,
        eps,
        x_2d.stride(0),
        g_2d.stride(0),
        out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.reshape(orig_shape)


def pytorch_fused_norm_silu_gate(
    x: torch.Tensor,
    g: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PyTorch fallback for fused RMSNorm + SiLU + Gate.

    Computes: output = g * silu(rmsnorm(x, weight))
    """
    # RMSNorm
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps) * weight

    # SiLU + Gate
    return g * F.silu(x_normed)


class FusedRMSNormSiLUGate(nn.Module):
    """
    Fused RMSNorm + SiLU + Gate module.

    Computes: output = g * silu(rmsnorm(x))

    Automatically uses Triton kernel on CUDA, PyTorch fallback on CPU.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_triton = HAS_TRITON and torch.cuda.is_available()

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., dim] (to be normalized)
            g: Gate tensor [..., dim]

        Returns:
            output = g * silu(rmsnorm(x))
        """
        if self.use_triton and x.is_cuda:
            try:
                return triton_fused_norm_silu_gate(x, g, self.weight, self.eps)
            except Exception:
                return pytorch_fused_norm_silu_gate(x, g, self.weight, self.eps)
        return pytorch_fused_norm_silu_gate(x, g, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{self.dim}, eps={self.eps}, triton={self.use_triton}"
