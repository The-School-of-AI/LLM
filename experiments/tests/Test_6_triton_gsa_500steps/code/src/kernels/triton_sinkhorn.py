"""
Triton Fused Sinkhorn-Knopp Kernel
===================================

Fuses the entire Sinkhorn-Knopp iteration loop into a single GPU kernel launch.

Standard PyTorch implementation launches 2 * num_iters kernels (one per
row-normalize, one per col-normalize).  This kernel does all iterations
in-register, launching only ONCE for arbitrarily many iterations.

For mHC with n=4, each matrix is 4x4 = 16 elements — fits entirely in
registers per thread block, making the fused version essentially free
compared to the surrounding linear projections.

Performance:
- Eliminates 2 * num_iters kernel launches (default: 40 launches -> 1)
- Zero extra global memory traffic between iterations
- All intermediate results stay in SRAM/registers
"""

import torch
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
    def _sinkhorn_kernel(
        # Pointers
        H_ptr,          # Input:  raw H_res values [..., n, n]
        out_ptr,        # Output: doubly stochastic [..., n, n]
        # Dimensions
        num_matrices,   # Total number of n x n matrices (B * S)
        n,              # Matrix dimension (expansion_rate, typically 4)
        # Hyperparameters
        eps: tl.constexpr,
        num_iters: tl.constexpr,
        # Strides (row-major within each n x n matrix)
        stride_mat,     # Stride between matrices (= n * n for contiguous)
        # Meta
        N_SQ: tl.constexpr,  # n * n, must be power-of-2 padded
    ):
        """
        Fused Sinkhorn-Knopp: exp + iterative row/col normalization.

        Each program instance handles one n x n matrix.
        For n=4, N_SQ=16 elements fit in a single vector register.

        Algorithm per matrix:
            M = exp(H)
            for _ in range(num_iters):
                M = M / row_sum(M)  (broadcast over cols)
                M = M / col_sum(M)  (broadcast over rows)
        """
        mat_idx = tl.program_id(0)
        if mat_idx >= num_matrices:
            return

        # Load the entire n x n matrix into registers
        elem_offsets = tl.arange(0, N_SQ)
        mask = elem_offsets < (n * n)

        base_ptr = H_ptr + mat_idx * stride_mat
        H_raw = tl.load(base_ptr + elem_offsets, mask=mask, other=0.0).to(tl.float32)

        # Step 1: exp to make all elements positive
        M = tl.exp(H_raw)

        # Derive row and column indices from flat offset
        row_idx = elem_offsets // n
        col_idx = elem_offsets % n

        # Step 2: Iterative row/col normalization
        for _iter in range(num_iters):
            # Row normalization: each row sums to 1
            row_sum = tl.zeros([N_SQ], dtype=tl.float32)
            for r in range(n):
                r_mask = (row_idx == r) & mask
                s = tl.sum(tl.where(r_mask, M, 0.0))
                row_sum = tl.where(row_idx == r, s + eps, row_sum)
            M = tl.where(mask, M / row_sum, 0.0)

            # Column normalization: each column sums to 1
            col_sum = tl.zeros([N_SQ], dtype=tl.float32)
            for c in range(n):
                c_mask = (col_idx == c) & mask
                s = tl.sum(tl.where(c_mask, M, 0.0))
                col_sum = tl.where(col_idx == c, s + eps, col_sum)
            M = tl.where(mask, M / col_sum, 0.0)

        # Store result
        out_base = out_ptr + mat_idx * stride_mat
        tl.store(out_base + elem_offsets, M, mask=mask)


def triton_sinkhorn_knopp(
    H: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Fused Sinkhorn-Knopp via Triton.

    Projects matrices onto the Birkhoff polytope (doubly stochastic matrices)
    in a single kernel launch.

    Args:
        H: Input tensor [..., n, n] (will be exponentiated)
        num_iters: Number of Sinkhorn iterations (paper default: 20)
        eps: Numerical stability constant

    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_sinkhorn_knopp")

    orig_shape = H.shape
    n = H.shape[-1]
    assert H.shape[-2] == n, f"Expected square matrices, got {H.shape[-2:]} "

    # Flatten leading dims: [..., n, n] -> [num_matrices, n*n]
    H_flat = H.reshape(-1, n * n).contiguous()
    num_matrices = H_flat.shape[0]

    # Output
    out_flat = torch.empty_like(H_flat)

    # Block size must be power of 2 >= n*n
    N_SQ = triton.next_power_of_2(n * n)

    # Launch: one program per matrix
    grid = (num_matrices,)

    _sinkhorn_kernel[grid](
        H_flat,
        out_flat,
        num_matrices,
        n,
        eps=eps,
        num_iters=num_iters,
        stride_mat=n * n,
        N_SQ=N_SQ,
    )

    return out_flat.reshape(orig_shape)


def pytorch_sinkhorn_knopp(
    H: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PyTorch fallback for Sinkhorn-Knopp.

    Args:
        H: Input tensor [..., n, n]
        num_iters: Number of iterations
        eps: Numerical stability constant

    Returns:
        Doubly stochastic matrix [..., n, n]
    """
    M = torch.exp(H)
    for _ in range(num_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M
