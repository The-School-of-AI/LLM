"""
Fused Triton kernel for CountSketch post-processing.

Fuses three operations that currently run as separate GPU kernels:
    1. Elementwise multiply by preconditioner  P[row, col]
    2. Elementwise multiply by sign table       sign[row, col]
    3. Scatter-add into sketch bins             sketch[b, hash[row, col]] += val

The outer-product matmul (bmm) stays with cuBLAS — it's already optimal.

Usage:
    from .triton_countsketch import fused_sketch_scatter, HAS_TRITON
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_sketch_scatter_kernel(
        # Pointers
        grad_ptr,  # [B, R, C]  — the bmm output (grad chunk)
        precond_ptr,  # [R, C]     — diagonal preconditioner slice (or null)
        sign_ptr,  # [R, C]     — pair_sign = row_sign[r] * col_sign[c]
        hash_ptr,  # [R, C]     — pair_hash = (row_hash[r] + col_hash[c]) % m
        sketch_ptr,  # [B, m]     — output sketch (accumulated)
        # Dimensions
        B: tl.constexpr,
        R: tl.constexpr,  # num rows in this chunk
        C: tl.constexpr,  # in_dim (columns)
        M: tl.constexpr,  # sketch_dim
        HAS_PRECOND: tl.constexpr,
        BLOCK_RC: tl.constexpr,
    ):
        """
        Each program instance processes one (batch, row_col_block) pair.
        For each element in the grad chunk:
            val = grad[b, r, c] * precond[r, c] * sign[r, c]
            sketch[b, hash[r, c]] += val  (via atomic add)
        """
        pid_b = tl.program_id(0)  # batch index
        pid_rc = tl.program_id(1)  # flattened (row, col) block index

        rc_offsets = pid_rc * BLOCK_RC + tl.arange(0, BLOCK_RC)
        mask = rc_offsets < (R * C)

        # Load grad[b, r, c]
        grad_off = pid_b * (R * C) + rc_offsets
        grad_val = tl.load(grad_ptr + grad_off, mask=mask, other=0.0)

        # Load and apply preconditioner
        if HAS_PRECOND:
            precond_val = tl.load(precond_ptr + rc_offsets, mask=mask, other=1.0)
            grad_val = grad_val * precond_val

        # Load and apply sign
        sign_val = tl.load(sign_ptr + rc_offsets, mask=mask, other=1.0)
        grad_val = grad_val * sign_val

        # Load hash indices and scatter into sketch
        hash_idx = tl.load(hash_ptr + rc_offsets, mask=mask, other=0)
        sketch_off = pid_b * M + hash_idx

        tl.atomic_add(sketch_ptr + sketch_off, grad_val, mask=mask)

    def fused_sketch_scatter(
        grad_chunk: torch.Tensor,  # [B, R, C] float32
        preconditioner: torch.Tensor | None,  # [R, C] float32 or None
        pair_sign: torch.Tensor,  # [R, C] float32
        pair_hash: torch.Tensor,  # [R, C] int64
        sketches: torch.Tensor,  # [B, M] float32 (modified in-place)
    ) -> None:
        """
        Fused scatter: applies preconditioner, signs, and scatter-adds into
        sketch bins in a single GPU kernel launch.

        Args:
            grad_chunk: [B, R, C] output of torch.bmm (grad outer product chunk)
            preconditioner: [R, C] diagonal preconditioner for this row slice, or None
            pair_sign: [R, C] CountSketch sign table
            pair_hash: [R, C] CountSketch hash-to-bin table (int64)
            sketches: [B, M] output sketch tensor (accumulated in-place)
        """
        B, R, C = grad_chunk.shape
        M = sketches.shape[1]

        # Flatten the (R, C) spatial dims for the kernel
        grad_flat = grad_chunk.reshape(B, R * C).contiguous()

        precond_flat = (
            preconditioner.reshape(R * C).contiguous()
            if preconditioner is not None
            else None
        )
        sign_flat = pair_sign.reshape(R * C).contiguous()
        hash_flat = pair_hash.reshape(R * C).contiguous()

        RC = R * C
        BLOCK_RC = triton.next_power_of_2(min(RC, 1024))

        grid = (B, triton.cdiv(RC, BLOCK_RC))

        _fused_sketch_scatter_kernel[grid](
            grad_flat,
            precond_flat if precond_flat is not None else grad_flat,  # dummy, not used
            sign_flat,
            hash_flat,
            sketches,
            B=B,
            R=R,
            C=C,
            M=M,
            HAS_PRECOND=(preconditioner is not None),
            BLOCK_RC=BLOCK_RC,
        )

else:
    # PyTorch fallback — same as the inline code in project_linear_batch
    def fused_sketch_scatter(
        grad_chunk: torch.Tensor,
        preconditioner: torch.Tensor | None,
        pair_sign: torch.Tensor,
        pair_hash: torch.Tensor,
        sketches: torch.Tensor,
    ) -> None:
        B = grad_chunk.shape[0]
        if preconditioner is not None:
            grad_chunk = grad_chunk * preconditioner.unsqueeze(0)
        signed = (grad_chunk * pair_sign.unsqueeze(0)).reshape(B, -1)
        idx = pair_hash.reshape(1, -1).expand(B, -1)
        sketches.scatter_add_(1, idx, signed)
