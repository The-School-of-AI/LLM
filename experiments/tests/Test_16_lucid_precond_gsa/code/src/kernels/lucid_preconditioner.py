"""
LUCID Preconditioner — arXiv:2602.10410
========================================

Decorrelates keys in RKHS by solving a lower-triangular system:
    P · Y = V
where P = M ⊙ exp(K_RN · K_RN⊤ / √d − √d)

The preconditioned values Y replace raw values V in the attention
computation, sharpening retrieval without lowering softmax temperature.

Provides:
    1. pytorch_lucid_precondition()      — full-matrix reference (for testing)
    2. pytorch_lucid_precondition_blockwise() — memory-efficient block solver
    3. triton_lucid_precondition()        — fused Triton kernel (forward only)
    4. lucid_precondition()              — dispatch: Triton if available, else PyTorch

All functions operate per-head: inputs are [B, T, D] for a single head,
or [B, T, H, D] with head dimension handled internally.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── Triton availability ──────────────────────────────────────────────
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# =====================================================================
# Utility: RMS-normalize keys for the preconditioner
# =====================================================================

def _rms_normalize_keys(K: torch.Tensor) -> torch.Tensor:
    """
    RMS-normalize keys: K_RN = √d · K / ‖K‖₂

    This ensures the preconditioner matrix has unit diagonal (self-similarity = 1)
    and controlled off-diagonal magnitudes for numerical stability.

    Args:
        K: [B, T, D] or [B, T, H, D] key tensor

    Returns:
        K_RN: same shape, RMS-normalized
    """
    d = K.shape[-1]
    norm = K.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return math.sqrt(d) * K / norm


# =====================================================================
# PyTorch Reference: Full-matrix solve (for testing / short sequences)
# =====================================================================

def pytorch_lucid_precondition(
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    LUCID preconditioning via full-matrix triangular solve.

    Computes Y such that P · Y = V, where:
        P = causal_mask ⊙ exp(K_RN · K_RN⊤ / √d − √d)

    This materializes the full [T, T] preconditioner — use only for
    testing or short sequences (T ≤ 1024).

    Args:
        K: [B, T, D] keys (single head) or [B, T, H, D] (multi-head)
        V: [B, T, D] values (single head) or [B, T, H, D] (multi-head)

    Returns:
        Y: same shape as V, preconditioned values
    """
    multi_head = K.dim() == 4
    if multi_head:
        B, T, H, D = K.shape
        # Process each head independently — reshape to [B*H, T, D]
        K = K.permute(0, 2, 1, 3).reshape(B * H, T, D)
        V = V.permute(0, 2, 1, 3).reshape(B * H, T, D)
    else:
        B, T, D = K.shape

    sqrt_d = math.sqrt(D)

    # Step 1: RMS-normalize keys
    K_RN = _rms_normalize_keys(K)  # [BH, T, D]

    # Step 2: Build preconditioner matrix P
    # P_ij = exp(K_RN_i · K_RN_j / √d − √d) for i >= j, else 0
    # Diagonal: exp(K_RN_i · K_RN_i / √d − √d) = exp(d/√d − √d) = exp(0) = 1
    # (since ‖K_RN‖² = d after RMS normalization)
    scores = torch.bmm(K_RN, K_RN.transpose(-2, -1)) / sqrt_d - sqrt_d  # [BH, T, T]

    # Apply causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(T, T, device=K.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    P = torch.exp(scores)  # [BH, T, T], lower triangular, unit diagonal

    # Step 3: Solve P · Y = V via triangular solve
    # P is lower-triangular with unit diagonal
    Y = torch.linalg.solve_triangular(P, V, upper=False)

    if multi_head:
        Y = Y.reshape(B, H, T, D).permute(0, 2, 1, 3)  # [B, T, H, D]

    return Y


# =====================================================================
# PyTorch Reference: Block-wise solve (memory-efficient, Algorithm 2)
# =====================================================================

def pytorch_lucid_precondition_blockwise(
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Memory-efficient LUCID preconditioning via block-wise forward substitution.

    Implements Algorithm 2 from the LUCID paper. Never materializes the full
    [T, T] preconditioner — only [BS, BS] blocks at a time.

    Args:
        K: [B, T, D] keys (single head) or [B, T, H, D] (multi-head)
        V: [B, T, D] values (single head) or [B, T, H, D] (multi-head)
        block_size: Block size for the solver (default 64)

    Returns:
        Y: same shape as V, preconditioned values
    """
    multi_head = K.dim() == 4
    if multi_head:
        B, T, H, D = K.shape
        K = K.permute(0, 2, 1, 3).reshape(B * H, T, D)
        V = V.permute(0, 2, 1, 3).reshape(B * H, T, D)
    else:
        B, T, D = K.shape

    sqrt_d = math.sqrt(D)
    BS = min(block_size, T)

    # Step 1: RMS-normalize keys
    K_RN = _rms_normalize_keys(K)  # [BH, T, D]

    # Step 2: Block-wise forward substitution
    Y = torch.zeros_like(V)

    for i in range(0, T, BS):
        i_end = min(i + BS, T)
        K_RN_i = K_RN[:, i:i_end, :]     # [BH, BS_i, D]
        rhs = V[:, i:i_end, :].clone()     # [BH, BS_i, D]

        # Subtract contributions from all previous blocks
        for j in range(0, i, BS):
            j_end = min(j + BS, T)
            K_RN_j = K_RN[:, j:j_end, :]  # [BH, BS_j, D]

            # exp(K_RN_i · K_RN_j⊤ / √d − √d) — always lower-triangular
            # since i > j, ALL entries in this off-diagonal block are valid
            exp_ij = torch.exp(
                torch.bmm(K_RN_i, K_RN_j.transpose(-2, -1)) / sqrt_d - sqrt_d
            )  # [BH, BS_i, BS_j]

            rhs = rhs - torch.bmm(exp_ij, Y[:, j:j_end, :])

        # Solve diagonal block
        # exp(K_RN_i · K_RN_i⊤ / √d − √d) — lower triangular within block
        exp_ii = torch.exp(
            torch.bmm(K_RN_i, K_RN_i.transpose(-2, -1)) / sqrt_d - sqrt_d
        )  # [BH, BS_i, BS_i]

        # Apply causal mask to diagonal block
        bs_actual = i_end - i
        diag_mask = torch.tril(torch.ones(bs_actual, bs_actual, device=K.device, dtype=torch.bool))
        exp_ii = exp_ii.masked_fill(~diag_mask, 0.0)

        Y[:, i:i_end, :] = torch.linalg.solve_triangular(
            exp_ii, rhs, upper=False)


    if multi_head:
        actual_B = Y.shape[0] // H
        Y = Y.reshape(actual_B, H, T, D).permute(0, 2, 1, 3)

    return Y


# =====================================================================
# Triton Kernel: Fused block-wise forward substitution
# =====================================================================

if HAS_TRITON:

    @triton.jit
    def _lucid_rms_norm_kernel(
        K_ptr, K_RN_ptr,
        T_val, D_val: tl.constexpr,
        stride_kb, stride_kt, stride_kd,
        stride_ob, stride_ot, stride_od,
    ):
        """Fused RMS normalization: K_RN = √D · K / ‖K‖₂"""
        batch_idx = tl.program_id(0)
        token_idx = tl.program_id(1)

        d_offsets = tl.arange(0, D_val)
        mask = d_offsets < D_val

        # Load row
        k_ptr = K_ptr + batch_idx * stride_kb + token_idx * stride_kt
        k_vals = tl.load(k_ptr + d_offsets * stride_kd, mask=mask, other=0.0).to(tl.float32)

        # Compute norm
        k_sq = k_vals * k_vals
        norm_sq = tl.sum(k_sq, axis=0)
        norm = tl.sqrt(norm_sq + 1e-16)
        sqrt_d = tl.sqrt(float(D_val))

        # Normalize
        k_rn = sqrt_d * k_vals / norm

        # Store
        o_ptr = K_RN_ptr + batch_idx * stride_ob + token_idx * stride_ot
        tl.store(o_ptr + d_offsets * stride_od, k_rn, mask=mask)


def triton_lucid_precondition(
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    LUCID preconditioning via Triton-accelerated block-wise forward substitution.

    Uses Triton for RMS normalization, then falls back to PyTorch's
    solve_triangular for the block solves (cuBLAS TRSM is already highly
    optimized and hard to beat with Triton for small block sizes).

    The key optimization: RMS norm + block score computation is fused,
    and we never materialize the full [T, T] matrix.

    Args:
        K: [B, T, D] or [B, T, H, D] keys
        V: [B, T, D] or [B, T, H, D] values
        block_size: Block size for solver

    Returns:
        Y: same shape as V, preconditioned values
    """
    if not HAS_TRITON:
        return pytorch_lucid_precondition_blockwise(K, V, block_size)

    multi_head = K.dim() == 4
    if multi_head:
        B, T, H, D = K.shape
        K_flat = K.permute(0, 2, 1, 3).reshape(B * H, T, D).contiguous()
        V_flat = V.permute(0, 2, 1, 3).reshape(B * H, T, D).contiguous()
    else:
        B, T, D = K.shape
        K_flat = K.contiguous()
        V_flat = V.contiguous()
        H = 1

    sqrt_d = math.sqrt(D)
    BS = min(block_size, T)
    BH = K_flat.shape[0]

    # Step 1: Triton-fused RMS normalization
    K_RN = torch.empty_like(K_flat)
    D_pow2 = triton.next_power_of_2(D)

    if D_pow2 <= 1024:  # Triton path for reasonable head dims
        grid = (BH, T)
        _lucid_rms_norm_kernel[grid](
            K_flat, K_RN,
            T, D_pow2,
            K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
            K_RN.stride(0), K_RN.stride(1), K_RN.stride(2),
        )
    else:
        # Fallback for large D
        K_RN = _rms_normalize_keys(K_flat)

    # Step 2: Block-wise forward substitution using cuBLAS TRSM
    # (cuBLAS TRSM is already batched and optimized — Triton doesn't
    # beat it for dense triangular solves at typical block sizes)
    Y = torch.zeros_like(V_flat)

    for i in range(0, T, BS):
        i_end = min(i + BS, T)
        K_RN_i = K_RN[:, i:i_end, :]
        rhs = V_flat[:, i:i_end, :].clone()

        # Subtract contributions from previous blocks
        for j in range(0, i, BS):
            j_end = min(j + BS, T)
            K_RN_j = K_RN[:, j:j_end, :]

            exp_ij = torch.exp(
                torch.bmm(K_RN_i, K_RN_j.transpose(-2, -1)) / sqrt_d - sqrt_d
            )
            rhs = rhs - torch.bmm(exp_ij, Y[:, j:j_end, :])

        # Diagonal block solve
        exp_ii_scores = torch.bmm(K_RN_i, K_RN_i.transpose(-2, -1)) / sqrt_d - sqrt_d
        bs_actual = i_end - i
        causal = torch.tril(torch.ones(bs_actual, bs_actual, device=K.device, dtype=torch.bool))
        exp_ii = torch.exp(exp_ii_scores.masked_fill(~causal, float('-inf')))

        Y[:, i:i_end, :] = torch.linalg.solve_triangular(
            exp_ii, rhs, upper=False)


    if multi_head:
        Y = Y.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()

    return Y


# =====================================================================
# Autograd wrapper for training (forward: fast, backward: PyTorch)
# =====================================================================

class LucidPreconditionFunction(torch.autograd.Function):
    """
    Custom autograd function for LUCID preconditioning.

    Forward: uses Triton-accelerated path (or blockwise PyTorch)
    Backward: lets PyTorch autograd handle gradients naturally

    Since we want gradients to flow through the preconditioner during
    training, we use a simple wrapper that re-runs the PyTorch version
    in the backward pass rather than implementing a custom backward kernel.
    """

    @staticmethod
    def forward(ctx, K, V, block_size):
        # Save for backward
        ctx.save_for_backward(K, V)
        ctx.block_size = block_size

        # Use the fastest available path
        if HAS_TRITON and K.is_cuda:
            Y = triton_lucid_precondition(K, V, block_size)
        else:
            Y = pytorch_lucid_precondition_blockwise(K, V, block_size)

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        K, V = ctx.saved_tensors

        # Re-run with autograd enabled to get gradients
        with torch.enable_grad():
            K_ag = K.detach().requires_grad_(True)
            V_ag = V.detach().requires_grad_(True)
            Y_ag = pytorch_lucid_precondition_blockwise(K_ag, V_ag, ctx.block_size)
            Y_ag.backward(grad_Y)

        return K_ag.grad, V_ag.grad, None


# =====================================================================
# Dispatch function (main entry point)
# =====================================================================

def lucid_precondition(
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    training: bool = True,
) -> torch.Tensor:
    """
    Apply LUCID preconditioning to values.

    Decorrelates keys in RKHS by solving:
        P · Y = V
    where P = causal_mask ⊙ exp(K_RN · K_RN⊤ / √d − √d)

    This is the main entry point used by GatedSparseAttention.

    Args:
        K: [B, T, H, D] attention keys (after RoPE)
        V: [B, T, H, D] gated values
        block_size: Block size for memory-efficient solver
        training: If True, use autograd wrapper for gradient flow

    Returns:
        Y: [B, T, H, D] preconditioned values
    """
    if training and K.requires_grad:
        return LucidPreconditionFunction.apply(K, V, block_size)
    else:
        if HAS_TRITON and K.is_cuda:
            return triton_lucid_precondition(K, V, block_size)
        else:
            return pytorch_lucid_precondition_blockwise(K, V, block_size)
