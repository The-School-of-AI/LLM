"""
LUCID Preconditioner — arXiv:2602.10410
========================================

Decorrelates keys in RKHS by solving a lower-triangular system:
    P · Y = V
where P = M ⊙ exp(K_RN · K_RN⊤ / √d − √d)

The preconditioned values Y replace raw values V in the attention
computation, sharpening retrieval without lowering softmax temperature.

Architecture:
    - Forward:  Triton-fused RMS norm → block-wise forward substitution (cuBLAS TRSM)
    - Backward: Triton-fused grad computation → block-wise backward substitution
    - Reversibility: Deterministic, stateless — safe for ReversibleMidpointStack

Provides:
    1. pytorch_lucid_precondition()           — full-matrix reference (testing)
    2. pytorch_lucid_precondition_blockwise() — memory-efficient block solver
    3. triton_lucid_precondition()            — fused Triton fwd kernel
    4. lucid_precondition()                   — dispatch (main entry point)

All functions operate per-head: inputs [B, T, H, D] handled internally.

Backward math (gradient of triangular solve P·Y = V):
    Given grad_Y (dL/dY), we need dL/dV and dL/dK.
    dL/dV = P⁻ᵀ · grad_Y  (backward substitution on Pᵀ)
    dL/dK = f(grad_Y, Y, K)  (chain rule through P's dependence on K)
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

    Ensures unit diagonal in preconditioner matrix (self-similarity = 1).

    Args:
        K: [..., D] key tensor

    Returns:
        K_RN: same shape, RMS-normalized
    """
    d = K.shape[-1]
    norm = K.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return math.sqrt(d) * K / norm


def _solve_triangular(A, B, upper=False):
    """
    Wrapper for solve_triangular with fp32 upcast.
    cuBLAS TRSM doesn't support bf16, so we cast to fp32 and back.
    """
    orig_dtype = B.dtype
    if orig_dtype == torch.bfloat16 or orig_dtype == torch.float16:
        return torch.linalg.solve_triangular(
            A.float(), B.float(), upper=upper
        ).to(orig_dtype)
    return torch.linalg.solve_triangular(A, B, upper=upper)


# =====================================================================
# Block-wise triangular solve (shared by forward and backward)
# =====================================================================

def _blockwise_forward_substitution(K_RN, V, sqrt_d, block_size, device):
    """
    Block-wise forward substitution: solve P·Y = V.

    P = causal ⊙ exp(K_RN · K_RN⊤ / √d − √d), lower-triangular, unit diagonal.
    Never materializes full [T, T] matrix — only [BS, BS] tiles.

    Args:
        K_RN: [BH, T, D] RMS-normalized keys
        V:    [BH, T, D] values (or gradient)
        sqrt_d: √D scalar
        block_size: tile size
        device: torch device

    Returns:
        Y: [BH, T, D] solution
    """
    BH, T, D = K_RN.shape
    BS = min(block_size, T)
    orig_dtype = V.dtype

    # Compute in fp32 for numerical stability (bf16 bmm accumulates errors)
    K_RN = K_RN.float()
    V = V.float()
    Y = torch.zeros_like(V)

    for i in range(0, T, BS):
        i_end = min(i + BS, T)
        K_RN_i = K_RN[:, i:i_end, :]
        rhs = V[:, i:i_end, :].clone()

        # Subtract contributions from all previous blocks
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
        causal = torch.tril(torch.ones(bs_actual, bs_actual, device=device, dtype=torch.bool))
        exp_ii = torch.exp(exp_ii_scores.masked_fill(~causal, float('-inf')))

        Y[:, i:i_end, :] = torch.linalg.solve_triangular(exp_ii, rhs, upper=False)

    return Y.to(orig_dtype)


def _blockwise_backward_substitution(K_RN, grad_Y, sqrt_d, block_size, device):
    """
    Block-wise backward substitution: solve Pᵀ · dV = grad_Y.

    This computes dL/dV = P⁻ᵀ · grad_Y by solving the upper-triangular
    system Pᵀ · dV = grad_Y, processing blocks from last to first.

    Args:
        K_RN:   [BH, T, D] RMS-normalized keys
        grad_Y: [BH, T, D] gradient w.r.t. output Y
        sqrt_d: √D scalar
        block_size: tile size
        device: torch device

    Returns:
        dV: [BH, T, D] gradient w.r.t. values V
    """
    BH, T, D = K_RN.shape
    BS = min(block_size, T)
    orig_dtype = grad_Y.dtype

    # Compute in fp32
    K_RN = K_RN.float()
    grad_Y = grad_Y.float()
    dV = torch.zeros(BH, T, D, device=device, dtype=torch.float32)

    # Process blocks from last to first (backward substitution)
    block_starts = list(range(0, T, BS))
    for idx in reversed(range(len(block_starts))):
        i = block_starts[idx]
        i_end = min(i + BS, T)
        K_RN_i = K_RN[:, i:i_end, :]
        rhs = grad_Y[:, i:i_end, :].clone()

        # Subtract contributions from all later blocks
        for jdx in range(idx + 1, len(block_starts)):
            j = block_starts[jdx]
            j_end = min(j + BS, T)
            K_RN_j = K_RN[:, j:j_end, :]

            P_ji = torch.exp(
                torch.bmm(K_RN_j, K_RN_i.transpose(-2, -1)) / sqrt_d - sqrt_d
            )
            rhs = rhs - torch.bmm(P_ji.transpose(-2, -1), dV[:, j:j_end, :])

        # Diagonal block: solve Pᵀ[i,i] · dV[i] = rhs
        exp_ii_scores = torch.bmm(K_RN_i, K_RN_i.transpose(-2, -1)) / sqrt_d - sqrt_d
        bs_actual = i_end - i
        causal = torch.tril(torch.ones(bs_actual, bs_actual, device=device, dtype=torch.bool))
        exp_ii = torch.exp(exp_ii_scores.masked_fill(~causal, float('-inf')))
        dV[:, i:i_end, :] = torch.linalg.solve_triangular(exp_ii.transpose(-2, -1), rhs, upper=True)

    return dV.to(orig_dtype)


# =====================================================================
# PyTorch Reference: Full-matrix solve (for testing / short sequences)
# =====================================================================

def pytorch_lucid_precondition(
    K: torch.Tensor,
    V: torch.Tensor,
) -> torch.Tensor:
    """
    LUCID preconditioning via full-matrix triangular solve.
    Use only for testing or short sequences (T ≤ 1024).

    Args:
        K: [B, T, D] or [B, T, H, D] keys
        V: [B, T, D] or [B, T, H, D] values

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
    K_RN = _rms_normalize_keys(K)

    scores = torch.bmm(K_RN, K_RN.transpose(-2, -1)) / sqrt_d - sqrt_d
    causal_mask = torch.tril(torch.ones(T, T, device=K.device, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float('-inf'))
    P = torch.exp(scores)

    Y = _solve_triangular(P, V, upper=False)

    if multi_head:
        Y = Y.reshape(B, H, T, D).permute(0, 2, 1, 3)

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
    Never materializes full [T, T] matrix.

    Args:
        K: [B, T, D] or [B, T, H, D] keys
        V: [B, T, D] or [B, T, H, D] values
        block_size: tile size (default 64)

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
        H = 1

    sqrt_d = math.sqrt(D)
    K_RN = _rms_normalize_keys(K)
    Y = _blockwise_forward_substitution(K_RN, V, sqrt_d, block_size, K.device)

    if multi_head:
        actual_B = Y.shape[0] // H
        Y = Y.reshape(actual_B, H, T, D).permute(0, 2, 1, 3)

    return Y


# =====================================================================
# Triton Kernel: Fused RMS normalization
# =====================================================================

if HAS_TRITON:

    @triton.jit
    def _lucid_rms_norm_fwd_kernel(
        K_ptr, K_RN_ptr, Norm_ptr,
        T_val, D_val: tl.constexpr,
        stride_kb, stride_kt, stride_kd,
        stride_ob, stride_ot, stride_od,
        stride_nb,
    ):
        """
        Fused forward RMS norm: K_RN = √D · K / ‖K‖₂
        Also saves the norm for backward reuse.
        """
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

        # Save norm for backward
        tl.store(Norm_ptr + batch_idx * stride_nb + token_idx, norm)

        # Normalize
        k_rn = sqrt_d * k_vals / norm

        # Store
        o_ptr = K_RN_ptr + batch_idx * stride_ob + token_idx * stride_ot
        tl.store(o_ptr + d_offsets * stride_od, k_rn, mask=mask)

    @triton.jit
    def _lucid_rms_norm_bwd_kernel(
        grad_K_RN_ptr, grad_K_ptr,
        K_ptr, Norm_ptr,
        T_val, D_val: tl.constexpr,
        stride_gb, stride_gt, stride_gd,
        stride_kb, stride_kt, stride_kd,
        stride_ob, stride_ot, stride_od,
        stride_nb,
    ):
        """
        Backward of RMS norm: given dL/dK_RN, compute dL/dK.

        K_RN = √D · K / ‖K‖
        dK_RN/dK = √D · (I - K·Kᵀ/‖K‖²) / ‖K‖
        dL/dK = dL/dK_RN · dK_RN/dK
        """
        batch_idx = tl.program_id(0)
        token_idx = tl.program_id(1)

        d_offsets = tl.arange(0, D_val)
        mask = d_offsets < D_val

        # Load grad_K_RN, K, norm
        g_ptr = grad_K_RN_ptr + batch_idx * stride_gb + token_idx * stride_gt
        k_ptr = K_ptr + batch_idx * stride_kb + token_idx * stride_kt
        grad_krn = tl.load(g_ptr + d_offsets * stride_gd, mask=mask, other=0.0).to(tl.float32)
        k_vals = tl.load(k_ptr + d_offsets * stride_kd, mask=mask, other=0.0).to(tl.float32)
        norm = tl.load(Norm_ptr + batch_idx * stride_nb + token_idx)

        sqrt_d = tl.sqrt(float(D_val))

        # dK = √D / ‖K‖ · (grad_krn - (grad_krn · K̂) · K̂)
        # where K̂ = K / ‖K‖
        inv_norm = 1.0 / (norm + 1e-16)
        k_hat = k_vals * inv_norm
        dot = tl.sum(grad_krn * k_hat, axis=0)
        grad_k = sqrt_d * inv_norm * (grad_krn - dot * k_hat)

        # Store
        o_ptr = grad_K_ptr + batch_idx * stride_ob + token_idx * stride_ot
        tl.store(o_ptr + d_offsets * stride_od, grad_k.to(k_vals.dtype), mask=mask)


# =====================================================================
# torch.autograd.Function — Fused Forward + Backward
# =====================================================================

class LucidPreconditionFunction(torch.autograd.Function):
    """
    Fused LUCID preconditioning with Triton-accelerated forward + backward.

    Forward:  Triton RMS norm → block-wise forward substitution (cuBLAS TRSM)
    Backward: block-wise backward substitution (Pᵀ solve) → Triton RMS norm bwd

    Saves K_RN and Y (not the full [T,T] matrix P) for backward — O(BHT D) memory.
    Deterministic and stateless — safe for ReversibleMidpointStack.
    """

    @staticmethod
    def forward(ctx, K, V, block_size):
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
        BH = K_flat.shape[0]

        # Step 1: RMS normalize keys (Triton if available)
        K_RN = torch.empty_like(K_flat)
        K_norms = torch.empty(BH, T, device=K.device, dtype=torch.float32)

        if HAS_TRITON and K.is_cuda:
            D_pow2 = triton.next_power_of_2(D)
            if D_pow2 <= 1024:
                grid = (BH, T)
                _lucid_rms_norm_fwd_kernel[grid](
                    K_flat, K_RN, K_norms,
                    T, D_pow2,
                    K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
                    K_RN.stride(0), K_RN.stride(1), K_RN.stride(2),
                    K_norms.stride(0),
                )
            else:
                K_RN = _rms_normalize_keys(K_flat)
                K_norms = K_flat.norm(dim=-1).clamp(min=1e-8)
        else:
            K_RN = _rms_normalize_keys(K_flat)
            K_norms = K_flat.norm(dim=-1).clamp(min=1e-8)

        # Step 2: Block-wise forward substitution
        Y_flat = _blockwise_forward_substitution(K_RN, V_flat, sqrt_d, block_size, K.device)

        # Save for backward (O(BH·T·D) each — no [T,T] matrices)
        ctx.save_for_backward(K_flat, K_RN, K_norms, Y_flat)
        ctx.block_size = block_size
        ctx.multi_head = multi_head
        ctx.sqrt_d = sqrt_d
        if multi_head:
            ctx.B, ctx.H = B, H

        # Reshape output
        if multi_head:
            Y = Y_flat.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()
        else:
            Y = Y_flat

        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        K_flat, K_RN, K_norms, Y_flat = ctx.saved_tensors
        block_size = ctx.block_size
        sqrt_d = ctx.sqrt_d
        multi_head = ctx.multi_head

        BH, T, D = K_RN.shape

        # Reshape grad_Y to [BH, T, D]
        if multi_head:
            B, H = ctx.B, ctx.H
            grad_Y_flat = grad_Y.permute(0, 2, 1, 3).reshape(BH, T, D).contiguous()
        else:
            grad_Y_flat = grad_Y.contiguous()

        # ── Step 1: dL/dV via backward substitution on Pᵀ ──
        # P·Y = V → dL/dV = P⁻ᵀ · dL/dY
        dV_flat = _blockwise_backward_substitution(
            K_RN, grad_Y_flat, sqrt_d, block_size, K_flat.device
        )

        # ── Step 2: dL/dK_RN via chain rule ──
        # P_ij = exp(K_RN_i · K_RN_j / √d − √d) for causal i >= j
        # dL/dK_RN comes from dL/dP · dP/dK_RN
        # dL/dP_ij = -dV_i · Y_j  (from the solve: Y = P⁻¹V)
        #            where dV = P⁻ᵀ grad_Y (already computed)
        # dP/dK_RN_i from row i: sum_j P_ij · (K_RN_j / √d)
        # Full: dL/dK_RN_i = (1/√d) · sum_j dL/dP_ij · P_ij · K_RN_j
        #                   = -(1/√d) · sum_j (dV_i · Y_j⊤) ⊙ P_ij · K_RN_j

        # Compute dL/dK_RN blockwise (same tiling as forward)
        # All computation in fp32 for numerical stability
        BS = min(block_size, T)
        K_RN_f = K_RN.float()
        Y_flat_f = Y_flat.float()
        dV_flat_f = dV_flat.float()
        dK_RN = torch.zeros(BH, T, D, device=K_flat.device, dtype=torch.float32)

        for i in range(0, T, BS):
            i_end = min(i + BS, T)
            K_RN_i = K_RN_f[:, i:i_end, :]
            dV_i = dV_flat_f[:, i:i_end, :]

            for j in range(0, i_end, BS):
                j_end = min(j + BS, T)
                K_RN_j = K_RN_f[:, j:j_end, :]
                Y_j = Y_flat_f[:, j:j_end, :]

                # Compute P_ij tile
                scores_ij = torch.bmm(K_RN_i, K_RN_j.transpose(-2, -1)) / sqrt_d - sqrt_d

                # Apply causal mask
                if i == j:
                    bs_i = i_end - i
                    bs_j = j_end - j
                    causal = torch.tril(torch.ones(bs_i, bs_j, device=K_flat.device, dtype=torch.bool))
                    scores_ij = scores_ij.masked_fill(~causal, float('-inf'))
                elif i < j:
                    continue  # upper triangle — skip

                P_ij = torch.exp(scores_ij)  # [BH, BS_i, BS_j]

                # dL/dP_ij = -(dV_i · Y_j⊤)
                dL_dP = -torch.bmm(dV_i, Y_j.transpose(-2, -1))  # [BH, BS_i, BS_j]

                # dL/dK_RN_i += (1/√d) · (dL_dP ⊙ P_ij) @ K_RN_j
                dK_RN[:, i:i_end, :] += (1.0 / sqrt_d) * torch.bmm(dL_dP * P_ij, K_RN_j)

                # dL/dK_RN_j contribution
                if j < i:  # off-diagonal
                    dK_RN[:, j:j_end, :] += (1.0 / sqrt_d) * torch.bmm((dL_dP * P_ij).transpose(-2, -1), K_RN_i)
                elif i == j:  # diagonal: j=i, both contribute to same slice
                    dK_RN[:, i:i_end, :] += (1.0 / sqrt_d) * torch.bmm((dL_dP * P_ij).transpose(-2, -1), K_RN_i)

        # ── Step 3: dL/dK from dL/dK_RN via chain rule through RMS norm ──
        if HAS_TRITON and K_flat.is_cuda:
            D_pow2 = triton.next_power_of_2(D)
            if D_pow2 <= 1024:
                grad_K = torch.empty_like(K_flat)
                grid = (BH, T)
                _lucid_rms_norm_bwd_kernel[grid](
                    dK_RN, grad_K,
                    K_flat, K_norms,
                    T, D_pow2,
                    dK_RN.stride(0), dK_RN.stride(1), dK_RN.stride(2),
                    K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
                    grad_K.stride(0), grad_K.stride(1), grad_K.stride(2),
                    K_norms.stride(0),
                )
            else:
                grad_K = _rms_norm_bwd_pytorch(dK_RN, K_flat, K_norms, D)
        else:
            grad_K = _rms_norm_bwd_pytorch(dK_RN, K_flat, K_norms, D)

        # Reshape outputs
        if multi_head:
            grad_K = grad_K.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()
            dV = dV_flat.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()
        else:
            dV = dV_flat

        return grad_K, dV, None


def _rms_norm_bwd_pytorch(dK_RN, K, K_norms, D):
    """PyTorch fallback for RMS norm backward."""
    sqrt_d = math.sqrt(D)
    inv_norm = 1.0 / (K_norms.unsqueeze(-1) + 1e-16)
    k_hat = K * inv_norm
    dot = (dK_RN * k_hat).sum(dim=-1, keepdim=True)
    return sqrt_d * inv_norm * (dK_RN - dot * k_hat)


# =====================================================================
# Triton-accelerated dispatch (forward uses Triton RMS + cuBLAS TRSM)
# =====================================================================

def triton_lucid_precondition(
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """
    LUCID preconditioning with Triton-fused RMS norm + cuBLAS block solver.
    No autograd — for inference / no-grad contexts.
    """
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
    BH = K_flat.shape[0]

    # Triton RMS norm
    K_RN = torch.empty_like(K_flat)
    K_norms = torch.empty(BH, T, device=K.device, dtype=torch.float32)
    D_pow2 = triton.next_power_of_2(D)

    if D_pow2 <= 1024:
        grid = (BH, T)
        _lucid_rms_norm_fwd_kernel[grid](
            K_flat, K_RN, K_norms,
            T, D_pow2,
            K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
            K_RN.stride(0), K_RN.stride(1), K_RN.stride(2),
            K_norms.stride(0),
        )
    else:
        K_RN = _rms_normalize_keys(K_flat)

    # Block solver
    Y_flat = _blockwise_forward_substitution(K_RN, V_flat, sqrt_d, block_size, K.device)

    if multi_head:
        return Y_flat.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()
    return Y_flat


# =====================================================================
# Dispatch function (main entry point for model)
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
    where P = causal ⊙ exp(K_RN · K_RN⊤ / √d − √d)

    Main entry point used by GatedSparseAttention.

    Args:
        K: [B, T, H, D] attention keys
        V: [B, T, H, D] gated values
        block_size: tile size for memory-efficient solver
        training: if True, use autograd Function for gradient flow

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
