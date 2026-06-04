"""
Triton Fused XSA (Exclusive Self-Attention) Projection Kernel
==============================================================

Orthogonal projection from XSA paper (arXiv:2603.09078v1):
    z_i = y_i - (dot(y_i, v_i) / dot(v_i, v_i)) * v_i

Removes the component of each token's attention output that is parallel
to its own value vector, forcing attention to capture only cross-token
information (not point-wise features redundant with the FFN).

Forward:  reads Y, V -> writes Z        (3 * B*T*H*D elements)
Backward: reads dZ, Y, V -> writes dY, dV  (5 * B*T*H*D elements)
Both memory-bound on L4 (300 GB/s).
"""

import torch
import triton
import triton.language as tl

# ═══════════════════════════════════════════════════════════════════════
# Triton Forward Kernel
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _xsa_fwd_kernel(
    Z_ptr,
    Y_ptr,
    V_ptr,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    z = y - (y·v / v·v) * v, one program per (b, t, h).
    Math in fp32 (or fp64 for gradcheck), output cast to input dtype.
    """
    pid = tl.program_id(0)
    base = pid * D
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    acc_type = tl.float64 if USE_FP64 else tl.float32
    y = tl.load(Y_ptr + base + offs, mask=mask, other=0.0).to(acc_type)
    v = tl.load(V_ptr + base + offs, mask=mask, other=0.0).to(acc_type)

    s = tl.sum(y * v, axis=0)  # dot(y, v)
    n = tl.sum(v * v, axis=0)  # dot(v, v)
    c = s / tl.maximum(n, eps)  # projection coefficient

    z = y - c * v
    tl.store(Z_ptr + base + offs, z.to(Z_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Triton Backward Kernel
# ═══════════════════════════════════════════════════════════════════════


@triton.jit
def _xsa_bwd_kernel(
    DY_ptr,
    DV_ptr,
    DZ_ptr,
    Y_ptr,
    V_ptr,
    D,
    eps,
    BLOCK_D: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """
    Backward for z = y - (y·v / v·v) * v.

    Let c = s/n, s = dot(y,v), n = dot(v,v), p = dot(dZ,v)/n:
        dY = dZ - p * v
        dV = -c * dZ - p * y + 2*c*p * v
    """
    pid = tl.program_id(0)
    base = pid * D
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D

    acc_type = tl.float64 if USE_FP64 else tl.float32
    dz = tl.load(DZ_ptr + base + offs, mask=mask, other=0.0).to(acc_type)
    y = tl.load(Y_ptr + base + offs, mask=mask, other=0.0).to(acc_type)
    v = tl.load(V_ptr + base + offs, mask=mask, other=0.0).to(acc_type)

    s = tl.sum(y * v, axis=0)  # dot(y, v)
    n = tl.sum(v * v, axis=0)  # dot(v, v)
    t = tl.sum(dz * v, axis=0)  # dot(dZ, v)
    n_safe = tl.maximum(n, eps)

    c = s / n_safe  # projection coefficient
    p = t / n_safe  # gradient projection coefficient

    dy = dz - p * v
    dv = -c * dz - p * y + 2.0 * c * p * v

    tl.store(DY_ptr + base + offs, dy.to(DY_ptr.dtype.element_ty), mask=mask)
    tl.store(DV_ptr + base + offs, dv.to(DV_ptr.dtype.element_ty), mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Autograd Function
# ═══════════════════════════════════════════════════════════════════════


class XSAProjectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, V, eps=1e-6):
        assert Y.shape == V.shape, f"Shape mismatch: Y={Y.shape}, V={V.shape}"
        Y, V = Y.contiguous(), V.contiguous()
        D = Y.shape[-1]
        N = Y.numel() // D
        BLOCK_D = triton.next_power_of_2(D)
        num_warps = 4 if BLOCK_D <= 256 else 8
        use_fp64 = Y.dtype == torch.float64

        Z = torch.empty_like(Y)
        _xsa_fwd_kernel[(N,)](
            Z, Y, V, D, eps, BLOCK_D=BLOCK_D, num_warps=num_warps, USE_FP64=use_fp64
        )
        ctx.save_for_backward(Y, V)
        ctx.eps = eps
        ctx.use_fp64 = use_fp64
        return Z

    @staticmethod
    def backward(ctx, dZ):
        Y, V = ctx.saved_tensors
        dZ = dZ.contiguous()
        D = Y.shape[-1]
        N = Y.numel() // D
        BLOCK_D = triton.next_power_of_2(D)
        num_warps = 4 if BLOCK_D <= 256 else 8

        dY = torch.empty_like(Y)
        dV = torch.empty_like(V)
        _xsa_bwd_kernel[(N,)](
            dY,
            dV,
            dZ,
            Y,
            V,
            D,
            ctx.eps,
            BLOCK_D=BLOCK_D,
            num_warps=num_warps,
            USE_FP64=ctx.use_fp64,
        )
        return dY, dV, None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def xsa_projection(Y, V, eps=1e-6):
    """
    XSA orthogonal projection: z = y - (y·v / v·v) * v per token per head.

    Args:
        Y: Attention output [..., D] (bf16/fp16/fp32)
        V: Value vectors [..., D] (same dtype/shape as Y)
        eps: Denominator clamp for degenerate v (default 1e-6)

    Returns:
        Z: Projected output [..., D], orthogonal to V
    """
    return XSAProjectionFn.apply(Y, V, eps)
