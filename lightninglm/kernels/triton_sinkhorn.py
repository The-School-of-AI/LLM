"""
Triton Sinkhorn-Knopp (n=4) - Fused forward + Fused implicit backward (NO torch.linalg.solve)
===========================================================================================

This version is compatible with Triton builds that DISALLOW:
- tensor indexing with constexpr (x[0], A[k,k], A[k,:], etc.)
- defining Python helper functions inside @triton.jit kernels

Forward:
- 1 kernel per matrix batch
- computes Sinkhorn scalings u,v and output M

Backward:
- 1 kernel per matrix batch
- implicit differentiation at fixed point
- builds 8x8 system and solves via Gauss-Jordan using only tl.where + tl.sum
  (no sys[k,k] indexing)

Env flags:
- T17_SINKHORN_AUTOGRAD=1 (default): custom autograd (fwd+bwd)
- T17_SINKHORN_AUTOGRAD=0: forward-only Triton path
- T17_SINKHORN_DAMP=1e-7  : damping added to pivots (default max(eps,1e-7))

Requirements:
- n must be exactly 4
- CUDA tensor input
"""

from __future__ import annotations

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


# ============================================================
# Triton kernels
# ============================================================
if HAS_TRITON:

    @triton.jit
    def _sinkhorn4_fwd_save_uv_kernel(
        H_ptr,  # [num_mats, 16] input
        M_ptr,  # [num_mats, 16] fp32 output
        u_ptr,  # [num_mats, 4]  fp32 saved
        v_ptr,  # [num_mats, 4]  fp32 saved
        num_mats,  # runtime
        EPS: tl.constexpr,
        NUM_ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_mats:
            return

        offs = tl.arange(0, 16)
        base = pid * 16

        H = tl.load(H_ptr + base + offs).to(tl.float32)
        Kflat = tl.exp(H)
        K = tl.reshape(Kflat, (4, 4))

        v = tl.full((4,), 0.25, tl.float32)
        u = tl.full((4,), 1.0, tl.float32)

        for _ in tl.static_range(0, NUM_ITERS):
            v_row = tl.reshape(v, (1, 4))
            Kv = tl.sum(K * v_row, axis=1)
            u = 1.0 / (Kv + EPS)

            u_col = tl.reshape(u, (4, 1))
            KTu = tl.sum(K * u_col, axis=0)
            v = 1.0 / (KTu + EPS)

        u_col = tl.reshape(u, (4, 1))
        v_row = tl.reshape(v, (1, 4))
        M = (u_col * K) * v_row

        tl.store(M_ptr + base + offs, tl.reshape(M, (16,)))

        uv_base = pid * 4
        tl.store(u_ptr + uv_base + tl.arange(0, 4), u)
        tl.store(v_ptr + uv_base + tl.arange(0, 4), v)


if HAS_TRITON:

    @triton.jit
    def _sinkhorn4_implicit_bwd_solve8_kernel(
        H_ptr,  # [num_mats, 16]
        u_ptr,  # [num_mats, 4]
        v_ptr,  # [num_mats, 4]
        dM_ptr,  # [num_mats, 16]
        dH_ptr,  # [num_mats, 16] fp32
        num_mats,  # runtime
        EPS: tl.constexpr,
        DAMP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_mats:
            return

        offs16 = tl.arange(0, 16)
        base16 = pid * 16

        H = tl.load(H_ptr + base16 + offs16).to(tl.float32)
        Kflat = tl.exp(H)  # (16,)
        Gflat = tl.load(dM_ptr + base16 + offs16).to(tl.float32)

        K = tl.reshape(Kflat, (4, 4))
        G = tl.reshape(Gflat, (4, 4))

        base4 = pid * 4
        u = tl.load(u_ptr + base4 + tl.arange(0, 4)).to(tl.float32)  # (4,)
        v = tl.load(v_ptr + base4 + tl.arange(0, 4)).to(tl.float32)  # (4,)

        # ---- compute Kv, KTu, b1, b2
        v_row = tl.reshape(v, (1, 4))
        u_col = tl.reshape(u, (4, 1))

        Kv = tl.sum(K * v_row, axis=1) + EPS  # (4,)
        KTu = tl.sum(K * u_col, axis=0) + EPS  # (4,)

        A = G * K
        b1 = -tl.sum(A * v_row, axis=1)  # (4,)
        b2 = -tl.sum(A * u_col, axis=0)  # (4,)

        # ---- index helpers
        idx4 = tl.arange(0, 4)
        idx8 = tl.arange(0, 8)
        idx16 = tl.arange(0, 16)

        # ---- pick scalars (no helper funcs, no indexing)
        Kv0 = tl.sum(tl.where(idx4 == 0, Kv, 0.0), axis=0)
        Kv1 = tl.sum(tl.where(idx4 == 1, Kv, 0.0), axis=0)
        Kv2 = tl.sum(tl.where(idx4 == 2, Kv, 0.0), axis=0)
        Kv3 = tl.sum(tl.where(idx4 == 3, Kv, 0.0), axis=0)

        KTu0 = tl.sum(tl.where(idx4 == 0, KTu, 0.0), axis=0)
        KTu1 = tl.sum(tl.where(idx4 == 1, KTu, 0.0), axis=0)
        KTu2 = tl.sum(tl.where(idx4 == 2, KTu, 0.0), axis=0)
        KTu3 = tl.sum(tl.where(idx4 == 3, KTu, 0.0), axis=0)

        v0 = tl.sum(tl.where(idx4 == 0, v, 0.0), axis=0)
        v1 = tl.sum(tl.where(idx4 == 1, v, 0.0), axis=0)
        v2 = tl.sum(tl.where(idx4 == 2, v, 0.0), axis=0)
        v3 = tl.sum(tl.where(idx4 == 3, v, 0.0), axis=0)

        u0 = tl.sum(tl.where(idx4 == 0, u, 0.0), axis=0)
        u1 = tl.sum(tl.where(idx4 == 1, u, 0.0), axis=0)
        u2 = tl.sum(tl.where(idx4 == 2, u, 0.0), axis=0)
        u3 = tl.sum(tl.where(idx4 == 3, u, 0.0), axis=0)

        b10 = tl.sum(tl.where(idx4 == 0, b1, 0.0), axis=0)
        b11 = tl.sum(tl.where(idx4 == 1, b1, 0.0), axis=0)
        b12 = tl.sum(tl.where(idx4 == 2, b1, 0.0), axis=0)
        b13 = tl.sum(tl.where(idx4 == 3, b1, 0.0), axis=0)

        b20 = tl.sum(tl.where(idx4 == 0, b2, 0.0), axis=0)
        b21 = tl.sum(tl.where(idx4 == 1, b2, 0.0), axis=0)
        b22 = tl.sum(tl.where(idx4 == 2, b2, 0.0), axis=0)
        b23 = tl.sum(tl.where(idx4 == 3, b2, 0.0), axis=0)

        # K scalars from row-major Kflat
        K00 = tl.sum(tl.where(idx16 == 0, Kflat, 0.0), axis=0)
        K01 = tl.sum(tl.where(idx16 == 1, Kflat, 0.0), axis=0)
        K02 = tl.sum(tl.where(idx16 == 2, Kflat, 0.0), axis=0)
        K03 = tl.sum(tl.where(idx16 == 3, Kflat, 0.0), axis=0)

        K10 = tl.sum(tl.where(idx16 == 4, Kflat, 0.0), axis=0)
        K11 = tl.sum(tl.where(idx16 == 5, Kflat, 0.0), axis=0)
        K12 = tl.sum(tl.where(idx16 == 6, Kflat, 0.0), axis=0)
        K13 = tl.sum(tl.where(idx16 == 7, Kflat, 0.0), axis=0)

        K20 = tl.sum(tl.where(idx16 == 8, Kflat, 0.0), axis=0)
        K21 = tl.sum(tl.where(idx16 == 9, Kflat, 0.0), axis=0)
        K22 = tl.sum(tl.where(idx16 == 10, Kflat, 0.0), axis=0)
        K23 = tl.sum(tl.where(idx16 == 11, Kflat, 0.0), axis=0)

        K30 = tl.sum(tl.where(idx16 == 12, Kflat, 0.0), axis=0)
        K31 = tl.sum(tl.where(idx16 == 13, Kflat, 0.0), axis=0)
        K32 = tl.sum(tl.where(idx16 == 14, Kflat, 0.0), axis=0)
        K33 = tl.sum(tl.where(idx16 == 15, Kflat, 0.0), axis=0)

        # ---- build sys (8x8)
        r = tl.reshape(tl.arange(0, 8), (8, 1))  # (8,1)
        c = tl.reshape(tl.arange(0, 8), (1, 8))  # (1,8)

        sys = tl.zeros((8, 8), dtype=tl.float32)

        # TL diag(Kv)
        sys = tl.where((r == 0) & (c == 0), Kv0, sys)
        sys = tl.where((r == 1) & (c == 1), Kv1, sys)
        sys = tl.where((r == 2) & (c == 2), Kv2, sys)
        sys = tl.where((r == 3) & (c == 3), Kv3, sys)

        # TR (rows 0..3, cols 4..7)
        sys = tl.where((r == 0) & (c == 4), K00 * v0, sys)
        sys = tl.where((r == 0) & (c == 5), K01 * v1, sys)
        sys = tl.where((r == 0) & (c == 6), K02 * v2, sys)
        sys = tl.where((r == 0) & (c == 7), K03 * v3, sys)

        sys = tl.where((r == 1) & (c == 4), K10 * v0, sys)
        sys = tl.where((r == 1) & (c == 5), K11 * v1, sys)
        sys = tl.where((r == 1) & (c == 6), K12 * v2, sys)
        sys = tl.where((r == 1) & (c == 7), K13 * v3, sys)

        sys = tl.where((r == 2) & (c == 4), K20 * v0, sys)
        sys = tl.where((r == 2) & (c == 5), K21 * v1, sys)
        sys = tl.where((r == 2) & (c == 6), K22 * v2, sys)
        sys = tl.where((r == 2) & (c == 7), K23 * v3, sys)

        sys = tl.where((r == 3) & (c == 4), K30 * v0, sys)
        sys = tl.where((r == 3) & (c == 5), K31 * v1, sys)
        sys = tl.where((r == 3) & (c == 6), K32 * v2, sys)
        sys = tl.where((r == 3) & (c == 7), K33 * v3, sys)

        # BL (rows 4..7, cols 0..3)
        sys = tl.where((r == 4) & (c == 0), K00 * u0, sys)
        sys = tl.where((r == 4) & (c == 1), K10 * u1, sys)
        sys = tl.where((r == 4) & (c == 2), K20 * u2, sys)
        sys = tl.where((r == 4) & (c == 3), K30 * u3, sys)

        sys = tl.where((r == 5) & (c == 0), K01 * u0, sys)
        sys = tl.where((r == 5) & (c == 1), K11 * u1, sys)
        sys = tl.where((r == 5) & (c == 2), K21 * u2, sys)
        sys = tl.where((r == 5) & (c == 3), K31 * u3, sys)

        sys = tl.where((r == 6) & (c == 0), K02 * u0, sys)
        sys = tl.where((r == 6) & (c == 1), K12 * u1, sys)
        sys = tl.where((r == 6) & (c == 2), K22 * u2, sys)
        sys = tl.where((r == 6) & (c == 3), K32 * u3, sys)

        sys = tl.where((r == 7) & (c == 0), K03 * u0, sys)
        sys = tl.where((r == 7) & (c == 1), K13 * u1, sys)
        sys = tl.where((r == 7) & (c == 2), K23 * u2, sys)
        sys = tl.where((r == 7) & (c == 3), K33 * u3, sys)

        # BR diag(KTu) rows/cols 4..7
        sys = tl.where((r == 4) & (c == 4), KTu0, sys)
        sys = tl.where((r == 5) & (c == 5), KTu1, sys)
        sys = tl.where((r == 6) & (c == 6), KTu2, sys)
        sys = tl.where((r == 7) & (c == 7), KTu3, sys)

        rhs = tl.zeros((8,), dtype=tl.float32)
        rhs = tl.where(idx8 == 0, b10, rhs)
        rhs = tl.where(idx8 == 1, b11, rhs)
        rhs = tl.where(idx8 == 2, b12, rhs)
        rhs = tl.where(idx8 == 3, b13, rhs)
        rhs = tl.where(idx8 == 4, b20, rhs)
        rhs = tl.where(idx8 == 5, b21, rhs)
        rhs = tl.where(idx8 == 6, b22, rhs)
        rhs = tl.where(idx8 == 7, b23, rhs)

        # ---- Gauss-Jordan without any sys[k,k] style indexing
        damp = tl.maximum(DAMP, EPS)

        for k in tl.static_range(0, 8):
            # pivot = sys[k,k] via masking
            mask_p = (r == k) & (c == k)
            pivot_vec = tl.sum(tl.where(mask_p, sys, 0.0), axis=0)  # (8,)
            pivot = tl.sum(pivot_vec, axis=0) + damp  # scalar
            invp = 1.0 / pivot

            # row_k = sys[k,:]
            row_k = tl.sum(tl.where(r == k, sys, 0.0), axis=0)  # (8,)
            row_k = row_k * invp

            # rhs_k = rhs[k]
            rhs_k = tl.sum(tl.where(idx8 == k, rhs, 0.0), axis=0) * invp

            # write normalized pivot row
            sys = tl.where(r == k, tl.reshape(row_k, (1, 8)), sys)
            rhs = tl.where(idx8 == k, rhs_k, rhs)

            # col_k = sys[:,k]
            col_k = tl.sum(tl.where(c == k, sys, 0.0), axis=1)  # (8,)
            col_k = tl.where(idx8 == k, 0.0, col_k)

            # eliminate
            sys = sys - tl.reshape(col_k, (8, 1)) * tl.reshape(row_k, (1, 8))
            rhs = rhs - col_k * rhs_k

        # lam/mu from rhs using masked reductions
        lam0 = tl.sum(tl.where(idx8 == 0, rhs, 0.0), axis=0)
        lam1 = tl.sum(tl.where(idx8 == 1, rhs, 0.0), axis=0)
        lam2 = tl.sum(tl.where(idx8 == 2, rhs, 0.0), axis=0)
        lam3 = tl.sum(tl.where(idx8 == 3, rhs, 0.0), axis=0)

        mu0 = tl.sum(tl.where(idx8 == 4, rhs, 0.0), axis=0)
        mu1 = tl.sum(tl.where(idx8 == 5, rhs, 0.0), axis=0)
        mu2 = tl.sum(tl.where(idx8 == 6, rhs, 0.0), axis=0)
        mu3 = tl.sum(tl.where(idx8 == 7, rhs, 0.0), axis=0)

        lam = tl.zeros((4,), dtype=tl.float32)
        lam = tl.where(idx4 == 0, lam0, lam)
        lam = tl.where(idx4 == 1, lam1, lam)
        lam = tl.where(idx4 == 2, lam2, lam)
        lam = tl.where(idx4 == 3, lam3, lam)

        mu = tl.zeros((4,), dtype=tl.float32)
        mu = tl.where(idx4 == 0, mu0, mu)
        mu = tl.where(idx4 == 1, mu1, mu)
        mu = tl.where(idx4 == 2, mu2, mu)
        mu = tl.where(idx4 == 3, mu3, mu)

        # dK = (u_i v_j) * (G_ij + lam_i + mu_j)
        lam_col = tl.reshape(lam, (4, 1))
        mu_row = tl.reshape(mu, (1, 4))
        corr = G + lam_col + mu_row

        u_col = tl.reshape(u, (4, 1))
        v_row = tl.reshape(v, (1, 4))
        dK = (u_col * v_row) * corr
        dH = dK * K

        tl.store(dH_ptr + base16 + offs16, tl.reshape(dH, (16,)))


# ============================================================
# Python wrappers + autograd
# ============================================================
def _triton_fwd_save_uv(H: torch.Tensor, num_iters: int, eps: float):
    if not HAS_TRITON:
        raise ImportError("Triton is required.")
    if not H.is_cuda:
        raise ValueError("H must be CUDA.")
    if H.shape[-2:] != (4, 4):
        raise ValueError(f"Only n=4 supported. Got {H.shape[-2:]}")

    orig_shape = H.shape
    H_flat = H.reshape(-1, 16).contiguous()
    num_mats = H_flat.shape[0]

    M_flat = torch.empty((num_mats, 16), device=H.device, dtype=torch.float32)
    u = torch.empty((num_mats, 4), device=H.device, dtype=torch.float32)
    v = torch.empty((num_mats, 4), device=H.device, dtype=torch.float32)

    _sinkhorn4_fwd_save_uv_kernel[(num_mats,)](
        H_flat,
        M_flat,
        u,
        v,
        num_mats,
        EPS=float(eps),
        NUM_ITERS=int(num_iters),
        num_warps=1,
        num_stages=1,
    )
    M = M_flat.reshape(orig_shape).to(H.dtype)
    return M, u, v


def _triton_bwd_solve8(
    H: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
):
    if not HAS_TRITON:
        raise ImportError("Triton is required.")
    orig_shape = H.shape
    H_flat = H.reshape(-1, 16).contiguous()
    dM_flat = grad_out.reshape(-1, 16).contiguous()
    num_mats = H_flat.shape[0]

    dH_flat = torch.empty((num_mats, 16), device=H.device, dtype=torch.float32)

    damp_env = os.getenv("T17_SINKHORN_DAMP", "").strip()
    damp = float(damp_env) if damp_env else max(float(eps), 1e-7)

    _sinkhorn4_implicit_bwd_solve8_kernel[(num_mats,)](
        H_flat,
        u,
        v,
        dM_flat,
        dH_flat,
        num_mats,
        EPS=float(eps),
        DAMP=float(damp),
        num_warps=1,
        num_stages=1,
    )
    return dH_flat.reshape(orig_shape).to(H.dtype)


def pytorch_sinkhorn_knopp(
    H: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    M = torch.exp(H)
    for _ in range(num_iters):
        M = M / (M.sum(dim=-1, keepdim=True) + eps)
        M = M / (M.sum(dim=-2, keepdim=True) + eps)
    return M


class _TritonSinkhorn4ImplicitSolve8Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H: torch.Tensor, num_iters: int, eps: float):
        M, u, v = _triton_fwd_save_uv(H, int(num_iters), float(eps))
        ctx.eps = float(eps)
        ctx.save_for_backward(H, u, v)
        return M

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        H, u, v = ctx.saved_tensors
        dH = _triton_bwd_solve8(H, u, v, grad_out, eps=ctx.eps)
        return dH, None, None


def triton_sinkhorn_knopp(
    H: torch.Tensor, num_iters: int = 20, eps: float = 1e-8
) -> torch.Tensor:
    if not HAS_TRITON:
        raise ImportError("Triton is required.")
    if H.shape[-2:] != (4, 4):
        raise ValueError(f"Only n=4 supported. Got {H.shape[-2:]}")

    use_autograd = os.getenv("T17_SINKHORN_AUTOGRAD", "1") == "1"
    if use_autograd and H.requires_grad and torch.is_grad_enabled():
        return _TritonSinkhorn4ImplicitSolve8Fn.apply(H, int(num_iters), float(eps))

    M, _, _ = _triton_fwd_save_uv(H, int(num_iters), float(eps))
    return M
