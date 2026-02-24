#!/usr/bin/env python3
"""
Optimized Triton kernels for the 1B reversible model.

Drop-in replacements for performance-critical operations:
1. fused_rope_triton      — RoPE rotation (1 kernel vs 4+ PyTorch ops)
2. fused_silu_mul_triton  — SiLU * element-wise multiply (1 kernel vs 2)
3. fused_rmsnorm_swishgate — RMSNorm + SiLU + gate (1 kernel vs 3 ops)
4. AutotunedRMSNormFn     — RMSNorm with @triton.autotune
5. BatchedMHCCoeffs       — 3 small linears merged into 1 matmul

All kernels support bf16 inputs with fp32 internal computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl


# ════════════════════════════════════════════════════════════════════════
# 1. FUSED ROPE KERNEL
# ════════════════════════════════════════════════════════════════════════
# Replaces liger_rotary_pos_emb:
#   4+ PyTorch ops (even/odd slice, 2×multiply, 2×add, stack, reshape)
#   → 1 Triton kernel with interleaved load/store.

@triton.jit
def _fused_rope_kernel(
    X_ptr, COS_ptr, SIN_ptr, OUT_ptr,
    D: tl.constexpr,
    stride_x_row,
    stride_cs_row,
    T, H,
    HALF_D: tl.constexpr,
):
    row_idx = tl.program_id(0)
    # row_idx = b*T*H + t*H + h → t = (row_idx // H) % T
    t_idx = (row_idx // H) % T

    d_offs = tl.arange(0, HALF_D)
    mask = d_offs < (D // 2)

    x_base = X_ptr + row_idx * stride_x_row
    x_even = tl.load(x_base + 2 * d_offs, mask=mask, other=0.0)
    x_odd = tl.load(x_base + 2 * d_offs + 1, mask=mask, other=0.0)

    cs_base = COS_ptr + t_idx * stride_cs_row
    ss_base = SIN_ptr + t_idx * stride_cs_row
    cos_h = tl.load(cs_base + 2 * d_offs, mask=mask, other=1.0)
    sin_h = tl.load(ss_base + 2 * d_offs, mask=mask, other=0.0)

    rot_even = x_even * cos_h - x_odd * sin_h
    rot_odd = x_even * sin_h + x_odd * cos_h

    out_base = OUT_ptr + row_idx * stride_x_row
    tl.store(out_base + 2 * d_offs, rot_even, mask=mask)
    tl.store(out_base + 2 * d_offs + 1, rot_odd, mask=mask)


class FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, T, H):
        D = x.shape[-1]
        x_2d = x.reshape(-1, D)
        n_rows = x_2d.shape[0]
        # cos/sin: [1, T, 1, D] → [T, D]
        cos_2d = cos.squeeze(0).squeeze(-2).contiguous()
        sin_2d = sin.squeeze(0).squeeze(-2).contiguous()

        out = torch.empty_like(x_2d)
        HALF_D = triton.next_power_of_2(D // 2)
        nw = 4 if HALF_D <= 64 else 8

        _fused_rope_kernel[(n_rows,)](
            x_2d, cos_2d, sin_2d, out,
            D, x_2d.stride(0), cos_2d.stride(0), T, H,
            HALF_D=HALF_D, num_warps=nw,
        )
        ctx.save_for_backward(cos_2d, sin_2d)
        ctx.shape = x.shape
        ctx.T = T
        ctx.H = H
        return out.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output):
        cos_2d, sin_2d = ctx.saved_tensors
        D = grad_output.shape[-1]
        g_2d = grad_output.reshape(-1, D)
        n_rows = g_2d.shape[0]
        out = torch.empty_like(g_2d)
        HALF_D = triton.next_power_of_2(D // 2)
        nw = 4 if HALF_D <= 64 else 8

        # RoPE backward is the inverse rotation: rotate(grad, cos, -sin)
        neg_sin = -sin_2d
        _fused_rope_kernel[(n_rows,)](
            g_2d, cos_2d, neg_sin, out,
            D, g_2d.stride(0), cos_2d.stride(0), ctx.T, ctx.H,
            HALF_D=HALF_D, num_warps=nw,
        )
        return out.reshape(ctx.shape), None, None, None, None


def fused_rope_triton(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for liger_rotary_pos_emb."""
    assert x.dim() in (3, 4), f"Expected 3D or 4D, got {x.dim()}D"
    D = x.shape[-1]
    assert D % 2 == 0, f"RoPE head dim must be even, got {D}"

    if x.dim() == 4:
        B, T, H, _ = x.shape
    else:
        B, T, _ = x.shape
        H = 1
    return FusedRoPEFunction.apply(x, cos, sin, T, H)


# ════════════════════════════════════════════════════════════════════════
# 2. FUSED SILU * MUL KERNEL
# ════════════════════════════════════════════════════════════════════════
# Replaces F.silu(gate) * up: 2 PyTorch kernels → 1 Triton kernel.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_silu_mul_fwd_kernel(
    GATE_ptr, UP_ptr, OUT_ptr, SIG_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0)
    up = tl.load(UP_ptr + offs, mask=mask, other=0.0)

    gate_f32 = gate.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate_f32))
    silu_val = gate_f32 * sig
    result = silu_val.to(gate.dtype) * up

    tl.store(OUT_ptr + offs, result, mask=mask)
    # Save sigmoid for backward
    tl.store(SIG_ptr + offs, sig.to(gate.dtype), mask=mask)


@triton.jit
def _fused_silu_mul_bwd_kernel(
    GRAD_ptr, GATE_ptr, UP_ptr, SIG_ptr,
    D_GATE_ptr, D_UP_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    grad = tl.load(GRAD_ptr + offs, mask=mask, other=0.0)
    gate = tl.load(GATE_ptr + offs, mask=mask, other=0.0)
    up = tl.load(UP_ptr + offs, mask=mask, other=0.0)
    sig = tl.load(SIG_ptr + offs, mask=mask, other=0.0)

    gate_f32 = gate.to(tl.float32)
    sig_f32 = sig.to(tl.float32)
    grad_f32 = grad.to(tl.float32)
    up_f32 = up.to(tl.float32)

    # d(silu(g)*u)/dg = u * sig * (1 + g*(1-sig))
    d_gate = up_f32 * sig_f32 * (1.0 + gate_f32 * (1.0 - sig_f32)) * grad_f32
    # d(silu(g)*u)/du = g * sig = silu(g)
    d_up = gate_f32 * sig_f32 * grad_f32

    tl.store(D_GATE_ptr + offs, d_gate.to(gate.dtype), mask=mask)
    tl.store(D_UP_ptr + offs, d_up.to(up.dtype), mask=mask)


class FusedSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        assert gate.shape == up.shape
        out = torch.empty_like(gate)
        sig = torch.empty_like(gate)
        n = gate.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)

        _fused_silu_mul_fwd_kernel[grid](
            gate, up, out, sig, n,
        )
        ctx.save_for_backward(gate, up, sig)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gate, up, sig = ctx.saved_tensors
        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        n = gate.numel()
        BLOCK = 4096
        grid = (triton.cdiv(n, BLOCK),)

        _fused_silu_mul_bwd_kernel[grid](
            grad_output, gate, up, sig, d_gate, d_up, n,
            BLOCK=BLOCK, num_warps=8,
        )
        return d_gate, d_up


def fused_silu_mul_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for liger_silu_mul."""
    return FusedSiLUMulFunction.apply(gate, up)


# ════════════════════════════════════════════════════════════════════════
# 3. FUSED RMSNORM + SWISHGATE KERNEL
# ════════════════════════════════════════════════════════════════════════
# Replaces FusedRMSNormSwishGate: norm(x) → silu → gate → 3 ops → 1 kernel.

@triton.jit
def _fused_rmsnorm_swishgate_fwd_kernel(
    X_ptr, G_ptr, W_ptr, OUT_ptr, RSTD_ptr,
    n_cols,
    eps,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    x_base = X_ptr + row_idx * stride_row
    g_base = G_ptr + row_idx * stride_row

    x = tl.load(x_base + col_offs, mask=mask, other=0.0)
    g = tl.load(g_base + col_offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)

    # RMSNorm in fp32
    x_f32 = x.to(tl.float32)
    ms = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(RSTD_ptr + row_idx, rstd)

    normed = (x_f32 * rstd).to(x.dtype)
    normed_w = normed * w

    # SiLU + gate
    nw_f32 = normed_w.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-nw_f32))
    silu_nw = (nw_f32 * sig).to(x.dtype)

    result = g * silu_nw

    out_base = OUT_ptr + row_idx * stride_row
    tl.store(out_base + col_offs, result, mask=mask)


@triton.jit
def _fused_rmsnorm_swishgate_bwd_kernel(
    GRAD_ptr, X_ptr, G_ptr, W_ptr, RSTD_ptr,
    DX_ptr, DG_ptr, DW_ptr,
    n_rows, n_cols,
    stride_row,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0).to(tl.int64)
    row_start = block_id * rows_per_program
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
    dw_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Process rows assigned to this block (guard out-of-range rows via mask)
    for row_idx in range(row_start, row_start + rows_per_program):
        # Combine col mask with row validity (Triton has no break)
        row_mask = mask & (row_idx < n_rows)

        base = row_idx * stride_row
        grad_out = tl.load(GRAD_ptr + base + col_offs, mask=row_mask, other=0.0)
        x = tl.load(X_ptr + base + col_offs, mask=row_mask, other=0.0)
        g = tl.load(G_ptr + base + col_offs, mask=row_mask, other=0.0)
        rstd = tl.load(RSTD_ptr + row_idx, mask=row_idx < n_rows, other=1.0)

        x_f32 = x.to(tl.float32)
        normed = x_f32 * rstd
        normed_w = normed * w.to(tl.float32)

        sig = 1.0 / (1.0 + tl.exp(-normed_w))
        silu_nw = normed_w * sig

        # grad_g = grad_out * silu(normed_w)
        grad_g = grad_out.to(tl.float32) * silu_nw
        tl.store(DG_ptr + base + col_offs, grad_g.to(grad_out.dtype), mask=row_mask)

        # grad through silu: dsilu/dx = sig*(1 + x*(1-sig))
        grad_silu = grad_out.to(tl.float32) * g.to(tl.float32)
        grad_nw = grad_silu * sig * (1.0 + normed_w * (1.0 - sig))

        # grad_w accumulated (zeroed contributions from invalid rows via masking)
        dw_acc += grad_nw * normed

        # grad through rmsnorm
        m = grad_nw * w.to(tl.float32)
        dot_mx = tl.sum(m * x_f32, axis=0)
        dx = rstd * (m - (1.0 / n_cols) * rstd * rstd * dot_mx * x_f32)
        tl.store(DX_ptr + base + col_offs, dx.to(x.dtype), mask=row_mask)

    dw_base = DW_ptr + block_id * n_cols
    tl.store(dw_base + col_offs, dw_acc, mask=mask)


class FusedRMSNormSwishGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, weight, eps):
        orig_shape = x.shape
        n_cols = x.shape[-1]
        x_2d = x.contiguous().reshape(-1, n_cols)
        g_2d = g.contiguous().reshape(-1, n_cols)
        n_rows = x_2d.shape[0]

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)
        nw = 4
        if BLOCK_SIZE >= 2048:
            nw = 8
        if BLOCK_SIZE >= 8192:
            nw = 16

        out = torch.empty_like(x_2d)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        _fused_rmsnorm_swishgate_fwd_kernel[(n_rows,)](
            x_2d, g_2d, weight, out, rstd,
            n_cols, eps, x_2d.stride(0),
            BLOCK_SIZE=BLOCK_SIZE, num_warps=nw,
        )

        ctx.save_for_backward(x_2d, g_2d, weight, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = nw
        ctx.n_rows = n_rows
        ctx.n_cols = n_cols
        ctx.orig_shape = orig_shape
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x_2d, g_2d, weight, rstd = ctx.saved_tensors
        n_rows = ctx.n_rows
        n_cols = ctx.n_cols
        BLOCK_SIZE = ctx.BLOCK_SIZE
        nw = ctx.num_warps

        grad_2d = grad_output.contiguous().reshape(-1, n_cols)
        dx = torch.empty_like(x_2d)
        dg = torch.empty_like(g_2d)

        sm_count = torch.cuda.get_device_properties(x_2d.device).multi_processor_count
        dw_partial = torch.empty(sm_count, n_cols, dtype=torch.float32, device=weight.device)
        rows_per_program = math.ceil(n_rows / sm_count)

        _fused_rmsnorm_swishgate_bwd_kernel[(sm_count,)](
            grad_2d, x_2d, g_2d, weight, rstd,
            dx, dg, dw_partial,
            n_rows, n_cols, x_2d.stride(0), rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=nw,
        )
        dw = dw_partial.sum(dim=0).to(weight.dtype)
        return dx.reshape(ctx.orig_shape), dg.reshape(ctx.orig_shape), dw, None


def fused_rmsnorm_swishgate(x, g, weight, eps=1e-6):
    """Drop-in replacement for FusedRMSNormSwishGate.forward."""
    return FusedRMSNormSwishGateFunction.apply(x, g, weight, eps)


# ════════════════════════════════════════════════════════════════════════
# 4. AUTOTUNED RMSNORM
# ════════════════════════════════════════════════════════════════════════
# Adds @triton.autotune to find optimal (num_warps, num_stages) per dim.

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=3),
    ],
    key=['n_cols', 'BLOCK_SIZE'],
)
@triton.jit
def _autotuned_rmsnorm_fwd_kernel(
    Y_ptr, X_ptr, W_ptr, RSTD_ptr,
    n_cols, eps,
    stride_x_row, stride_y_row,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    x = tl.load(X_ptr + row_idx * stride_x_row + col_offs, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    ms = tl.sum(x_f32 * x_f32, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(ms + eps)
    tl.store(RSTD_ptr + row_idx, rstd)

    normed = (x_f32 * rstd).to(x.dtype)
    w = tl.load(W_ptr + col_offs, mask=mask, other=1.0)
    out = normed * w

    tl.store(Y_ptr + row_idx * stride_y_row + col_offs, out, mask=mask)


class AutotunedRMSNormFunction(torch.autograd.Function):
    """RMSNorm with autotuned forward kernel."""

    @staticmethod
    def forward(ctx, X, W, eps):
        orig_shape = X.shape
        n_cols = X.shape[-1]
        X_2d = X.contiguous().reshape(-1, n_cols)
        n_rows = X_2d.shape[0]

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        BLOCK_SIZE = max(BLOCK_SIZE, 128)

        Y = torch.empty_like(X_2d)
        RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        _autotuned_rmsnorm_fwd_kernel[(n_rows,)](
            Y, X_2d, W, RSTD,
            n_cols, eps,
            X_2d.stride(0), Y.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.save_for_backward(X_2d, W, RSTD)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.n_rows = n_rows
        ctx.n_cols = n_cols
        ctx.orig_shape = orig_shape
        return Y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, dY):
        # Reuse existing backward kernel (already efficient)
        from src.kernels.triton_rmsnorm import _rmsnorm_bwd_kernel
        X_2d, W, RSTD = ctx.saved_tensors
        n_rows = ctx.n_rows
        n_cols = ctx.n_cols
        BLOCK_SIZE = ctx.BLOCK_SIZE

        dY_2d = dY.contiguous().reshape(-1, n_cols)
        dX = torch.empty_like(X_2d)
        sm_count = torch.cuda.get_device_properties(X_2d.device).multi_processor_count
        _dW = torch.empty(sm_count, n_cols, dtype=torch.float32, device=W.device)
        rows_per_program = math.ceil(n_rows / sm_count)

        nw = 4
        if BLOCK_SIZE >= 2048:
            nw = 8
        if BLOCK_SIZE >= 8192:
            nw = 16

        _rmsnorm_bwd_kernel[(sm_count,)](
            dY_2d, dX, X_2d, W, RSTD, _dW,
            n_rows, n_cols,
            dY_2d.stride(0), dX.stride(0), X_2d.stride(0),
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=nw,
        )
        dW = _dW.sum(dim=0).to(W.dtype)
        return dX.reshape(ctx.orig_shape), dW, None


def autotuned_rmsnorm(x, weight, eps=1e-6):
    """Drop-in replacement for triton_rmsnorm with autotune."""
    return AutotunedRMSNormFunction.apply(x, weight, eps)


# ════════════════════════════════════════════════════════════════════════
# 5. BATCHED MHCCoeffs PROJECTIONS
# ════════════════════════════════════════════════════════════════════════
# Replaces 3 separate small Linear layers (out_dims 4, 4, 16) with 1.
# Reduces cuBLAS launch overhead 3x for tiny matmuls.

class BatchedMHCCoeffsProjection(nn.Module):
    """Single linear layer replacing phi_pre + phi_post + phi_res."""

    def __init__(self, d_in, n_streams):
        super().__init__()
        self.n = n_streams
        # phi_pre: [d_in, n], phi_post: [d_in, n], phi_res: [d_in, n*n]
        self.total_out = n_streams + n_streams + n_streams * n_streams
        self.proj = nn.Linear(d_in, self.total_out, bias=False)

    @classmethod
    def from_separate(cls, phi_pre, phi_post, phi_res, n_streams):
        """Initialize from existing separate Linear layers."""
        d_in = phi_pre.in_features
        obj = cls(d_in, n_streams)
        # Stack weights: [total_out, d_in]
        with torch.no_grad():
            obj.proj.weight.copy_(torch.cat([
                phi_pre.weight,   # [n, d_in]
                phi_post.weight,  # [n, d_in]
                phi_res.weight,   # [n*n, d_in]
            ], dim=0))
        return obj

    def forward(self, x):
        combined = self.proj(x)
        n = self.n
        pre = combined[..., :n]
        post = combined[..., n:2*n]
        res = combined[..., 2*n:]
        return pre, post, res


# ════════════════════════════════════════════════════════════════════════
# 6. MODEL PATCHING UTILITIES
# ════════════════════════════════════════════════════════════════════════

def patch_rope(model):
    """Replace PyTorch RoPE with fused Triton RoPE on all RotaryEmbedding instances."""
    patched = 0
    for layer in model.layers:
        attn = layer.attn_block.sublayer
        if hasattr(attn, 'rotary_emb'):
            original_apply = attn.rotary_emb._apply_rotary

            def _new_apply(x, cos, sin, _orig=original_apply):
                try:
                    return fused_rope_triton(x, cos, sin)
                except Exception:
                    return _orig(x, cos, sin)

            attn.rotary_emb._apply_rotary = _new_apply
            patched += 1
    return patched


def patch_silu_mul(liger_ops_module):
    """Replace PyTorch SiLU*mul with fused Triton kernel."""
    liger_ops_module.liger_silu_mul = fused_silu_mul_triton


def patch_rmsnorm_swishgate(model):
    """Replace FusedRMSNormSwishGate.forward with fused Triton kernel."""
    patched = 0
    for layer in model.layers:
        attn = layer.attn_block.sublayer
        if hasattr(attn, 'o_norm'):
            norm_module = attn.o_norm
            weight = norm_module.norm.weight
            eps = norm_module.norm.eps

            def _new_forward(x, g, _w=weight, _eps=eps):
                return fused_rmsnorm_swishgate(x, g, _w, _eps)

            norm_module.forward = _new_forward
            patched += 1
    return patched


def patch_mhc_coeffs(model):
    """Replace 3 separate MHCCoeffs projections with batched version."""
    patched = 0
    for module in model.modules():
        if type(module).__name__ == 'MHCCoeffs':
            param = next(module.parameters())
            batched = BatchedMHCCoeffsProjection.from_separate(
                module.phi_pre, module.phi_post, module.phi_res, module.n
            ).to(device=param.device, dtype=param.dtype)

            # Replace the forward to use batched projection
            original_forward = module.forward

            def _new_forward(x_stream, _m=module, _b=batched):
                B, T, n, D = x_stream.shape
                x_flat = x_stream.reshape(B, T, n * D)
                x_flat = _m.rms(x_flat)
                x_flat = x_flat.to(_m.phi_pre.weight.dtype)

                pre_raw, post_raw, res_raw = _b(x_flat)
                pre_logits = _m.alpha_pre * pre_raw + _m.b_pre
                post_logits = _m.alpha_post * post_raw + _m.b_post
                res_logits = _m.alpha_res * res_raw
                res_logits = res_logits.view(B, T, n, n) + _m.b_res

                H_pre = torch.sigmoid(pre_logits)
                H_post = 2.0 * torch.sigmoid(post_logits)

                # Import sinkhorn from model module
                from src.models.recurrence_model_1b import sinkhorn_knopp
                H_res = sinkhorn_knopp(res_logits, iters=_m.iters)
                return H_pre, H_post, H_res

            module.forward = _new_forward
            patched += 1
    return patched


def patch_reduce_sync(train_module):
    """
    Remove unnecessary cuda.synchronize() after non_blocking data transfers.
    The sync is redundant because subsequent CUDA operations will naturally wait.
    """
    # This is applied at the training loop level, not model level.
    # Documented here for the benchmark to demonstrate impact.
    pass


def apply_all_optimizations(model, liger_ops_module=None):
    """Apply all kernel optimizations to the model."""
    results = {}
    results['rope_patched'] = patch_rope(model)
    results['rmsnorm_swishgate_patched'] = patch_rmsnorm_swishgate(model)
    results['mhc_coeffs_patched'] = patch_mhc_coeffs(model)
    if liger_ops_module is not None:
        patch_silu_mul(liger_ops_module)
        results['silu_mul_patched'] = True
    return results
