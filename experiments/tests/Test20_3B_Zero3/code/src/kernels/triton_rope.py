"""
Triton Fused RoPE (Rotary Position Embedding) Kernel — Forward AND Backward
=============================================================================

Fuses the rotary embedding application into a single Triton kernel per direction.

Standard PyTorch:
    x_even, x_odd = x[..., 0::2], x[..., 1::2]   # 2 slices (views, cheap)
    rot_even = x_even * cos - x_odd * sin           # 2 muls + 1 sub
    rot_odd = x_even * sin + x_odd * cos             # 2 muls + 1 add
    out = torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x)  # stack+reshape
    => multiple intermediate tensors from stack + reshape

Fused:
    For each pair (even, odd) in the last dim:
        out[even] = x[even] * cos - x[odd] * sin
        out[odd]  = x[even] * sin + x[odd] * cos
    => 0 intermediate tensors; reads x, cos, sin once, writes out once

Backward is the inverse rotation:
    d_x[even] = grad[even] * cos + grad[odd] * sin
    d_x[odd]  = -grad[even] * sin + grad[odd] * cos

Attribution:
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel (Apache-2.0)
- Unsloth: https://github.com/unslothai/unsloth (Apache-2.0)
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


# ═══════════════════════════════════════════════════════════════════════
# Triton Kernel (applies rotation to interleaved even/odd pairs)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _rope_fwd_kernel(
        OUT_ptr, X_ptr, COS_ptr, SIN_ptr,
        n_rows, half_dim,
        stride_x_row, stride_out_row,
        stride_cos_row,
        BLOCK_HALF: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < half_dim

        x_base = row_id * stride_x_row
        cos_base = row_id * stride_cos_row
        out_base = row_id * stride_out_row

        x_even_raw = tl.load(X_ptr + x_base + offs * 2, mask=mask)
        out_dtype = x_even_raw.dtype

        x_even = x_even_raw.to(tl.float32)
        x_odd = tl.load(X_ptr + x_base + offs * 2 + 1, mask=mask).to(tl.float32)
        cos_val = tl.load(COS_ptr + cos_base + offs, mask=mask).to(tl.float32)
        sin_val = tl.load(SIN_ptr + cos_base + offs, mask=mask).to(tl.float32)

        out_even = x_even * cos_val - x_odd * sin_val
        out_odd = x_even * sin_val + x_odd * cos_val

        tl.store(OUT_ptr + out_base + offs * 2, out_even.to(out_dtype), mask=mask)
        tl.store(OUT_ptr + out_base + offs * 2 + 1, out_odd.to(out_dtype), mask=mask)

    @triton.jit
    def _rope_bwd_kernel(
        D_X_ptr, GRAD_ptr, COS_ptr, SIN_ptr,
        n_rows, half_dim,
        stride_grad_row, stride_dx_row,
        stride_cos_row,
        BLOCK_HALF: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offs = tl.arange(0, BLOCK_HALF)
        mask = offs < half_dim

        grad_base = row_id * stride_grad_row
        cos_base = row_id * stride_cos_row
        dx_base = row_id * stride_dx_row

        g_even_raw = tl.load(GRAD_ptr + grad_base + offs * 2, mask=mask)
        out_dtype = g_even_raw.dtype

        g_even = g_even_raw.to(tl.float32)
        g_odd = tl.load(GRAD_ptr + grad_base + offs * 2 + 1, mask=mask).to(tl.float32)
        cos_val = tl.load(COS_ptr + cos_base + offs, mask=mask).to(tl.float32)
        sin_val = tl.load(SIN_ptr + cos_base + offs, mask=mask).to(tl.float32)

        dx_even = g_even * cos_val + g_odd * sin_val
        dx_odd = -g_even * sin_val + g_odd * cos_val

        tl.store(D_X_ptr + dx_base + offs * 2, dx_even.to(out_dtype), mask=mask)
        tl.store(D_X_ptr + dx_base + offs * 2 + 1, dx_odd.to(out_dtype), mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Autograd Function
# ═══════════════════════════════════════════════════════════════════════

class _FusedRoPEFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, cos, sin):
        assert x.size(-1) % 2 == 0, f"RoPE head dim must be even, got {x.size(-1)}"

        orig_shape = x.shape
        full_dim = x.size(-1)
        half_dim = full_dim // 2

        x_flat = x.contiguous().view(-1, full_dim)
        n_rows = x_flat.shape[0]

        # cos/sin: shape [..., half_dim] — broadcast to match x rows
        cos_flat = cos[..., :half_dim].contiguous().view(-1, half_dim)
        sin_flat = sin[..., :half_dim].contiguous().view(-1, half_dim)

        # Broadcast cos/sin to n_rows if they have fewer rows (e.g. [T, half_dim] vs [B*H*T, full_dim])
        if cos_flat.shape[0] != n_rows:
            repeat_factor = n_rows // cos_flat.shape[0]
            cos_flat = cos_flat.repeat(repeat_factor, 1)
            sin_flat = sin_flat.repeat(repeat_factor, 1)

        out = torch.empty_like(x_flat)

        BLOCK_HALF = triton.next_power_of_2(half_dim)

        with kernel_region("rope_fwd"):
            _rope_fwd_kernel[(n_rows,)](
                out, x_flat, cos_flat, sin_flat,
                n_rows, half_dim,
                x_flat.stride(0), out.stride(0),
                cos_flat.stride(0),
                BLOCK_HALF=BLOCK_HALF,
            )

        ctx.save_for_backward(cos_flat, sin_flat)
        ctx.orig_shape = orig_shape
        return out.view(orig_shape)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        cos_flat, sin_flat = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        full_dim = orig_shape[-1]
        half_dim = full_dim // 2

        grad_flat = grad_output.contiguous().view(-1, full_dim)
        n_rows = grad_flat.shape[0]

        dx = torch.empty_like(grad_flat)
        BLOCK_HALF = triton.next_power_of_2(half_dim)

        with kernel_region("rope_bwd"):
            _rope_bwd_kernel[(n_rows,)](
                dx, grad_flat, cos_flat, sin_flat,
                n_rows, half_dim,
                grad_flat.stride(0), dx.stride(0),
                cos_flat.stride(0),
                BLOCK_HALF=BLOCK_HALF,
            )

        return dx.view(orig_shape), None, None


# ═══════════════════════════════════════════════════════════════════════
# PyTorch Fallback
# ═══════════════════════════════════════════════════════════════════════

def _pytorch_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.size(-1) % 2 != 0:
        raise ValueError(f"RoPE head dim must be even, got {x.size(-1)}")
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_half = cos[..., 0::2] if cos.size(-1) == x.size(-1) else cos[..., :x.size(-1) // 2]
    sin_half = sin[..., 0::2] if sin.size(-1) == x.size(-1) else sin[..., :x.size(-1) // 2]
    rot_even = x_even * cos_half - x_odd * sin_half
    rot_odd = x_even * sin_half + x_odd * cos_half
    return torch.stack((rot_even, rot_odd), dim=-1).reshape_as(x)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def fused_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Fused RoPE — Triton on CUDA, PyTorch fallback otherwise."""
    if HAS_TRITON and x.is_cuda and x.is_contiguous():
        try:
            return _FusedRoPEFunction.apply(x, cos, sin)
        except Exception:
            pass
    return _pytorch_rope(x, cos, sin)
