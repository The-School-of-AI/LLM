"""
Unsloth-style RoPE kernel for (B, T, H, D) Q/K and (T, D//2) cos/sin.

Ported from unslothai/unsloth (GNU LGPL / Apache-2.0).
Repository: https://github.com/unslothai/unsloth
Used for experimentation: reduce peak memory and/or increase throughput
when applied in the unfused delta entrance path (env T17_USE_UNSLOTH_ROPE=1).
"""

from __future__ import annotations

import os
import torch
import triton
import triton.language as tl

try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    p = 2
    while p < n:
        p *= 2
    return p


def _rope_settings(half_head_dim: int):
    """BLOCK_SIZE and num_warps for RoPE kernel (no Unsloth device_type dependency)."""
    BLOCK_SIZE = min(_next_power_of_2(half_head_dim), 1024)
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _rope_qk_kernel(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    stride_qb,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_cos_t,
    stride_cos_d,
    stride_sin_t,
    stride_sin_d,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_DH: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """
    Apply RoPE to Q and K in place.
    Layout: Q, K (B, T, H, D); cos, sin (T, D//2).
    Grid: (B*T, H). Each program handles one (b, t, h) and D elements.
    RoPE: even/odd interleaved — out_e = q_e*cos - q_o*sin, out_o = q_e*sin + q_o*cos.
    """
    row_id = tl.program_id(0)
    h = tl.program_id(1)
    t = row_id % T
    b = row_id // T

    col_offsets = tl.arange(0, BLOCK_DH)
    mask = col_offsets < (D // 2)

    cos = tl.load(
        Cos_ptr + t * stride_cos_t + col_offsets * stride_cos_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + t * stride_sin_t + col_offsets * stride_sin_d,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if BACKWARD_PASS:
        sin = -sin

    base_q = b * stride_qb + t * stride_qt + h * stride_qh
    base_k = b * stride_kb + t * stride_kt + h * stride_kh

    # Interleaved layout: even at 2*col, odd at 2*col+1
    idx_e = col_offsets * 2
    idx_o = col_offsets * 2 + 1

    q_e = tl.load(Q_ptr + base_q + idx_e, mask=mask, other=0.0).to(tl.float32)
    q_o = tl.load(Q_ptr + base_q + idx_o, mask=mask, other=0.0).to(tl.float32)
    k_e = tl.load(K_ptr + base_k + idx_e, mask=mask, other=0.0).to(tl.float32)
    k_o = tl.load(K_ptr + base_k + idx_o, mask=mask, other=0.0).to(tl.float32)

    out_q_e = q_e * cos - q_o * sin
    out_q_o = q_e * sin + q_o * cos
    out_k_e = k_e * cos - k_o * sin
    out_k_o = k_e * sin + k_o * cos

    tl.store(Q_ptr + base_q + idx_e, out_q_e.to(OUT_DTYPE), mask=mask)
    tl.store(Q_ptr + base_q + idx_o, out_q_o.to(OUT_DTYPE), mask=mask)
    tl.store(K_ptr + base_k + idx_e, out_k_e.to(OUT_DTYPE), mask=mask)
    tl.store(K_ptr + base_k + idx_o, out_k_o.to(OUT_DTYPE), mask=mask)


class FastRoPEQK(torch.autograd.Function):
    """Apply RoPE to Q and K in (B, T, H, D) with cos/sin (T, D//2). In-place."""

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # cos, sin: (T, D//2)
        B, T, H, D = q.shape
        assert cos.shape == (T, D // 2) and sin.shape == (T, D // 2), (
            f"cos/sin must be (T, D//2), got cos={cos.shape}, sin={sin.shape}"
        )
        # Kernel uses strides; no contiguous() on q/k to avoid extra allocations (70B OOM).
        cos = cos.contiguous()
        sin = sin.contiguous()

        BLOCK_DH, num_warps = _rope_settings(D // 2)
        grid = (B * T, H)
        out_dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else (tl.float16 if q.dtype == torch.float16 else tl.float32)

        with kernel_region("unsloth_rope_fwd"):
            _rope_qk_kernel[grid](
                q,
                k,
                cos,
                sin,
                stride_qb=q.stride(0),
                stride_qt=q.stride(1),
                stride_qh=q.stride(2),
                stride_qd=q.stride(3),
                stride_kb=k.stride(0),
                stride_kt=k.stride(1),
                stride_kh=k.stride(2),
                stride_kd=k.stride(3),
                stride_cos_t=cos.stride(0),
                stride_cos_d=cos.stride(1),
                stride_sin_t=sin.stride(0),
                stride_sin_d=sin.stride(1),
                T=T,
                H=H,
                D=D,
                BACKWARD_PASS=False,
                BLOCK_DH=BLOCK_DH,
                OUT_DTYPE=out_dtype,
                num_warps=num_warps,
            )

        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        B, T, H, D = dq.shape
        # Kernel uses strides; no contiguous() on dq/dk to avoid extra gradient copies (70B OOM).
        cos = cos.contiguous()
        sin = sin.contiguous()

        BLOCK_DH, num_warps = _rope_settings(D // 2)
        grid = (B * T, H)
        out_dtype = tl.bfloat16 if dq.dtype == torch.bfloat16 else (tl.float16 if dq.dtype == torch.float16 else tl.float32)

        with kernel_region("unsloth_rope_bwd"):
            _rope_qk_kernel[grid](
                dq,
                dk,
                cos,
                sin,
                stride_qb=dq.stride(0),
                stride_qt=dq.stride(1),
                stride_qh=dq.stride(2),
                stride_qd=dq.stride(3),
                stride_kb=dk.stride(0),
                stride_kt=dk.stride(1),
                stride_kh=dk.stride(2),
                stride_kd=dk.stride(3),
                stride_cos_t=cos.stride(0),
                stride_cos_d=cos.stride(1),
                stride_sin_t=sin.stride(0),
                stride_sin_d=sin.stride(1),
                T=T,
                H=H,
                D=D,
                BACKWARD_PASS=True,
                BLOCK_DH=BLOCK_DH,
                OUT_DTYPE=out_dtype,
                num_warps=num_warps,
            )

        return dq, dk, None, None


def fast_rope_qk(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply Unsloth-style RoPE to Q and K in-place.

    Args:
        q, k: (B, T, H, D)
        cos, sin: (T, D//2) — compact RoPE tables (e.g. cos[:, 0::2] from full cos (T, D))

    Returns:
        q, k (same tensors, modified in-place)
    """
    return FastRoPEQK.apply(q, k, cos, sin)


def use_unsloth_rope() -> bool:
    """True if T17_USE_UNSLOTH_ROPE=1 (use Unsloth RoPE in unfused delta entrance)."""
    return os.environ.get("T17_USE_UNSLOTH_ROPE", "0") == "1"
