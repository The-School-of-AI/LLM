"""
Fused Dual QK-RoPE: apply rotary position embedding to Q and K in one kernel.
Saves loading cos/sin tables twice from global memory.
Called 2x per forward (GSA layers). GSA: H=16, D=256.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_rope_kernel(
    Q_ptr,
    K_ptr,
    Cos_ptr,
    Sin_ptr,
    Q_out_ptr,
    K_out_ptr,
    B,
    T,
    H,
    half_d: tl.constexpr,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_cs_t,
    stride_cs_d,
    stride_qo_b,
    stride_qo_t,
    stride_qo_h,
    stride_qo_d,
    stride_ko_b,
    stride_ko_t,
    stride_ko_h,
    stride_ko_d,
    BLOCK_HD: tl.constexpr,
):
    """Apply RoPE to Q and K, loading cos/sin once per (b,t,h)."""
    pid = tl.program_id(0)  # linearized (b, t, h)

    # Decode indices
    pid_h = pid % H
    pid_bt = pid // H
    pid_t = pid_bt % T
    pid_b = pid_bt // T

    d_off = tl.arange(0, BLOCK_HD)
    d_mask = d_off < half_d

    # Load cos/sin ONCE (shared between Q and K, independent of head)
    cos = tl.load(
        Cos_ptr + pid_t * stride_cs_t + d_off * stride_cs_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + pid_t * stride_cs_t + d_off * stride_cs_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    # Q base pointer
    q_base = Q_ptr + pid_b * stride_q_b + pid_t * stride_q_t + pid_h * stride_q_h
    k_base = K_ptr + pid_b * stride_k_b + pid_t * stride_k_t + pid_h * stride_k_h
    qo_base = (
        Q_out_ptr + pid_b * stride_qo_b + pid_t * stride_qo_t + pid_h * stride_qo_h
    )
    ko_base = (
        K_out_ptr + pid_b * stride_ko_b + pid_t * stride_ko_t + pid_h * stride_ko_h
    )

    # Apply RoPE to Q: interleaved even/odd pairs
    # q[..., 0::2] * cos - q[..., 1::2] * sin  (even positions)
    # q[..., 0::2] * sin + q[..., 1::2] * cos  (odd positions)
    q_even = tl.load(q_base + (2 * d_off) * stride_q_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    q_odd = tl.load(q_base + (2 * d_off + 1) * stride_q_d, mask=d_mask, other=0.0).to(
        tl.float32
    )

    q_out_even = q_even * cos - q_odd * sin
    q_out_odd = q_even * sin + q_odd * cos

    tl.store(
        qo_base + (2 * d_off) * stride_qo_d,
        q_out_even.to(Q_out_ptr.dtype.element_ty),
        mask=d_mask,
    )
    tl.store(
        qo_base + (2 * d_off + 1) * stride_qo_d,
        q_out_odd.to(Q_out_ptr.dtype.element_ty),
        mask=d_mask,
    )

    # Apply RoPE to K (cos/sin already in registers)
    k_even = tl.load(k_base + (2 * d_off) * stride_k_d, mask=d_mask, other=0.0).to(
        tl.float32
    )
    k_odd = tl.load(k_base + (2 * d_off + 1) * stride_k_d, mask=d_mask, other=0.0).to(
        tl.float32
    )

    k_out_even = k_even * cos - k_odd * sin
    k_out_odd = k_even * sin + k_odd * cos

    tl.store(
        ko_base + (2 * d_off) * stride_ko_d,
        k_out_even.to(K_out_ptr.dtype.element_ty),
        mask=d_mask,
    )
    tl.store(
        ko_base + (2 * d_off + 1) * stride_ko_d,
        k_out_odd.to(K_out_ptr.dtype.element_ty),
        mask=d_mask,
    )


@triton.jit
def _fused_qk_rope_bwd_kernel(
    dQ_out_ptr,
    dK_out_ptr,
    Cos_ptr,
    Sin_ptr,
    dQ_ptr,
    dK_ptr,
    B,
    T,
    H,
    half_d: tl.constexpr,
    stride_q_b,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_b,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_cs_t,
    stride_cs_d,
    stride_qo_b,
    stride_qo_t,
    stride_qo_h,
    stride_qo_d,
    stride_ko_b,
    stride_ko_t,
    stride_ko_h,
    stride_ko_d,
    BLOCK_HD: tl.constexpr,
):
    """RoPE backward: same as forward but with negated sin for the inverse rotation."""
    pid = tl.program_id(0)
    pid_h = pid % H
    pid_bt = pid // H
    pid_t = pid_bt % T
    pid_b = pid_bt // T

    d_off = tl.arange(0, BLOCK_HD)
    d_mask = d_off < half_d

    cos = tl.load(
        Cos_ptr + pid_t * stride_cs_t + d_off * stride_cs_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + pid_t * stride_cs_t + d_off * stride_cs_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    # dQ backward: inverse rotation (negate sin)
    dqo_base = (
        dQ_out_ptr + pid_b * stride_qo_b + pid_t * stride_qo_t + pid_h * stride_qo_h
    )
    dq_base = dQ_ptr + pid_b * stride_q_b + pid_t * stride_q_t + pid_h * stride_q_h

    dq_out_even = tl.load(
        dqo_base + (2 * d_off) * stride_qo_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    dq_out_odd = tl.load(
        dqo_base + (2 * d_off + 1) * stride_qo_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    # Inverse rotation: cos, -sin
    dq_even = dq_out_even * cos + dq_out_odd * sin
    dq_odd = -dq_out_even * sin + dq_out_odd * cos

    tl.store(
        dq_base + (2 * d_off) * stride_q_d,
        dq_even.to(dQ_ptr.dtype.element_ty),
        mask=d_mask,
    )
    tl.store(
        dq_base + (2 * d_off + 1) * stride_q_d,
        dq_odd.to(dQ_ptr.dtype.element_ty),
        mask=d_mask,
    )

    # dK backward
    dko_base = (
        dK_out_ptr + pid_b * stride_ko_b + pid_t * stride_ko_t + pid_h * stride_ko_h
    )
    dk_base = dK_ptr + pid_b * stride_k_b + pid_t * stride_k_t + pid_h * stride_k_h

    dk_out_even = tl.load(
        dko_base + (2 * d_off) * stride_ko_d, mask=d_mask, other=0.0
    ).to(tl.float32)
    dk_out_odd = tl.load(
        dko_base + (2 * d_off + 1) * stride_ko_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    dk_even = dk_out_even * cos + dk_out_odd * sin
    dk_odd = -dk_out_even * sin + dk_out_odd * cos

    tl.store(
        dk_base + (2 * d_off) * stride_k_d,
        dk_even.to(dK_ptr.dtype.element_ty),
        mask=d_mask,
    )
    tl.store(
        dk_base + (2 * d_off + 1) * stride_k_d,
        dk_odd.to(dK_ptr.dtype.element_ty),
        mask=d_mask,
    )


class FusedQKRoPEFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        """
        q, k: [B, T, H, D]
        cos, sin: [T, D//2]
        Returns: (q_rot, k_rot) each [B, T, H, D]
        """
        B, T, H, D = q.shape
        half_d = D // 2
        BLOCK_HD = triton.next_power_of_2(half_d)

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        grid = (B * T * H,)
        _fused_qk_rope_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_out,
            k_out,
            B,
            T,
            H,
            half_d,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            cos.stride(0),
            cos.stride(1),
            q_out.stride(0),
            q_out.stride(1),
            q_out.stride(2),
            q_out.stride(3),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            k_out.stride(3),
            BLOCK_HD=BLOCK_HD,
            num_warps=4,
        )

        ctx.save_for_backward(cos, sin)
        ctx.shape = (B, T, H, D)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq_out, dk_out):
        cos, sin = ctx.saved_tensors
        B, T, H, D = ctx.shape
        half_d = D // 2
        BLOCK_HD = triton.next_power_of_2(half_d)

        dq = torch.empty_like(dq_out)
        dk = torch.empty_like(dk_out)

        grid = (B * T * H,)
        _fused_qk_rope_bwd_kernel[grid](
            dq_out,
            dk_out,
            cos,
            sin,
            dq,
            dk,
            B,
            T,
            H,
            half_d,
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            cos.stride(0),
            cos.stride(1),
            dq_out.stride(0),
            dq_out.stride(1),
            dq_out.stride(2),
            dq_out.stride(3),
            dk_out.stride(0),
            dk_out.stride(1),
            dk_out.stride(2),
            dk_out.stride(3),
            BLOCK_HD=BLOCK_HD,
            num_warps=4,
        )

        return dq, dk, None, None


def fused_qk_rope(q, k, cos, sin):
    """Apply RoPE to Q and K in a single kernel launch."""
    return FusedQKRoPEFn.apply(q, k, cos, sin)


def pytorch_apply_rotary(x, cos, sin):
    """Single-tensor RoPE reference (interleaved layout)."""
    D = x.shape[-1]
    half_d = D // 2
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos_b = cos[: x.shape[1]].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D//2]
    sin_b = sin[: x.shape[1]].unsqueeze(0).unsqueeze(2)
    out_even = x_even * cos_b - x_odd * sin_b
    out_odd = x_even * sin_b + x_odd * cos_b
    out = torch.stack([out_even, out_odd], dim=-1).reshape_as(x)
    return out


def pytorch_fused_qk_rope(q, k, cos, sin):
    return pytorch_apply_rotary(q, cos, sin), pytorch_apply_rotary(k, cos, sin)
