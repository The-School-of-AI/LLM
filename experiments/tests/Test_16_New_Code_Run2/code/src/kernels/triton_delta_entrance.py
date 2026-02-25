# triton_delta_entrance_v19.py
# =============================================================================
# V19: "Tile-Program" Delta Entrance (Forward fused, proper kernel)
# Fuses: causal depthwise conv(4) + bias + SiLU + mask + L2Norm(Q/K)
#        + interleaved RoPE(Q/K)
#
# Key changes vs V18:
# - Tile-based: each program processes BLOCK_T tokens (not 1 token).
#   Grid goes from (T, B*H) = 524K programs to (T/BLOCK_T, B*H) = ~8K.
#   This drastically reduces kernel launch and scheduling overhead.
# - Includes conv1d bias (V18 silently dropped it).
# - Backward: existing per-token Triton kernel (V18 backward is unchanged).
#
# Outputs: (B, T, H, D) like V18.
# =============================================================================

import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Import profiling helpers
try:
    from ..profiler import kernel_region
except ImportError:
    # Fallback: no-op context manager
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


# =============================================================================
# Reference (Unfused) for correctness + benchmark oracle
# =============================================================================
def pytorch_unfused(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps=1e-6):
    """Reference PyTorch implementation matching the fused kernel exactly."""
    B, T, C = q.shape
    D = cos.shape[1]
    H = C // D

    # 1) depthwise causal conv (4 taps) + bias
    qc = F.conv1d(q.transpose(1, 2), wq.view(C, 1, 4), bias=bq, groups=C, padding=3)[..., :-3].transpose(1, 2)
    kc = F.conv1d(k.transpose(1, 2), wk.view(C, 1, 4), bias=bk, groups=C, padding=3)[..., :-3].transpose(1, 2)
    vc = F.conv1d(v.transpose(1, 2), wv.view(C, 1, 4), bias=bv, groups=C, padding=3)[..., :-3].transpose(1, 2)

    # 2) gating + mask
    m = mask.unsqueeze(-1).to(q.dtype)
    qc, kc, vc = [F.silu(x) * m for x in (qc, kc, vc)]

    # 3) norm
    qn = F.normalize(qc.view(B, T, H, D), p=2, dim=-1, eps=eps)
    kn = F.normalize(kc.view(B, T, H, D), p=2, dim=-1, eps=eps)

    # 4) interleaved RoPE
    def apply_rope(x):
        c = cos[:, 0::2].unsqueeze(0).unsqueeze(2)
        s = sin[:, 0::2].unsqueeze(0).unsqueeze(2)
        xe, xo = x[..., 0::2], x[..., 1::2]
        out_e = xe * c - xo * s
        out_o = xe * s + xo * c
        return torch.stack((out_e, out_o), dim=-1).flatten(-2)

    return apply_rope(qn), apply_rope(kn), vc.view(B, T, H, D)


# =============================================================================
# Triton Forward Kernel (V19): BLOCK_T tokens per program
# =============================================================================
@triton.jit
def _delta_entrance_fwd_tile_kernel(
    # Inputs (B, T, C)
    Q_ptr, K_ptr, V_ptr,
    # Weights (C, 4) — squeezed from (C, 1, 4)
    Wq_ptr, Wk_ptr, Wv_ptr,
    # Biases (C,) — conv1d bias per channel
    Bq_ptr, Bk_ptr, Bv_ptr,
    # RoPE tables (T, D)
    Cos_ptr, Sin_ptr,
    # Mask (B, T) uint8 0/1
    Mask_ptr,
    # Outputs (B, T, H, D)
    Qo_ptr, Ko_ptr, Vo_ptr,
    # Stats (B, T, H) float32
    InvNq_ptr, InvNk_ptr,

    # Strides (elements)
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_mb, stride_mt,
    stride_sb, stride_st, stride_sh,

    # Sizes
    B: tl.constexpr, T: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,

    # Meta
    BLOCK_T: tl.constexpr,
    BLOCK_DH: tl.constexpr,   # D//2
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_block = tl.program_id(0)  # token block index
    pid_bh = tl.program_id(1)    # batch-head index

    b = pid_bh // H
    h = pid_bh % H
    block_start = pid_block * BLOCK_T

    # Lane indices for interleaved even/odd
    dh = tl.arange(0, BLOCK_DH)  # 0..D/2-1
    idx_e = dh * 2
    idx_o = idx_e + 1

    # Channel indices in (B, T, C)
    c_e = h * D + idx_e
    c_o = h * D + idx_o

    lane_e_ok = idx_e < D
    lane_o_ok = idx_o < D

    # Pre-load conv biases for this head's channels (reused across all tokens)
    bias_qe = tl.load(Bq_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bias_qo = tl.load(Bq_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bias_ke = tl.load(Bk_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bias_ko = tl.load(Bk_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bias_ve = tl.load(Bv_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bias_vo = tl.load(Bv_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)

    # Process BLOCK_T tokens sequentially
    for t_off in tl.static_range(BLOCK_T):
        t = block_start + t_off
        in_bounds = (b < B) & (t < T)

        # ── 1) Causal depthwise conv (4 taps) + bias ──
        qe = tl.zeros((BLOCK_DH,), dtype=tl.float32)
        qo = tl.zeros((BLOCK_DH,), dtype=tl.float32)
        ke = tl.zeros((BLOCK_DH,), dtype=tl.float32)
        ko = tl.zeros((BLOCK_DH,), dtype=tl.float32)
        ve = tl.zeros((BLOCK_DH,), dtype=tl.float32)
        vo = tl.zeros((BLOCK_DH,), dtype=tl.float32)

        m = tl.load(
            Mask_ptr + b * stride_mb + t * stride_mt,
            mask=in_bounds, other=0
        ).to(tl.float32)

        for i in tl.static_range(4):
            tap_t = t - (3 - i)
            tap_ok = in_bounds & (tap_t >= 0) & (tap_t < T)

            # Conv weights for this tap
            wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
            wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
            wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
            wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
            wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
            wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

            # Input values at tap position
            q_base = b * stride_qb + tap_t * stride_qt
            k_base = b * stride_kb + tap_t * stride_kt
            v_base = b * stride_vb + tap_t * stride_vt

            q_pe = tl.load(Q_ptr + q_base + c_e * stride_qc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
            q_po = tl.load(Q_ptr + q_base + c_o * stride_qc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)
            k_pe = tl.load(K_ptr + k_base + c_e * stride_kc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
            k_po = tl.load(K_ptr + k_base + c_o * stride_kc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)
            v_pe = tl.load(V_ptr + v_base + c_e * stride_vc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
            v_po = tl.load(V_ptr + v_base + c_o * stride_vc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)

            qe += q_pe * wqe
            qo += q_po * wqo
            ke += k_pe * wke
            ko += k_po * wko
            ve += v_pe * wve
            vo += v_po * wvo

        # Add conv bias
        qe += bias_qe
        qo += bias_qo
        ke += bias_ke
        ko += bias_ko
        ve += bias_ve
        vo += bias_vo

        # ── 2) SiLU + mask ──
        qe = (qe * tl.sigmoid(qe)) * m
        qo = (qo * tl.sigmoid(qo)) * m
        ke = (ke * tl.sigmoid(ke)) * m
        ko = (ko * tl.sigmoid(ko)) * m
        ve = (ve * tl.sigmoid(ve)) * m
        vo = (vo * tl.sigmoid(vo)) * m

        # ── 3) L2 norm over full D ──
        q_inv = tl.rsqrt(tl.sum(qe * qe + qo * qo, axis=0) + EPS)
        k_inv = tl.rsqrt(tl.sum(ke * ke + ko * ko, axis=0) + EPS)

        qne = qe * q_inv
        qno = qo * q_inv
        kne = ke * k_inv
        kno = ko * k_inv

        # ── 4) Interleaved RoPE ──
        cos_val = tl.load(
            Cos_ptr + t * D + idx_e,
            mask=(in_bounds & lane_e_ok), other=1.0
        ).to(tl.float32)
        sin_val = tl.load(
            Sin_ptr + t * D + idx_e,
            mask=(in_bounds & lane_e_ok), other=0.0
        ).to(tl.float32)

        qr_e = qne * cos_val - qno * sin_val
        qr_o = qne * sin_val + qno * cos_val
        kr_e = kne * cos_val - kno * sin_val
        kr_o = kne * sin_val + kno * cos_val

        # ── 5) Store to (B, T, H, D) interleaved ──
        out_base = b * stride_ob + t * stride_ot + h * stride_oh

        tl.store(Qo_ptr + out_base + idx_e * stride_od, qr_e.to(OUT_DTYPE), mask=(in_bounds & lane_e_ok))
        tl.store(Qo_ptr + out_base + idx_o * stride_od, qr_o.to(OUT_DTYPE), mask=(in_bounds & lane_o_ok))
        tl.store(Ko_ptr + out_base + idx_e * stride_od, kr_e.to(OUT_DTYPE), mask=(in_bounds & lane_e_ok))
        tl.store(Ko_ptr + out_base + idx_o * stride_od, kr_o.to(OUT_DTYPE), mask=(in_bounds & lane_o_ok))
        tl.store(Vo_ptr + out_base + idx_e * stride_od, ve.to(OUT_DTYPE), mask=(in_bounds & lane_e_ok))
        tl.store(Vo_ptr + out_base + idx_o * stride_od, vo.to(OUT_DTYPE), mask=(in_bounds & lane_o_ok))

        # Stats (for backward)
        s_off = b * stride_sb + t * stride_st + h * stride_sh
        tl.store(InvNq_ptr + s_off, q_inv, mask=in_bounds)
        tl.store(InvNk_ptr + s_off, k_inv, mask=in_bounds)


# =============================================================================
# Triton Backward Kernel: per-token (reuses V18 backward — grid (T, B*H))
# =============================================================================
@triton.jit
def _delta_entrance_bwd_token_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr,
    Wq_ptr, Wk_ptr, Wv_ptr,
    Bq_ptr, Bk_ptr, Bv_ptr,
    Cos_ptr, Sin_ptr,
    Mask_ptr,
    # Forward Stats
    InvNq_ptr, InvNk_ptr,
    # Gradients of outputs
    DQo_ptr, DKo_ptr, DVo_ptr,
    # Output Gradients
    DQ_ptr, DK_ptr, DV_ptr,
    DWq_ptr, DWk_ptr, DWv_ptr,
    DBq_ptr, DBk_ptr, DBv_ptr,

    # Strides (elements)
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_dqob, stride_dqot, stride_dqoh, stride_dqod,
    stride_mb, stride_mt,
    stride_sb, stride_st, stride_sh,

    # Sizes
    B: tl.constexpr, T: tl.constexpr, C: tl.constexpr,
    H: tl.constexpr, D: tl.constexpr,

    # Meta
    BLOCK_DH: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_t  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H
    t = pid_t

    in_bounds = (b < B) & (t < T)
    dh = tl.arange(0, BLOCK_DH)
    idx_e = dh * 2
    idx_o = idx_e + 1
    c_e = h * D + idx_e
    c_o = h * D + idx_o
    lane_e_ok = idx_e < D
    lane_o_ok = idx_o < D

    # ── 1) Re-compute Forward intermediate (Conv + bias → SiLU → Norm) ──
    qe = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    qo = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ke = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ko = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ve = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    vo = tl.zeros((BLOCK_DH,), dtype=tl.float32)

    m = tl.load(Mask_ptr + b * stride_mb + t * stride_mt, mask=in_bounds, other=0).to(tl.float32)

    for i in tl.static_range(4):
        tap_t = t - (3 - i)
        tap_ok = in_bounds & (tap_t >= 0) & (tap_t < T)

        wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

        q_pe = tl.load(Q_ptr + b * stride_qb + tap_t * stride_qt + c_e * stride_qc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
        q_po = tl.load(Q_ptr + b * stride_qb + tap_t * stride_qt + c_o * stride_qc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)
        k_pe = tl.load(K_ptr + b * stride_kb + tap_t * stride_kt + c_e * stride_kc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
        k_po = tl.load(K_ptr + b * stride_kb + tap_t * stride_kt + c_o * stride_kc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)
        v_pe = tl.load(V_ptr + b * stride_vb + tap_t * stride_vt + c_e * stride_vc, mask=(tap_ok & lane_e_ok), other=0.0).to(tl.float32)
        v_po = tl.load(V_ptr + b * stride_vb + tap_t * stride_vt + c_o * stride_vc, mask=(tap_ok & lane_o_ok), other=0.0).to(tl.float32)

        qe += q_pe * wqe
        qo += q_po * wqo
        ke += k_pe * wke
        ko += k_po * wko
        ve += v_pe * wve
        vo += v_po * wvo

    # Add conv bias
    qe += tl.load(Bq_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    qo += tl.load(Bq_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    ke += tl.load(Bk_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    ko += tl.load(Bk_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    ve += tl.load(Bv_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    vo += tl.load(Bv_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)

    # Save pre-activation for SiLU backward
    xcq_e, xcq_o = qe, qo
    xck_e, xck_o = ke, ko
    xcv_e, xcv_o = ve, vo

    sqe = tl.sigmoid(xcq_e); qe = xcq_e * sqe * m
    sqo = tl.sigmoid(xcq_o); qo = xcq_o * sqo * m
    ske = tl.sigmoid(xck_e); ke = xck_e * ske * m
    sko = tl.sigmoid(xck_o); ko = xck_o * sko * m
    sve = tl.sigmoid(xcv_e); ve = xcv_e * sve * m
    svo = tl.sigmoid(xcv_o); vo = xcv_o * svo * m

    s_off = b * stride_sb + t * stride_st + h * stride_sh
    inv_nq = tl.load(InvNq_ptr + s_off, mask=in_bounds, other=1.0)
    inv_nk = tl.load(InvNk_ptr + s_off, mask=in_bounds, other=1.0)
    qne, qno = qe * inv_nq, qo * inv_nq
    kne, kno = ke * inv_nk, ko * inv_nk

    # ── 2) RoPE Backward ──
    cos_val = tl.load(Cos_ptr + t * D + idx_e, mask=(in_bounds & lane_e_ok), other=1.0).to(tl.float32)
    sin_val = tl.load(Sin_ptr + t * D + idx_e, mask=(in_bounds & lane_e_ok), other=0.0).to(tl.float32)

    dqo_e = tl.load(DQo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_e * stride_dqod, mask=(in_bounds & lane_e_ok), other=0.0).to(tl.float32)
    dqo_o = tl.load(DQo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_o * stride_dqod, mask=(in_bounds & lane_o_ok), other=0.0).to(tl.float32)
    dko_e = tl.load(DKo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_e * stride_dqod, mask=(in_bounds & lane_e_ok), other=0.0).to(tl.float32)
    dko_o = tl.load(DKo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_o * stride_dqod, mask=(in_bounds & lane_o_ok), other=0.0).to(tl.float32)

    dqne = dqo_e * cos_val + dqo_o * sin_val
    dqno = -dqo_e * sin_val + dqo_o * cos_val
    dkne = dko_e * cos_val + dko_o * sin_val
    dkno = -dko_e * sin_val + dko_o * cos_val

    # ── 3) L2 Norm Backward ──
    dot_q = tl.sum(qne * dqne + qno * dqno, axis=0)
    dot_k = tl.sum(kne * dkne + kno * dkno, axis=0)

    dqe = (dqne - qne * dot_q) * inv_nq
    dqo_grad = (dqno - qno * dot_q) * inv_nq
    dke = (dkne - kne * dot_k) * inv_nk
    dko_grad = (dkno - kno * dot_k) * inv_nk

    # ── 4) SiLU + Mask Backward ──
    dsqe = sqe * (1.0 + xcq_e * (1.0 - sqe)) * m
    dsqo = sqo * (1.0 + xcq_o * (1.0 - sqo)) * m
    dske = ske * (1.0 + xck_e * (1.0 - ske)) * m
    dsko = sko * (1.0 + xck_o * (1.0 - sko)) * m

    dvo_e = tl.load(DVo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_e * stride_dqod, mask=(in_bounds & lane_e_ok), other=0.0).to(tl.float32)
    dvo_o = tl.load(DVo_ptr + b * stride_dqob + t * stride_dqot + h * stride_dqoh + idx_o * stride_dqod, mask=(in_bounds & lane_o_ok), other=0.0).to(tl.float32)
    dsve = sve * (1.0 + xcv_e * (1.0 - sve)) * m
    dsvo = svo * (1.0 + xcv_o * (1.0 - svo)) * m

    dqc_e = dqe * dsqe
    dqc_o = dqo_grad * dsqo
    dkc_e = dke * dske
    dkc_o = dko_grad * dsko
    dvc_e = dvo_e * dsve
    dvc_o = dvo_o * dsvo

    # ── 5) Conv1d Backward: input grads + weight grads + bias grads (Atomic) ──
    for i in tl.static_range(4):
        prev_t = t - (3 - i)
        prev_ok = in_bounds & (prev_t >= 0)

        wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

        off_e = b * stride_qb + prev_t * stride_qt + c_e * stride_qc
        off_o = b * stride_qb + prev_t * stride_qt + c_o * stride_qc

        # Input gradients (atomic because multiple tokens contribute to same position)
        tl.atomic_add(DQ_ptr + off_e, (dqc_e * wqe).to(tl.float32), mask=(prev_ok & lane_e_ok))
        tl.atomic_add(DQ_ptr + off_o, (dqc_o * wqo).to(tl.float32), mask=(prev_ok & lane_o_ok))
        tl.atomic_add(DK_ptr + off_e, (dkc_e * wke).to(tl.float32), mask=(prev_ok & lane_e_ok))
        tl.atomic_add(DK_ptr + off_o, (dkc_o * wko).to(tl.float32), mask=(prev_ok & lane_o_ok))
        tl.atomic_add(DV_ptr + off_e, (dvc_e * wve).to(tl.float32), mask=(prev_ok & lane_e_ok))
        tl.atomic_add(DV_ptr + off_o, (dvc_o * wvo).to(tl.float32), mask=(prev_ok & lane_o_ok))

        # Weight gradients
        q_pe = tl.load(Q_ptr + off_e, mask=(prev_ok & lane_e_ok), other=0.0).to(tl.float32)
        q_po = tl.load(Q_ptr + off_o, mask=(prev_ok & lane_o_ok), other=0.0).to(tl.float32)
        k_pe = tl.load(K_ptr + off_e, mask=(prev_ok & lane_e_ok), other=0.0).to(tl.float32)
        k_po = tl.load(K_ptr + off_o, mask=(prev_ok & lane_o_ok), other=0.0).to(tl.float32)
        v_pe = tl.load(V_ptr + off_e, mask=(prev_ok & lane_e_ok), other=0.0).to(tl.float32)
        v_po = tl.load(V_ptr + off_o, mask=(prev_ok & lane_o_ok), other=0.0).to(tl.float32)

        tl.atomic_add(DWq_ptr + c_e * 4 + i, (dqc_e * q_pe).to(tl.float32), mask=(in_bounds & lane_e_ok))
        tl.atomic_add(DWq_ptr + c_o * 4 + i, (dqc_o * q_po).to(tl.float32), mask=(in_bounds & lane_o_ok))
        tl.atomic_add(DWk_ptr + c_e * 4 + i, (dkc_e * k_pe).to(tl.float32), mask=(in_bounds & lane_e_ok))
        tl.atomic_add(DWk_ptr + c_o * 4 + i, (dkc_o * k_po).to(tl.float32), mask=(in_bounds & lane_o_ok))
        tl.atomic_add(DWv_ptr + c_e * 4 + i, (dvc_e * v_pe).to(tl.float32), mask=(in_bounds & lane_e_ok))
        tl.atomic_add(DWv_ptr + c_o * 4 + i, (dvc_o * v_po).to(tl.float32), mask=(in_bounds & lane_o_ok))

    # Bias gradients (atomic across tokens)
    tl.atomic_add(DBq_ptr + c_e, dqc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBq_ptr + c_o, dqc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))
    tl.atomic_add(DBk_ptr + c_e, dkc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBk_ptr + c_o, dkc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))
    tl.atomic_add(DBv_ptr + c_e, dvc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBv_ptr + c_o, dvc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))


# =============================================================================
# Autograd wrapper
# =============================================================================
class FusedDeltaEntranceV19(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps=1e-6):
        B, T, C = q.shape
        D = cos.shape[1]
        H = C // D
        assert D % 2 == 0, "RoPE interleaved requires even D"

        # weights -> (C, 4)
        if wq.ndim == 3:
            wq = wq.squeeze(1)
            wk = wk.squeeze(1)
            wv = wv.squeeze(1)
        wq = wq.contiguous()
        wk = wk.contiguous()
        wv = wv.contiguous()

        # mask -> uint8 0/1
        if mask.dtype == torch.bool:
            mask_u8 = mask.to(torch.uint8)
        elif mask.dtype == torch.uint8:
            mask_u8 = mask
        else:
            mask_u8 = (mask != 0).to(torch.uint8)

        qo = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        ko = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        vo = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        inv_nq = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
        inv_nk = torch.empty((B, T, H), device=q.device, dtype=torch.float32)

        BLOCK_DH = D // 2
        BLOCK_T = 64  # Process 64 tokens per program
        grid = (triton.cdiv(T, BLOCK_T), B * H)

        out_dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

        with kernel_region("delta_entrance_fwd"):
            _delta_entrance_fwd_tile_kernel[grid](
                q, k, v,
                wq, wk, wv,
                bq, bk, bv,
                cos, sin,
                mask_u8,
                qo, ko, vo,
                inv_nq, inv_nk,

                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                qo.stride(0), qo.stride(1), qo.stride(2), qo.stride(3),
                mask_u8.stride(0), mask_u8.stride(1),
                inv_nq.stride(0), inv_nq.stride(1), inv_nq.stride(2),

                B, T, C, H, D,

                BLOCK_T=BLOCK_T,
                BLOCK_DH=BLOCK_DH,
                EPS=eps,
                OUT_DTYPE=out_dtype,

                num_warps=4,
                num_stages=2,
            )

        ctx.save_for_backward(q, k, v, wq, wk, wv, bq, bk, bv,
                              cos, sin, mask_u8, inv_nq, inv_nk)
        ctx.eps = eps
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask_u8, inv_nq, inv_nk = ctx.saved_tensors
        B, T, C = q.shape
        D = cos.shape[1]
        H = C // D

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dwq = torch.zeros_like(wq, dtype=torch.float32)
        dwk = torch.zeros_like(wk, dtype=torch.float32)
        dwv = torch.zeros_like(wv, dtype=torch.float32)
        dbq = torch.zeros(C, device=q.device, dtype=torch.float32)
        dbk = torch.zeros(C, device=q.device, dtype=torch.float32)
        dbv = torch.zeros(C, device=q.device, dtype=torch.float32)

        BLOCK_DH = D // 2
        grid = (T, B * H)

        with kernel_region("delta_entrance_bwd"):
            _delta_entrance_bwd_token_kernel[grid](
                q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask_u8,
                inv_nq, inv_nk,
                dqo, dko, dvo,
                dq, dk, dv,
                dwq, dwk, dwv,
                dbq, dbk, dbv,

                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                dqo.stride(0), dqo.stride(1), dqo.stride(2), dqo.stride(3),
                mask_u8.stride(0), mask_u8.stride(1),
                inv_nq.stride(0), inv_nq.stride(1), inv_nq.stride(2),

                B, T, C, H, D,
                BLOCK_DH=BLOCK_DH,
                EPS=ctx.eps,
                num_warps=4,
                num_stages=2,
            )

        return (
            dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype),
            dwq.to(wq.dtype), dwk.to(wk.dtype), dwv.to(wv.dtype),
            dbq.to(bq.dtype), dbk.to(bk.dtype), dbv.to(bv.dtype),
            None, None, None, None
        )


def fused_delta_entrance(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps=1e-6):
    """Fused conv1d + SiLU + mask + L2Norm + RoPE for DeltaNet pre-processing.

    Args:
        q, k, v: [B, T, C] after linear projection (before conv)
        wq, wk, wv: [C, 1, 4] or [C, 4] conv weights (depthwise)
        bq, bk, bv: [C] conv biases
        cos, sin: [T, D] RoPE tables
        mask: [B, T] attention mask (bool, uint8, or float)

    Returns:
        qo, ko, vo: [B, T, H, D] ready for FLA
    """
    return FusedDeltaEntranceV19.apply(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps)


# =============================================================================
# Benchmark harness
# =============================================================================
if __name__ == "__main__":
    import triton.testing

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["T"],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=["pytorch", "triton_v19"],
            line_names=["PyTorch (Unfused)", "Triton V19 (Tile-Program)"],
            styles=[("red", "-"), ("green", "-")],
            ylabel="Execution Time (ms)",
            plot_name="Delta-Entrance Performance (Forward Pass) - V19",
            args={"B": 1, "H": 32, "D": 128, "dtype": torch.bfloat16},
        )
    )
    def benchmark(B, T, H, D, dtype, provider):
        C = H * D
        device = "cuda"

        q = torch.randn((B, T, C), device=device, dtype=dtype)
        k = torch.randn((B, T, C), device=device, dtype=dtype)
        v = torch.randn((B, T, C), device=device, dtype=dtype)
        wq = torch.randn((C, 1, 4), device=device, dtype=dtype)
        wk = torch.randn((C, 1, 4), device=device, dtype=dtype)
        wv = torch.randn((C, 1, 4), device=device, dtype=dtype)
        bq = torch.randn((C,), device=device, dtype=dtype)
        bk = torch.randn((C,), device=device, dtype=dtype)
        bv = torch.randn((C,), device=device, dtype=dtype)
        cos = torch.randn((T, D), device=device, dtype=dtype)
        sin = torch.randn((T, D), device=device, dtype=dtype)
        mask = torch.ones((B, T), device=device, dtype=torch.uint8)

        # warmup
        if provider == "triton_v19":
            fused_delta_entrance(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)
        else:
            pytorch_unfused(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "pytorch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: pytorch_unfused(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_delta_entrance(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask),
                quantiles=quantiles,
            )

        return ms, max_ms, min_ms

    # Quick correctness smoke test
    B, T, H, D = 1, 256, 8, 64
    C = H * D
    dtype = torch.bfloat16
    device = "cuda"
    q = torch.randn((B, T, C), device=device, dtype=dtype)
    k = torch.randn((B, T, C), device=device, dtype=dtype)
    v = torch.randn((B, T, C), device=device, dtype=dtype)
    wq = torch.randn((C, 1, 4), device=device, dtype=dtype)
    wk = torch.randn((C, 1, 4), device=device, dtype=dtype)
    wv = torch.randn((C, 1, 4), device=device, dtype=dtype)
    bq = torch.randn((C,), device=device, dtype=dtype)
    bk = torch.randn((C,), device=device, dtype=dtype)
    bv = torch.randn((C,), device=device, dtype=dtype)
    cos = torch.randn((T, D), device=device, dtype=dtype)
    sin = torch.randn((T, D), device=device, dtype=dtype)
    mask = torch.ones((B, T), device=device, dtype=torch.uint8)

    with torch.no_grad():
        qo_ref, ko_ref, vo_ref = pytorch_unfused(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)
        qo_tri, ko_tri, vo_tri = fused_delta_entrance(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)

        def max_abs(a, b):
            return (a.float() - b.float()).abs().max().item()

        print("max|Qo-ref|:", max_abs(qo_ref, qo_tri))
        print("max|Ko-ref|:", max_abs(ko_ref, ko_tri))
        print("max|Vo-ref|:", max_abs(vo_ref, vo_tri))

    benchmark.run(show_plots=True, print_data=True)
