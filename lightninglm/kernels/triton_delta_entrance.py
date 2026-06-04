# triton_delta_entrance_v19.py
# =============================================================================
# V19: "Token-Program" Delta Entrance (Forward+Backward fused autograd path)
# Fuses: causal depthwise conv(4) + bias + SiLU + L2Norm(Q/K) + interleaved RoPE(Q/K)
#
# Key change vs your V17:
# - One program instance computes ONE token t for ONE (batch, head).
# - Vector width is D/2 (even + odd lanes) so register footprint is O(D), not O(BLOCK_T*D).
# - This avoids the giant fp32 tiles that were almost certainly spilling at large T.
#
# RoPE tables are COMPACT by contract: cos/sin shape = (T, D//2).
# Outputs: (B, T, H, D) like your current wrapper.
# Backward: Triton recompute + analytic grads for entrance ops and conv weights.
# =============================================================================
# References used for operator behavior / integration choices:
# - FLA GatedDeltaNet layer:
#   https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
# - FLA gated delta-rule kernels:
#   https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# - NVLabs GatedDeltaNet FLA kernels:
#   https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py
# =============================================================================

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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
def pytorch_unfused_exact(
    q,
    k,
    v,
    wq,
    wk,
    wv,
    bq,
    bk,
    bv,
    cos,
    sin,
    mask,
    eps=1e-6,
    mask_after=True,
):
    B, T, C = q.shape
    assert (
        cos.ndim == 2 and sin.ndim == 2 and cos.shape == sin.shape
    ), "cos/sin must be shape (T, D//2)"
    assert cos.shape[0] == T, "RoPE T mismatch"
    dh = cos.shape[1]
    D = dh * 2
    assert C % D == 0, f"C={C} must be divisible by D={D}"
    H = C // D

    # 1) depthwise causal conv (4 taps)
    wq2 = wq if wq.ndim == 3 else wq.view(C, 1, 4)
    wk2 = wk if wk.ndim == 3 else wk.view(C, 1, 4)
    wv2 = wv if wv.ndim == 3 else wv.view(C, 1, 4)
    qc = F.conv1d(q.transpose(1, 2), wq2, bias=bq, groups=C, padding=3)[
        ..., :-3
    ].transpose(1, 2)
    kc = F.conv1d(k.transpose(1, 2), wk2, bias=bk, groups=C, padding=3)[
        ..., :-3
    ].transpose(1, 2)
    vc = F.conv1d(v.transpose(1, 2), wv2, bias=bv, groups=C, padding=3)[
        ..., :-3
    ].transpose(1, 2)

    # 2) gating
    qc, kc, vc = [F.silu(x) for x in (qc, kc, vc)]

    # 3) norm
    qn = F.normalize(qc.view(B, T, H, D), p=2, dim=-1, eps=eps)
    kn = F.normalize(kc.view(B, T, H, D), p=2, dim=-1, eps=eps)

    # 4) interleaved RoPE
    def apply_rope(x):
        c = cos.unsqueeze(0).unsqueeze(2)
        s = sin.unsqueeze(0).unsqueeze(2)
        xe, xo = x[..., 0::2], x[..., 1::2]
        out_e = xe * c - xo * s
        out_o = xe * s + xo * c
        return torch.stack((out_e, out_o), dim=-1).flatten(-2)

    qo = apply_rope(qn)
    ko = apply_rope(kn)
    vo = vc.view(B, T, H, D)

    if mask_after and mask is not None:
        m = mask.view(B, T, 1, 1).to(q.dtype)
        qo = qo * m
        ko = ko * m
        vo = vo * m

    return qo, ko, vo


# =============================================================================
# Triton Kernel (V19): one (b,h,t) per program
# =============================================================================
@triton.jit
def _delta_entrance_fwd_token_kernel(
    # Inputs (B, T, C)
    Q_ptr,
    K_ptr,
    V_ptr,
    # Weights (C, 4)  (NOTE: wrapper will squeeze if (C,1,4))
    Wq_ptr,
    Wk_ptr,
    Wv_ptr,
    # Biases (C,)
    Bq_ptr,
    Bk_ptr,
    Bv_ptr,
    # RoPE tables (T, D//2)
    Cos_ptr,
    Sin_ptr,
    # Mask (B, T) uint8 0/1
    Mask_ptr,
    # Outputs (B, T, H, D)
    Qo_ptr,
    Ko_ptr,
    Vo_ptr,
    # Stats (B, T, H) float32 (optional but handy)
    QNorm_ptr,
    KNorm_ptr,
    # Strides (elements)
    stride_qb,
    stride_qt,
    stride_qc,
    stride_kb,
    stride_kt,
    stride_kc,
    stride_vb,
    stride_vt,
    stride_vc,
    stride_cos_t,
    stride_cos_dh,
    stride_sin_t,
    stride_sin_dh,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    stride_mb,
    stride_mt,
    stride_sb,
    stride_st,
    stride_sh,
    # Sizes
    B,
    T,
    C,
    H,
    D,
    # Meta
    BLOCK_DH: tl.constexpr,  # compact RoPE dim (D//2)
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_t = tl.program_id(0)  # token index t
    pid_bh = tl.program_id(1)  # batch-head index

    b = pid_bh // H
    h = pid_bh % H
    t = pid_t

    # bounds
    in_bounds = (b < B) & (t < T)

    # lane indices for interleaved even/odd
    dh = tl.arange(0, BLOCK_DH)  # 0..D/2-1
    idx_e = dh * 2
    idx_o = idx_e + 1

    # channel indices in (B,T,C)
    c_e = h * D + idx_e
    c_o = h * D + idx_o

    # D is even by contract, keep mask for robustness.
    lane_e_ok = idx_e < D
    lane_o_ok = idx_o < D

    # ------------------------------
    # 1) causal depthwise conv(4)
    # tap i corresponds to t-(3-i)
    # ------------------------------
    qe = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    qo = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ke = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ko = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ve = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    vo = tl.zeros((BLOCK_DH,), dtype=tl.float32)

    # Load mask scalar once
    m = tl.load(Mask_ptr + b * stride_mb + t * stride_mt, mask=in_bounds, other=0).to(
        tl.float32
    )

    for i in tl.static_range(4):
        tap_t = t - (3 - i)
        tap_ok = in_bounds & (tap_t >= 0) & (tap_t < T)

        # weights (BLOCK_DH,)
        wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

        # loads (BLOCK_DH,)
        q_pe = tl.load(
            Q_ptr + b * stride_qb + tap_t * stride_qt + c_e * stride_qc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        q_po = tl.load(
            Q_ptr + b * stride_qb + tap_t * stride_qt + c_o * stride_qc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)

        k_pe = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_e * stride_kc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        k_po = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_o * stride_kc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)

        v_pe = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_e * stride_vc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        v_po = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_o * stride_vc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)

        qe += q_pe * wqe
        qo += q_po * wqo
        ke += k_pe * wke
        ko += k_po * wko
        ve += v_pe * wve
        vo += v_po * wvo

    # ------------------------------
    # 2) Bias + SiLU
    # ------------------------------
    # SiLU(x) = x * sigmoid(x)
    bqe = tl.load(Bq_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bqo = tl.load(Bq_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bke = tl.load(Bk_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bko = tl.load(Bk_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bve = tl.load(Bv_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bvo = tl.load(Bv_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)

    qe = qe + bqe
    qo = qo + bqo
    ke = ke + bke
    ko = ko + bko
    ve = ve + bve
    vo = vo + bvo

    qe = qe * tl.sigmoid(qe)
    qo = qo * tl.sigmoid(qo)
    ke = ke * tl.sigmoid(ke)
    ko = ko * tl.sigmoid(ko)
    ve = ve * tl.sigmoid(ve)
    vo = vo * tl.sigmoid(vo)

    # ------------------------------
    # 3) L2 norm over full D
    # ------------------------------
    q_sq = tl.sum(qe * qe + qo * qo, axis=0)
    k_sq = tl.sum(ke * ke + ko * ko, axis=0)
    q_norm = tl.sqrt(q_sq)
    k_norm = tl.sqrt(k_sq)

    q_inv = tl.where(q_norm <= EPS, 1.0 / EPS, 1.0 / q_norm)
    k_inv = tl.where(k_norm <= EPS, 1.0 / EPS, 1.0 / k_norm)

    qne = qe * q_inv
    qno = qo * q_inv
    kne = ke * k_inv
    kno = ko * k_inv

    # ------------------------------
    # 4) RoPE (compact tables are (T, D//2), indexed by dh)
    # ------------------------------
    cos = tl.load(
        Cos_ptr + t * stride_cos_t + dh * stride_cos_dh,
        mask=(in_bounds & lane_e_ok),
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + t * stride_sin_t + dh * stride_sin_dh,
        mask=(in_bounds & lane_e_ok),
        other=0.0,
    ).to(tl.float32)

    qr_e = qne * cos - qno * sin
    qr_o = qne * sin + qno * cos
    kr_e = kne * cos - kno * sin
    kr_o = kne * sin + kno * cos

    # Keep mask semantics aligned with model path: mask after entrance output.
    qr_e = qr_e * m
    qr_o = qr_o * m
    kr_e = kr_e * m
    kr_o = kr_o * m
    ve = ve * m
    vo = vo * m

    # ------------------------------
    # 5) Store to (B, T, H, D) interleaved
    # ------------------------------
    out_base = b * stride_ob + t * stride_ot + h * stride_oh

    tl.store(
        Qo_ptr + out_base + idx_e * stride_od,
        qr_e.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok),
    )
    tl.store(
        Qo_ptr + out_base + idx_o * stride_od,
        qr_o.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok),
    )

    tl.store(
        Ko_ptr + out_base + idx_e * stride_od,
        kr_e.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok),
    )
    tl.store(
        Ko_ptr + out_base + idx_o * stride_od,
        kr_o.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok),
    )

    tl.store(
        Vo_ptr + out_base + idx_e * stride_od,
        ve.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok),
    )
    tl.store(
        Vo_ptr + out_base + idx_o * stride_od,
        vo.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok),
    )

    # stats
    s_off = b * stride_sb + t * stride_st + h * stride_sh
    tl.store(QNorm_ptr + s_off, q_norm, mask=in_bounds)
    tl.store(KNorm_ptr + s_off, k_norm, mask=in_bounds)


# =============================================================================
# Triton Backward Kernel (V19): one (b,h,t) per program
# =============================================================================
@triton.jit
def _delta_entrance_bwd_token_kernel(
    # Inputs
    Q_ptr,
    K_ptr,
    V_ptr,
    Wq_ptr,
    Wk_ptr,
    Wv_ptr,
    Bq_ptr,
    Bk_ptr,
    Bv_ptr,
    Cos_ptr,
    Sin_ptr,
    Mask_ptr,
    # Forward Stats
    QNorm_ptr,
    KNorm_ptr,
    # Gradients of outputs
    DQo_ptr,
    DKo_ptr,
    DVo_ptr,
    # Output Gradients
    DQ_ptr,
    DK_ptr,
    DV_ptr,
    DWq_ptr,
    DWk_ptr,
    DWv_ptr,
    DBq_ptr,
    DBk_ptr,
    DBv_ptr,
    # Strides (elements)
    stride_qb,
    stride_qt,
    stride_qc,
    stride_kb,
    stride_kt,
    stride_kc,
    stride_vb,
    stride_vt,
    stride_vc,
    stride_cos_t,
    stride_cos_dh,
    stride_sin_t,
    stride_sin_dh,
    stride_dqob,
    stride_dqot,
    stride_dqoh,
    stride_dqod,
    stride_mb,
    stride_mt,
    stride_sb,
    stride_st,
    stride_sh,
    # Sizes
    B,
    T,
    C,
    H,
    D,
    # Meta
    BLOCK_DH: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_t = tl.program_id(0)
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

    # ------------------------------
    # 1) Re-compute Forward intermediate (Conv -> Bias -> SiLU -> Norm)
    # ------------------------------
    qe = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    qo = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ke = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ko = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    ve = tl.zeros((BLOCK_DH,), dtype=tl.float32)
    vo = tl.zeros((BLOCK_DH,), dtype=tl.float32)

    m = tl.load(Mask_ptr + b * stride_mb + t * stride_mt, mask=in_bounds, other=0).to(
        tl.float32
    )

    for i in tl.static_range(4):
        tap_t = t - (3 - i)
        tap_ok = in_bounds & (tap_t >= 0) & (tap_t < T)

        wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

        q_pe = tl.load(
            Q_ptr + b * stride_qb + tap_t * stride_qt + c_e * stride_qc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        q_po = tl.load(
            Q_ptr + b * stride_qb + tap_t * stride_qt + c_o * stride_qc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)
        k_pe = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_e * stride_kc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        k_po = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_o * stride_kc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)
        v_pe = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_e * stride_vc,
            mask=(tap_ok & lane_e_ok),
            other=0.0,
        ).to(tl.float32)
        v_po = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_o * stride_vc,
            mask=(tap_ok & lane_o_ok),
            other=0.0,
        ).to(tl.float32)

        qe += q_pe * wqe
        qo += q_po * wqo
        ke += k_pe * wke
        ko += k_po * wko
        ve += v_pe * wve
        vo += v_po * wvo

    xcq_e, xcq_o = qe, qo
    xck_e, xck_o = ke, ko
    xcv_e, xcv_o = ve, vo

    bqe = tl.load(Bq_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bqo = tl.load(Bq_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bke = tl.load(Bk_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bko = tl.load(Bk_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)
    bve = tl.load(Bv_ptr + c_e, mask=lane_e_ok, other=0.0).to(tl.float32)
    bvo = tl.load(Bv_ptr + c_o, mask=lane_o_ok, other=0.0).to(tl.float32)

    xcq_e = xcq_e + bqe
    xcq_o = xcq_o + bqo
    xck_e = xck_e + bke
    xck_o = xck_o + bko
    xcv_e = xcv_e + bve
    xcv_o = xcv_o + bvo

    sqe = tl.sigmoid(xcq_e)
    qe = xcq_e * sqe
    sqo = tl.sigmoid(xcq_o)
    qo = xcq_o * sqo
    ske = tl.sigmoid(xck_e)
    ke = xck_e * ske
    sko = tl.sigmoid(xck_o)
    ko = xck_o * sko
    sve = tl.sigmoid(xcv_e)
    ve = xcv_e * sve
    svo = tl.sigmoid(xcv_o)
    vo = xcv_o * svo

    s_off = b * stride_sb + t * stride_st + h * stride_sh
    q_norm = tl.load(QNorm_ptr + s_off, mask=in_bounds, other=1.0)
    k_norm = tl.load(KNorm_ptr + s_off, mask=in_bounds, other=1.0)
    inv_nq = tl.where(q_norm <= EPS, 1.0 / EPS, 1.0 / q_norm)
    inv_nk = tl.where(k_norm <= EPS, 1.0 / EPS, 1.0 / k_norm)
    qne, qno = qe * inv_nq, qo * inv_nq
    kne, kno = ke * inv_nk, ko * inv_nk

    # ------------------------------
    # 2) RoPE Backward
    # ------------------------------
    cos = tl.load(
        Cos_ptr + t * stride_cos_t + dh * stride_cos_dh,
        mask=(in_bounds & lane_e_ok),
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + t * stride_sin_t + dh * stride_sin_dh,
        mask=(in_bounds & lane_e_ok),
        other=0.0,
    ).to(tl.float32)

    dqo_e = tl.load(
        DQo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_e * stride_dqod,
        mask=(in_bounds & lane_e_ok),
        other=0.0,
    ).to(tl.float32)
    dqo_o = tl.load(
        DQo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_o * stride_dqod,
        mask=(in_bounds & lane_o_ok),
        other=0.0,
    ).to(tl.float32)
    dko_e = tl.load(
        DKo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_e * stride_dqod,
        mask=(in_bounds & lane_e_ok),
        other=0.0,
    ).to(tl.float32)
    dko_o = tl.load(
        DKo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_o * stride_dqod,
        mask=(in_bounds & lane_o_ok),
        other=0.0,
    ).to(tl.float32)
    dvo_e = tl.load(
        DVo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_e * stride_dqod,
        mask=(in_bounds & lane_e_ok),
        other=0.0,
    ).to(tl.float32)
    dvo_o = tl.load(
        DVo_ptr
        + b * stride_dqob
        + t * stride_dqot
        + h * stride_dqoh
        + idx_o * stride_dqod,
        mask=(in_bounds & lane_o_ok),
        other=0.0,
    ).to(tl.float32)

    # Mask is applied at entrance output; reflect that at gradient boundary.
    dqo_e = dqo_e * m
    dqo_o = dqo_o * m
    dko_e = dko_e * m
    dko_o = dko_o * m
    dvo_e = dvo_e * m
    dvo_o = dvo_o * m

    dqne = dqo_e * cos + dqo_o * sin
    dqno = -dqo_e * sin + dqo_o * cos
    dkne = dko_e * cos + dko_o * sin
    dkno = -dko_e * sin + dko_o * cos

    # ------------------------------
    # 3) L2 Norm Backward
    # ------------------------------
    dot_q = tl.sum(qne * dqne + qno * dqno, axis=0)
    dot_k = tl.sum(kne * dkne + kno * dkno, axis=0)

    dqe_nonclamp = (dqne - qne * dot_q) * inv_nq
    dqo_nonclamp = (dqno - qno * dot_q) * inv_nq
    dke_nonclamp = (dkne - kne * dot_k) * inv_nk
    dko_nonclamp = (dkno - kno * dot_k) * inv_nk

    dqe_clamp = dqne * (1.0 / EPS)
    dqo_clamp = dqno * (1.0 / EPS)
    dke_clamp = dkne * (1.0 / EPS)
    dko_clamp = dkno * (1.0 / EPS)

    dqe = tl.where(q_norm <= EPS, dqe_clamp, dqe_nonclamp)
    dqo = tl.where(q_norm <= EPS, dqo_clamp, dqo_nonclamp)
    dke = tl.where(k_norm <= EPS, dke_clamp, dke_nonclamp)
    dko = tl.where(k_norm <= EPS, dko_clamp, dko_nonclamp)

    # ------------------------------
    # 4) SiLU backward
    # ------------------------------
    dsqe = sqe * (1.0 + xcq_e * (1.0 - sqe))
    dsqo = sqo * (1.0 + xcq_o * (1.0 - sqo))
    dske = ske * (1.0 + xck_e * (1.0 - ske))
    dsko = sko * (1.0 + xck_o * (1.0 - sko))
    dsve = sve * (1.0 + xcv_e * (1.0 - sve))
    dsvo = svo * (1.0 + xcv_o * (1.0 - svo))

    dqc_e = dqe * dsqe
    dqc_o = dqo * dsqo
    dkc_e = dke * dske
    dkc_o = dko * dsko
    dvc_e = dvo_e * dsve
    dvc_o = dvo_o * dsvo

    # ------------------------------
    # 5) Conv1d Backward & Weight Grads (Atomic)
    # ------------------------------
    for i in tl.static_range(4):
        prev_t = t - (3 - i)
        prev_ok = in_bounds & (prev_t >= 0)

        wqe = tl.load(Wq_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wqo = tl.load(Wq_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wke = tl.load(Wk_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wko = tl.load(Wk_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)
        wve = tl.load(Wv_ptr + c_e * 4 + i, mask=lane_e_ok, other=0.0).to(tl.float32)
        wvo = tl.load(Wv_ptr + c_o * 4 + i, mask=lane_o_ok, other=0.0).to(tl.float32)

        off_qe = b * stride_qb + prev_t * stride_qt + c_e * stride_qc
        off_qo = b * stride_qb + prev_t * stride_qt + c_o * stride_qc
        off_ke = b * stride_kb + prev_t * stride_kt + c_e * stride_kc
        off_ko = b * stride_kb + prev_t * stride_kt + c_o * stride_kc
        off_ve = b * stride_vb + prev_t * stride_vt + c_e * stride_vc
        off_vo = b * stride_vb + prev_t * stride_vt + c_o * stride_vc

        tl.atomic_add(
            DQ_ptr + off_qe, (dqc_e * wqe).to(tl.float32), mask=(prev_ok & lane_e_ok)
        )
        tl.atomic_add(
            DQ_ptr + off_qo, (dqc_o * wqo).to(tl.float32), mask=(prev_ok & lane_o_ok)
        )
        tl.atomic_add(
            DK_ptr + off_ke, (dkc_e * wke).to(tl.float32), mask=(prev_ok & lane_e_ok)
        )
        tl.atomic_add(
            DK_ptr + off_ko, (dkc_o * wko).to(tl.float32), mask=(prev_ok & lane_o_ok)
        )
        tl.atomic_add(
            DV_ptr + off_ve, (dvc_e * wve).to(tl.float32), mask=(prev_ok & lane_e_ok)
        )
        tl.atomic_add(
            DV_ptr + off_vo, (dvc_o * wvo).to(tl.float32), mask=(prev_ok & lane_o_ok)
        )

        q_pe = tl.load(Q_ptr + off_qe, mask=(prev_ok & lane_e_ok), other=0.0).to(
            tl.float32
        )
        q_po = tl.load(Q_ptr + off_qo, mask=(prev_ok & lane_o_ok), other=0.0).to(
            tl.float32
        )
        k_pe = tl.load(K_ptr + off_ke, mask=(prev_ok & lane_e_ok), other=0.0).to(
            tl.float32
        )
        k_po = tl.load(K_ptr + off_ko, mask=(prev_ok & lane_o_ok), other=0.0).to(
            tl.float32
        )
        v_pe = tl.load(V_ptr + off_ve, mask=(prev_ok & lane_e_ok), other=0.0).to(
            tl.float32
        )
        v_po = tl.load(V_ptr + off_vo, mask=(prev_ok & lane_o_ok), other=0.0).to(
            tl.float32
        )

        tl.atomic_add(
            DWq_ptr + c_e * 4 + i,
            (dqc_e * q_pe).to(tl.float32),
            mask=(in_bounds & lane_e_ok),
        )
        tl.atomic_add(
            DWq_ptr + c_o * 4 + i,
            (dqc_o * q_po).to(tl.float32),
            mask=(in_bounds & lane_o_ok),
        )
        tl.atomic_add(
            DWk_ptr + c_e * 4 + i,
            (dkc_e * k_pe).to(tl.float32),
            mask=(in_bounds & lane_e_ok),
        )
        tl.atomic_add(
            DWk_ptr + c_o * 4 + i,
            (dkc_o * k_po).to(tl.float32),
            mask=(in_bounds & lane_o_ok),
        )
        tl.atomic_add(
            DWv_ptr + c_e * 4 + i,
            (dvc_e * v_pe).to(tl.float32),
            mask=(in_bounds & lane_e_ok),
        )
        tl.atomic_add(
            DWv_ptr + c_o * 4 + i,
            (dvc_o * v_po).to(tl.float32),
            mask=(in_bounds & lane_o_ok),
        )

    # Bias grads are sum of post-SiLU conv grads over tokens and batch.
    tl.atomic_add(DBq_ptr + c_e, dqc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBq_ptr + c_o, dqc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))
    tl.atomic_add(DBk_ptr + c_e, dkc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBk_ptr + c_o, dkc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))
    tl.atomic_add(DBv_ptr + c_e, dvc_e.to(tl.float32), mask=(in_bounds & lane_e_ok))
    tl.atomic_add(DBv_ptr + c_o, dvc_o.to(tl.float32), mask=(in_bounds & lane_o_ok))


# =============================================================================
# Autograd wrapper (forward Triton, backward analytic)
# =============================================================================
class TritonDeltaEntranceV19(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps=1e-6):
        B, T, C = q.shape
        assert (
            cos.ndim == 2 and sin.ndim == 2 and cos.shape == sin.shape
        ), "cos/sin must be shape (T, D//2)"
        if cos.shape[0] != T:
            raise ValueError(
                f"RoPE T mismatch: cos/sin T={cos.shape[0]} vs input T={T}"
            )
        DH = cos.shape[1]
        D = DH * 2
        if C % D != 0:
            raise ValueError(
                f"C={C} must be divisible by D={D} (derived from compact RoPE)"
            )
        H = C // D

        # weights -> (C, 4) expected by kernel
        # Keep track of original ndim so backward can return matching grad shape.
        wq_was_3d = wq.ndim == 3
        wk_was_3d = wk.ndim == 3
        wv_was_3d = wv.ndim == 3
        if wq.ndim == 3:
            if (
                wq.size(1) != 1
                or wk.size(1) != 1
                or wv.size(1) != 1
                or wq.size(2) != 4
                or wk.size(2) != 4
                or wv.size(2) != 4
            ):
                raise ValueError("Expected depthwise conv weights with shape [C, 1, 4]")
            wq = wq.squeeze(1)
            wk = wk.squeeze(1)
            wv = wv.squeeze(1)
        else:
            if wq.shape != (C, 4) or wk.shape != (C, 4) or wv.shape != (C, 4):
                raise ValueError("Expected depthwise conv weights with shape [C, 4]")
        wq = wq.contiguous()
        wk = wk.contiguous()
        wv = wv.contiguous()

        # Require all biases or none to avoid accidental silent parity skew.
        if (bq is None) != (bk is None) or (bk is None) != (bv is None):
            raise ValueError("Either provide all biases (bq,bk,bv) or all None.")
        has_bq = bq is not None
        has_bk = bk is not None
        has_bv = bv is not None
        if bq is None:
            bq = torch.zeros((C,), device=q.device, dtype=q.dtype)
            bk = torch.zeros((C,), device=q.device, dtype=q.dtype)
            bv = torch.zeros((C,), device=q.device, dtype=q.dtype)
        bq = bq.contiguous()
        bk = bk.contiguous()
        bv = bv.contiguous()

        # mask -> uint8 0/1
        if mask is None:
            mask_u8 = torch.ones((B, T), device=q.device, dtype=torch.uint8)
        elif mask.dtype == torch.bool:
            mask_u8 = mask.to(torch.uint8)
        elif mask.dtype == torch.uint8:
            mask_u8 = mask.contiguous()
        else:
            mask_u8 = (mask != 0).to(torch.uint8)
        cos = cos.contiguous()
        sin = sin.contiguous()

        qo = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        ko = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        vo = torch.empty((B, T, H, D), device=q.device, dtype=q.dtype)
        q_norm = torch.empty((B, T, H), device=q.device, dtype=torch.float32)
        k_norm = torch.empty((B, T, H), device=q.device, dtype=torch.float32)

        BLOCK_DH = DH
        grid = (T, B * H)

        if q.dtype == torch.bfloat16:
            out_dtype = tl.bfloat16
        elif q.dtype == torch.float16:
            out_dtype = tl.float16
        else:
            out_dtype = tl.float32

        with kernel_region("delta_entrance_fwd"):
            _delta_entrance_fwd_token_kernel[grid](
                q,
                k,
                v,
                wq,
                wk,
                wv,
                bq,
                bk,
                bv,
                cos,
                sin,
                mask_u8,
                qo,
                ko,
                vo,
                q_norm,
                k_norm,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                cos.stride(0),
                cos.stride(1),
                sin.stride(0),
                sin.stride(1),
                qo.stride(0),
                qo.stride(1),
                qo.stride(2),
                qo.stride(3),
                mask_u8.stride(0),
                mask_u8.stride(1),
                q_norm.stride(0),
                q_norm.stride(1),
                q_norm.stride(2),
                B,
                T,
                C,
                H,
                D,
                BLOCK_DH=BLOCK_DH,
                EPS=eps,
                OUT_DTYPE=out_dtype,
                # tuning knobs
                num_warps=4,
                num_stages=2,
            )

        ctx.save_for_backward(
            q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask_u8, q_norm, k_norm
        )
        ctx.eps = eps
        ctx.wq_was_3d = wq_was_3d
        ctx.wk_was_3d = wk_was_3d
        ctx.wv_was_3d = wv_was_3d
        ctx.has_bq = has_bq
        ctx.has_bk = has_bk
        ctx.has_bv = has_bv
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask_u8, q_norm, k_norm = (
            ctx.saved_tensors
        )
        B, T, C = q.shape
        D = cos.shape[1] * 2
        H = C // D
        eps = ctx.eps

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dwq = torch.zeros_like(wq, dtype=torch.float32)
        dwk = torch.zeros_like(wk, dtype=torch.float32)
        dwv = torch.zeros_like(wv, dtype=torch.float32)
        dbq = torch.zeros_like(bq, dtype=torch.float32)
        dbk = torch.zeros_like(bk, dtype=torch.float32)
        dbv = torch.zeros_like(bv, dtype=torch.float32)

        BLOCK_DH = D // 2
        grid = (T, B * H)

        with kernel_region("delta_entrance_bwd"):
            _delta_entrance_bwd_token_kernel[grid](
                q,
                k,
                v,
                wq,
                wk,
                wv,
                bq,
                bk,
                bv,
                cos,
                sin,
                mask_u8,
                q_norm,
                k_norm,
                dqo,
                dko,
                dvo,
                dq,
                dk,
                dv,
                dwq,
                dwk,
                dwv,
                dbq,
                dbk,
                dbv,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                cos.stride(0),
                cos.stride(1),
                sin.stride(0),
                sin.stride(1),
                dqo.stride(0),
                dqo.stride(1),
                dqo.stride(2),
                dqo.stride(3),
                mask_u8.stride(0),
                mask_u8.stride(1),
                q_norm.stride(0),
                q_norm.stride(1),
                q_norm.stride(2),
                B,
                T,
                C,
                H,
                D,
                BLOCK_DH=BLOCK_DH,
                EPS=eps,
                num_warps=4,
                num_stages=2,
            )

        dwq_out = dwq.to(wq.dtype)
        dwk_out = dwk.to(wk.dtype)
        dwv_out = dwv.to(wv.dtype)
        if ctx.wq_was_3d:
            dwq_out = dwq_out.unsqueeze(1)
        if ctx.wk_was_3d:
            dwk_out = dwk_out.unsqueeze(1)
        if ctx.wv_was_3d:
            dwv_out = dwv_out.unsqueeze(1)

        dbq_out = dbq.to(bq.dtype) if ctx.has_bq else None
        dbk_out = dbk.to(bk.dtype) if ctx.has_bk else None
        dbv_out = dbv.to(bv.dtype) if ctx.has_bv else None

        return (
            dq.to(q.dtype),
            dk.to(k.dtype),
            dv.to(v.dtype),
            dwq_out,
            dwk_out,
            dwv_out,
            dbq_out,
            dbk_out,
            dbv_out,
            None,
            None,
            None,
            None,
        )


def fused_delta_entrance(
    q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask=None, eps=1e-6
):
    return TritonDeltaEntranceV19.apply(
        q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask, eps
    )


# =============================================================================
# Benchmark harness (same interface style as yours)
# =============================================================================
if __name__ == "__main__":
    import triton.testing

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["T"],
            x_vals=[512, 1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=["pytorch", "triton_v19"],
            line_names=["PyTorch (Unfused)", "Triton V19 (Token-Program)"],
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
        cos = torch.randn((T, D // 2), device=device, dtype=dtype)
        sin = torch.randn((T, D // 2), device=device, dtype=dtype)
        mask = torch.ones((B, T), device=device, dtype=torch.uint8)

        # warmup (important for triton JIT + caching)
        if provider == "triton_v19":
            fused_delta_entrance(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)
        else:
            pytorch_unfused_exact(q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "pytorch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: pytorch_unfused_exact(
                    q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask
                ),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_delta_entrance(
                    q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask
                ),
                quantiles=quantiles,
            )

        return ms, max_ms, min_ms

    # quick correctness smoke test (small)
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
    cos = torch.randn((T, D // 2), device=device, dtype=dtype)
    sin = torch.randn((T, D // 2), device=device, dtype=dtype)
    mask = torch.ones((B, T), device=device, dtype=torch.uint8)

    with torch.no_grad():
        qo_ref, ko_ref, vo_ref = pytorch_unfused_exact(
            q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask
        )
        qo_tri, ko_tri, vo_tri = fused_delta_entrance(
            q, k, v, wq, wk, wv, bq, bk, bv, cos, sin, mask
        )

        # Compare in fp32 for tolerances
        def max_abs(a, b):
            return (a.float() - b.float()).abs().max().item()

        print("max|Qo-ref|:", max_abs(qo_ref, qo_tri))
        print("max|Ko-ref|:", max_abs(ko_ref, ko_tri))
        print("max|Vo-ref|:", max_abs(vo_ref, vo_tri))

    benchmark.run(show_plots=True, print_data=True)
