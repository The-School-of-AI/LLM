# triton_delta_entrance_v18.py
# =============================================================================
# V18: "Token-Program" Delta Entrance (Forward fused)
# Fuses: causal depthwise conv(4) + SiLU + mask + L2Norm(Q/K) + interleaved RoPE(Q/K)
#
# Key change vs your V17:
# - One program instance computes ONE token t for ONE (batch, head).
# - Vector width is D/2 (even + odd lanes) so register footprint is O(D), not O(BLOCK_T*D).
# - This avoids the giant fp32 tiles that were almost certainly spilling at large T.
#
# Outputs: (B, T, H, D) like your current wrapper.
# Backward: oracle PyTorch recompute (same as your previous version).
# =============================================================================

import torch
import triton
import triton.language as tl
import torch.nn.functional as F


# =============================================================================
# Reference (Unfused) for correctness + benchmark oracle
# =============================================================================
def pytorch_unfused(q, k, v, wq, wk, wv, cos, sin, mask, eps=1e-6):
    B, T, C = q.shape
    D = cos.shape[1]
    H = C // D

    # 1) depthwise causal conv (4 taps)
    qc = F.conv1d(q.transpose(1, 2), wq.view(C, 1, 4), groups=C, padding=3)[..., :-3].transpose(1, 2)
    kc = F.conv1d(k.transpose(1, 2), wk.view(C, 1, 4), groups=C, padding=3)[..., :-3].transpose(1, 2)
    vc = F.conv1d(v.transpose(1, 2), wv.view(C, 1, 4), groups=C, padding=3)[..., :-3].transpose(1, 2)

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
# Triton Kernel (V18): one (b,h,t) per program
# =============================================================================
@triton.jit
def _delta_entrance_fwd_token_kernel(
    # Inputs (B, T, C)
    Q_ptr, K_ptr, V_ptr,
    # Weights (C, 4)  (NOTE: wrapper will squeeze if (C,1,4))
    Wq_ptr, Wk_ptr, Wv_ptr,
    # RoPE tables (T, D)
    Cos_ptr, Sin_ptr,
    # Mask (B, T) uint8 0/1
    Mask_ptr,

    # Outputs (B, T, H, D)
    Qo_ptr, Ko_ptr, Vo_ptr,
    # Stats (B, T, H) float32 (optional but handy)
    InvNq_ptr, InvNk_ptr,

    # Strides (elements)
    stride_qb, stride_qt, stride_qc,
    stride_kb, stride_kt, stride_kc,
    stride_vb, stride_vt, stride_vc,
    stride_ob, stride_ot, stride_oh, stride_od,
    stride_mb, stride_mt,
    stride_sb, stride_st, stride_sh,

    # Sizes
    B, T, C, H, D,

    # Meta
    BLOCK_DH: tl.constexpr,   # D//2
    EPS: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_t  = tl.program_id(0)  # token index t
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

    # mask for lanes (D is even by contract, but keep safe)
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
    m = tl.load(
        Mask_ptr + b * stride_mb + t * stride_mt,
        mask=in_bounds,
        other=0
    ).to(tl.float32)

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
            other=0.0
        ).to(tl.float32)
        q_po = tl.load(
            Q_ptr + b * stride_qb + tap_t * stride_qt + c_o * stride_qc,
            mask=(tap_ok & lane_o_ok),
            other=0.0
        ).to(tl.float32)

        k_pe = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_e * stride_kc,
            mask=(tap_ok & lane_e_ok),
            other=0.0
        ).to(tl.float32)
        k_po = tl.load(
            K_ptr + b * stride_kb + tap_t * stride_kt + c_o * stride_kc,
            mask=(tap_ok & lane_o_ok),
            other=0.0
        ).to(tl.float32)

        v_pe = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_e * stride_vc,
            mask=(tap_ok & lane_e_ok),
            other=0.0
        ).to(tl.float32)
        v_po = tl.load(
            V_ptr + b * stride_vb + tap_t * stride_vt + c_o * stride_vc,
            mask=(tap_ok & lane_o_ok),
            other=0.0
        ).to(tl.float32)

        qe += q_pe * wqe
        qo += q_po * wqo
        ke += k_pe * wke
        ko += k_po * wko
        ve += v_pe * wve
        vo += v_po * wvo

    # ------------------------------
    # 2) SiLU + mask
    # ------------------------------
    # SiLU(x) = x * sigmoid(x)
    qe = (qe * tl.sigmoid(qe)) * m
    qo = (qo * tl.sigmoid(qo)) * m
    ke = (ke * tl.sigmoid(ke)) * m
    ko = (ko * tl.sigmoid(ko)) * m
    ve = (ve * tl.sigmoid(ve)) * m
    vo = (vo * tl.sigmoid(vo)) * m

    # ------------------------------
    # 3) L2 norm over full D
    # ------------------------------
    q_inv = tl.rsqrt(tl.sum(qe * qe + qo * qo, axis=0) + EPS)
    k_inv = tl.rsqrt(tl.sum(ke * ke + ko * ko, axis=0) + EPS)

    qne = qe * q_inv
    qno = qo * q_inv
    kne = ke * k_inv
    kno = ko * k_inv

    # ------------------------------
    # 4) RoPE (tables are (T, D), but only even lanes used like your PyTorch)
    # ------------------------------
    cos = tl.load(
        Cos_ptr + t * D + idx_e,
        mask=(in_bounds & lane_e_ok),
        other=1.0
    ).to(tl.float32)
    sin = tl.load(
        Sin_ptr + t * D + idx_e,
        mask=(in_bounds & lane_e_ok),
        other=0.0
    ).to(tl.float32)

    qr_e = qne * cos - qno * sin
    qr_o = qne * sin + qno * cos
    kr_e = kne * cos - kno * sin
    kr_o = kne * sin + kno * cos

    # ------------------------------
    # 5) Store to (B, T, H, D) interleaved
    # ------------------------------
    out_base = b * stride_ob + t * stride_ot + h * stride_oh

    tl.store(
        Qo_ptr + out_base + idx_e * stride_od,
        qr_e.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok)
    )
    tl.store(
        Qo_ptr + out_base + idx_o * stride_od,
        qr_o.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok)
    )

    tl.store(
        Ko_ptr + out_base + idx_e * stride_od,
        kr_e.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok)
    )
    tl.store(
        Ko_ptr + out_base + idx_o * stride_od,
        kr_o.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok)
    )

    tl.store(
        Vo_ptr + out_base + idx_e * stride_od,
        ve.to(OUT_DTYPE),
        mask=(in_bounds & lane_e_ok)
    )
    tl.store(
        Vo_ptr + out_base + idx_o * stride_od,
        ve.to(OUT_DTYPE), # Wait, looking at original code in step 12 line 258: tl.store(..., ve.to(OUT_DTYPE), ...) and 259: tl.store(..., vo.to(OUT_DTYPE), ...)
        # Re-checking step 12 line 258/259:
        # 258:         ve.to(OUT_DTYPE),
        # 259:         mask=(in_bounds & lane_e_ok)
        # 260:     )
        # 261:     tl.store(
        # 262:         Vo_ptr + out_base + idx_o * stride_od,
        # 263:         vo.to(OUT_DTYPE),
        # 264:         mask=(in_bounds & lane_o_ok)
        # 265:     )
        # Ah, I see. Let me correct my manual type-in.
        mask=(in_bounds & lane_e_ok)
    )
    tl.store(
        Vo_ptr + out_base + idx_o * stride_od,
        vo.to(OUT_DTYPE),
        mask=(in_bounds & lane_o_ok)
    )

    # stats
    s_off = b * stride_sb + t * stride_st + h * stride_sh
    tl.store(InvNq_ptr + s_off, q_inv, mask=in_bounds)
    tl.store(InvNk_ptr + s_off, k_inv, mask=in_bounds)


# =============================================================================
# Autograd wrapper (forward Triton, backward oracle recompute)
# =============================================================================
class TritonDeltaEntranceV18(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, wq, wk, wv, cos, sin, mask, eps=1e-6):
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
        grid = (T, B * H)

        out_dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

        _delta_entrance_fwd_token_kernel[grid](
            q, k, v,
            wq, wk, wv,
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

            BLOCK_DH=BLOCK_DH,
            EPS=eps,
            OUT_DTYPE=out_dtype,

            # tuning knobs
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(q, k, v, wq, wk, wv, cos, sin, mask_u8)
        ctx.eps = eps
        return qo, ko, vo

    @staticmethod
    def backward(ctx, dqo, dko, dvo):
        # Oracle fallback recompute (correct, slow)
        q_raw, k_raw, v_raw, wq, wk, wv, cos, sin, mask_u8 = ctx.saved_tensors
        B, T, C = q_raw.shape
        D = cos.shape[1]
        H = C // D
        eps = ctx.eps

        with torch.enable_grad():
            qi, ki, vi = [x.detach().requires_grad_(True) for x in (q_raw, k_raw, v_raw)]
            wqi, wki, wvi = [x.detach().requires_grad_(True) for x in (wq, wk, wv)]

            def causal_conv(x, w):
                return torch.nn.functional.conv1d(
                    x.transpose(1, 2),
                    w.view(C, 1, 4),
                    groups=C,
                    padding=3
                )[:, :, :-3].transpose(1, 2)

            m = mask_u8.to(qi.dtype).unsqueeze(-1)
            qc = torch.nn.functional.silu(causal_conv(qi, wqi)) * m
            kc = torch.nn.functional.silu(causal_conv(ki, wki)) * m
            vc = torch.nn.functional.silu(causal_conv(vi, wvi)) * m

            qn = torch.nn.functional.normalize(qc.view(B, T, H, D), p=2, dim=-1, eps=eps)
            kn = torch.nn.functional.normalize(kc.view(B, T, H, D), p=2, dim=-1, eps=eps)

            def apply_rope(x):
                c = cos[:, 0::2].unsqueeze(0).unsqueeze(2)
                s = sin[:, 0::2].unsqueeze(0).unsqueeze(2)
                xe, xo = x[..., 0::2], x[..., 1::2]
                out_e = xe * c - xo * s
                out_o = xe * s + xo * c
                return torch.stack((out_e, out_o), dim=-1).flatten(-2)

            qo = apply_rope(qn)
            ko = apply_rope(kn)
            vo = vc.view(B, T, H, D)

            torch.autograd.backward([qo, ko, vo], [dqo, dko, dvo])

            return qi.grad, ki.grad, vi.grad, wqi.grad, wki.grad, wvi.grad, None, None, None, None


def fused_delta_entrance(q, k, v, wq, wk, wv, cos, sin, mask, eps=1e-6):
    return TritonDeltaEntranceV18.apply(q, k, v, wq, wk, wv, cos, sin, mask, eps)


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
            line_vals=["pytorch", "triton_v18"],
            line_names=["PyTorch (Unfused)", "Triton V18 (Token-Program)"],
            styles=[("red", "-"), ("green", "-")],
            ylabel="Execution Time (ms)",
            plot_name="Delta-Entrance Performance (Forward Pass) - V18",
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
        cos = torch.randn((T, D), device=device, dtype=dtype)
        sin = torch.randn((T, D), device=device, dtype=dtype)
        mask = torch.ones((B, T), device=device, dtype=torch.uint8)

        # warmup (important for triton JIT + caching)
        if provider == "triton_v18":
            fused_delta_entrance(q, k, v, wq, wk, wv, cos, sin, mask)
        else:
            pytorch_unfused(q, k, v, wq, wk, wv, cos, sin, mask)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "pytorch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: pytorch_unfused(q, k, v, wq, wk, wv, cos, sin, mask),
                quantiles=quantiles,
            )
        else:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: fused_delta_entrance(q, k, v, wq, wk, wv, cos, sin, mask),
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
    cos = torch.randn((T, D), device=device, dtype=dtype)
    sin = torch.randn((T, D), device=device, dtype=dtype)
    mask = torch.ones((B, T), device=device, dtype=torch.uint8)

    with torch.no_grad():
        qo_ref, ko_ref, vo_ref = pytorch_unfused(q, k, v, wq, wk, wv, cos, sin, mask)
        qo_tri, ko_tri, vo_tri = fused_delta_entrance(q, k, v, wq, wk, wv, cos, sin, mask)

        # Compare in fp32 for tolerances
        def max_abs(a, b):
            return (a.float() - b.float()).abs().max().item()

        print("max|Qo-ref|:", max_abs(qo_ref, qo_tri))
        print("max|Ko-ref|:", max_abs(ko_ref, ko_tri))
        print("max|Vo-ref|:", max_abs(vo_ref, vo_tri))

    benchmark.run(show_plots=True, print_data=True)
