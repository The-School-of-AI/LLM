"""
Triton Sparse Attention Kernel — V2 (Tensor Core Optimized)
===========================================================

Based on V1 (Rohan's baseline) with the following optimizations:
- tl.dot for QK scores and PV accumulation (Tensor Cores, ~10× throughput)
- tl.dot in backward dQ (3 operations) and dK/dV (2 operations)
- num_warps=8 for D>64, num_stages=3 for pipeline depth
- Conditional fallback to tl.sum for BLOCK_K < 16 (small test dims)

All V1 safety features preserved:
- NaN-safe softmax (handles fully-masked rows)
- Index bounds checking (qi_indices < seq_kv)
- Input sanitization (clamping masked-out indices)
- Separate seq_kv parameter
"""

import torch
import torch.nn.functional as F
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel — V2 with tl.dot
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_fwd_kernel_v2(
        Q_ptr, K_ptr, V_ptr, IDX_ptr, MASK_ptr,
        OUT_ptr, LSE_ptr,
        batch_size,
        seq_q, seq_kv, n_heads, d_head, k_selected,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        stride_ob, stride_oq, stride_oh, stride_od,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Sparse attention forward — V2 with tl.dot for Tensor Core usage.
        Grid: (batch_size * n_heads, seq_q)
        """
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh  % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        # load query vector
        q_row_ptr = (Q_ptr
                     + pid_b * stride_qb
                     + pid_q * stride_qq
                     + pid_h * stride_qh)
        q_i = tl.load(q_row_ptr + d_offs * stride_qd,
                       mask=d_offs < d_head, other=0.0).to(tl.float32)

        # online softmax accumulators
        m_i = tl.full((1,), float('-inf'), dtype=tl.float32)
        l_i = tl.full((1,), 0.0,          dtype=tl.float32)
        acc = tl.zeros((BLOCK_D,),         dtype=tl.float32)

        idx_row_ptr  = IDX_ptr  + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row_ptr = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs

            idx_load_mask = k_block_offs < k_selected
            qi_indices = tl.load(idx_row_ptr + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row_ptr + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            qi_mask = (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            # gather K, V via indirect load → [BLOCK_K, BLOCK_D]
            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_load_mask = qi_mask[:, None] & (d_offs[None, :] < d_head)
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)

            # ── V2: tl.dot for QK scores ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                q_2d = tl.reshape(q_i, (1, BLOCK_D))
                scores_2d = tl.dot(q_2d, tl.trans(k_vals))  # [1, BLOCK_K]
                scores = tl.reshape(scores_2d, (BLOCK_K,)) * scale
            else:
                scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale

            valid = idx_load_mask & qi_mask
            scores = tl.where(valid, scores, float('-inf'))

            # online softmax update (NaN-safe)
            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(m_new == float('-inf'), 0.0, tl.exp(m_i - m_new))
            beta  = tl.where(m_new == float('-inf'), 0.0, tl.exp(scores - m_new))

            l_i = alpha * l_i + tl.sum(beta, axis=0)

            # ── V2: tl.dot for PV accumulation ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                beta_row = tl.reshape(beta, (1, BLOCK_K))
                pv_2d = tl.dot(beta_row, v_vals)  # [1, BLOCK_D]
                pv = tl.reshape(pv_2d, (BLOCK_D,))
                acc = alpha * acc + pv
            else:
                acc = alpha * acc + tl.sum(beta[:, None] * v_vals, axis=0)

            m_i = m_new

        # normalise (safe division)
        l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i_safe

        # store output
        out_row_ptr = (OUT_ptr
                       + pid_b * stride_ob
                       + pid_q * stride_oq
                       + pid_h * stride_oh)
        tl.store(out_row_ptr + d_offs * stride_od, acc,
                 mask=d_offs < d_head)

        # store LSE
        lse_ptr = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + pid_q
        tl.store(lse_ptr + tl.arange(0, 1), m_i + tl.log(l_i))


# ═══════════════════════════════════════════════════════════════════════
# Backward kernels — V2 with tl.dot
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_bwd_preprocess_v2(
        O_ptr, DO_ptr, DELTA_ptr,
        seq_len, n_heads, d_head,
        stride_ob, stride_oq, stride_oh, stride_od,
        stride_dob, stride_doq, stride_doh, stride_dod,
        BLOCK_D: tl.constexpr,
    ):
        """delta[b,h,q] = sum_d( O * dO ). Grid: (B*H, T)"""
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        o_base = O_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh

        o_vals = tl.load(o_base + d_offs * stride_od, mask=d_mask, other=0.0).to(tl.float32)
        do_vals = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        delta = tl.sum(o_vals * do_vals, axis=0)
        delta_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        tl.store(DELTA_ptr + delta_offset, delta)

    @triton.jit
    def _sparse_attn_bwd_dq_kernel_v2(
        Q_ptr, K_ptr, V_ptr, DO_ptr,
        IDX_ptr, MASK_ptr,
        LSE_ptr, DELTA_ptr,
        DQ_ptr,
        seq_len, seq_kv, n_heads, d_head, k_selected,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_dob, stride_doq, stride_doh, stride_dod,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        stride_dqb, stride_dqq, stride_dqh, stride_dqd,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """dQ kernel — V2 with tl.dot. Grid: (B*H, T)"""
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        d_mask = d_offs < d_head

        q_base = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)
        delta_i = tl.load(DELTA_ptr + ld_offset)

        idx_row = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq

        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            qi_indices = tl.load(idx_row + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # ── V2: tl.dot for scores ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                q_2d = tl.reshape(q_i, (1, BLOCK_D))
                scores_2d = tl.dot(q_2d, tl.trans(k_vals))
                scores = tl.reshape(scores_2d, (BLOCK_K,)) * scale
            else:
                scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale

            scores = tl.where(valid, scores, float('-inf'))
            p_i = tl.exp(scores - lse_i)
            p_i = tl.where(valid, p_i, 0.0)

            # ── V2: tl.dot for dO·V ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                do_2d = tl.reshape(do_i, (1, BLOCK_D))
                do_v_2d = tl.dot(do_2d, tl.trans(v_vals))
                do_v = tl.reshape(do_v_2d, (BLOCK_K,))
            else:
                do_v = tl.sum(do_i[None, :] * v_vals, axis=1)

            ds_i = p_i * (do_v - delta_i)

            # ── V2: tl.dot for dS·K ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                ds_row = tl.reshape(ds_i, (1, BLOCK_K))
                dq_2d = tl.dot(ds_row, k_vals)
                dq_acc += tl.reshape(dq_2d, (BLOCK_D,)) * scale
            else:
                dq_acc += tl.sum(ds_i[:, None] * k_vals, axis=0) * scale

        dq_base = DQ_ptr + pid_b * stride_dqb + pid_q * stride_dqq + pid_h * stride_dqh
        tl.store(dq_base + d_offs * stride_dqd, dq_acc, mask=d_mask)

    @triton.jit
    def _sparse_attn_bwd_dkdv_kernel_v2(
        Q_ptr, K_ptr, V_ptr, DO_ptr,
        IDX_ptr, MASK_ptr,
        LSE_ptr, DELTA_ptr,
        DK_ptr, DV_ptr,
        seq_len, seq_kv, n_heads, d_head, k_selected,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_dob, stride_doq, stride_doh, stride_dod,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        stride_dkb, stride_dkk, stride_dkh, stride_dkd,
        stride_dvb, stride_dvk, stride_dvh, stride_dvd,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """dK/dV kernel — V2 with tl.dot for scores. Atomics kept. Grid: (B*H, T)"""
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        d_mask = d_offs < d_head

        q_base = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)
        delta_i = tl.load(DELTA_ptr + ld_offset)

        idx_row = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq

        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        dk_base = DK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
        dv_base = DV_ptr + pid_b * stride_dvb + pid_h * stride_dvh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            qi_indices = tl.load(idx_row + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # ── V2: tl.dot for scores ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                q_2d = tl.reshape(q_i, (1, BLOCK_D))
                scores_2d = tl.dot(q_2d, tl.trans(k_vals))
                scores = tl.reshape(scores_2d, (BLOCK_K,)) * scale
            else:
                scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale

            scores = tl.where(valid, scores, float('-inf'))
            p_i = tl.exp(scores - lse_i)
            p_i = tl.where(valid, p_i, 0.0)

            # ── V2: tl.dot for dO·V ──
            if BLOCK_K >= 16 and BLOCK_D >= 16:
                do_2d = tl.reshape(do_i, (1, BLOCK_D))
                do_v_2d = tl.dot(do_2d, tl.trans(v_vals))
                do_v = tl.reshape(do_v_2d, (BLOCK_K,))
            else:
                do_v = tl.sum(do_i[None, :] * v_vals, axis=1)

            ds_i = p_i * (do_v - delta_i)

            # ── Atomic scatter (same as V1) ──
            dk_contrib = (ds_i[:, None] * q_i[None, :]) * scale
            dk_ptrs = dk_base + qi_indices[:, None] * stride_dkk + d_offs[None, :] * stride_dkd
            scatter_mask = valid[:, None] & d_mask[None, :]
            tl.atomic_add(dk_ptrs, dk_contrib, mask=scatter_mask)

            dv_contrib = p_i[:, None] * do_i[None, :]
            dv_ptrs = dv_base + qi_indices[:, None] * stride_dvk + d_offs[None, :] * stride_dvd
            tl.atomic_add(dv_ptrs, dv_contrib, mask=scatter_mask)


# ═══════════════════════════════════════════════════════════════════════
# torch.autograd.Function wrapper — V2
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    class TritonSparseAttnFnV2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, indices, mask, scale):
            B, T, H, D = q.shape
            k_sel = indices.size(-1)

            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if mask.dtype != torch.float32:
                mask = mask.to(torch.float32)

            out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
            lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

            BLOCK_K = triton.next_power_of_2(min(64, k_sel))
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * H, T)

            # T4 has 64KB shared memory; 3 stages with BLOCK_D>=128 exceeds it
            num_stages = 2 if BLOCK_D >= 128 else 3

            _sparse_attn_fwd_kernel_v2[grid](
                q, k, v, indices, mask,
                out, lse,
                B, T, T, H, D, k_sel,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                scale,
                BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                num_warps=4 if BLOCK_D <= 64 else 8,
                num_stages=num_stages,
            )

            out_typed = out.to(q.dtype)

            ctx.save_for_backward(q, k, v, indices, mask, out, lse)
            ctx.scale = scale
            ctx.BLOCK_K = BLOCK_K
            ctx.BLOCK_D = BLOCK_D

            return out_typed

        @staticmethod
        def backward(ctx, grad_output):
            q, k, v, indices, mask, out_fp32, lse = ctx.saved_tensors
            scale = ctx.scale
            BLOCK_K = ctx.BLOCK_K
            BLOCK_D = ctx.BLOCK_D

            B, T, H, D = q.shape
            T_kv = k.shape[1]
            k_sel = indices.size(-1)
            grid = (B * H, T)

            do = grad_output.contiguous().to(torch.float32)
            num_warps_bwd = 4 if BLOCK_D <= 64 else 8
            num_stages_bwd = 2 if BLOCK_D >= 128 else 3

            # Step 1: delta
            delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
            _sparse_attn_bwd_preprocess_v2[grid](
                out_fp32, do, delta,
                T, H, D,
                out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2), out_fp32.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                BLOCK_D=BLOCK_D,
            )

            # Step 2: dQ
            dq = torch.empty_like(q, dtype=torch.float32)
            _sparse_attn_bwd_dq_kernel_v2[grid](
                q, k, v, do,
                indices, mask,
                lse, delta,
                dq,
                T, T_kv, H, D, k_sel,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                scale,
                BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                num_warps=num_warps_bwd,
                num_stages=num_stages_bwd,
            )

            # Step 3: dK/dV
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            _sparse_attn_bwd_dkdv_kernel_v2[grid](
                q, k, v, do,
                indices, mask,
                lse, delta,
                dk, dv,
                T, T_kv, H, D, k_sel,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                scale,
                BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                num_warps=num_warps_bwd,
                num_stages=num_stages_bwd,
            )

            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


# ═══════════════════════════════════════════════════════════════════════
# Public API — V2
# ═══════════════════════════════════════════════════════════════════════

USE_TRITON_BACKWARD = True

def triton_sparse_attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    use_triton_backward: bool = None,
) -> torch.Tensor:
    """V2 sparse attention with tl.dot Tensor Core optimization."""
    if not HAS_TRITON:
        raise ImportError("Triton is required")

    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)

    # Sanitization (from Rohan's V1)
    T_kv = k.shape[1]
    bool_mask = mask > 0.5
    bad = ((indices < 0) | (indices >= T_kv)) & bool_mask
    if bad.any():
        raise ValueError("Critical Error: GSA generated masked-in sparse indices that are out of bounds!")
    safe_indices = torch.where(bool_mask, indices.clamp(0, T_kv - 1), torch.zeros_like(indices))

    _use_triton = use_triton_backward if use_triton_backward is not None else USE_TRITON_BACKWARD

    if _use_triton:
        return TritonSparseAttnFnV2.apply(q, k, v, safe_indices, mask, scale)
    else:
        from triton_sparse_attn import pytorch_sparse_attention
        return pytorch_sparse_attention(q, k, v, safe_indices, mask, scale)


def triton_sparse_attention_v2_fwd_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """V2 with Triton backward always enabled (for benchmarking)."""
    return triton_sparse_attention_v2(q, k, v, indices, mask, scale, use_triton_backward=True)
