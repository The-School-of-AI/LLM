"""
Triton Sparse Attention Kernel
==============================

Computes attention only over selected token indices (from the GSA indexer),
achieving O(L*k) complexity instead of O(L^2).

Each program instance handles one query row for one (batch, head) pair.
Online softmax is used to accumulate the output in a single pass over
the k_selected keys, keeping register pressure low.

Includes:
- Triton JIT forward kernel with online softmax
- Triton JIT backward kernels (dQ, dK/dV) with FlashAttention-style recomputation
- torch.autograd.Function wrapper for end-to-end differentiability
- PyTorch chunked fallback (for testing / debugging / gradient reference)
"""

import torch
import torch.nn.functional as F

# Check for Triton availability
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel (unchanged)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _sparse_attn_fwd_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        IDX_ptr,
        MASK_ptr,
        OUT_ptr,
        LSE_ptr,
        batch_size,
        seq_q,
        seq_kv,
        n_heads,
        d_head,
        k_selected,
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vh,
        stride_vd,
        stride_ib,
        stride_ih,
        stride_iq,
        stride_ik,
        stride_mb,
        stride_mh,
        stride_mq,
        stride_mk,
        stride_ob,
        stride_oq,
        stride_oh,
        stride_od,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Sparse attention forward kernel.  One program computes one query row
        for one (batch, head) pair — no BLOCK_Q tiling needed, which keeps
        register pressure low and avoids the inner qi loop.

        Works directly on (B, T, H, D) tensors via stride-based access —
        no contiguous copies required in the wrapper.

        Grid: (batch_size * n_heads, seq_q)
        """
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        # load query vector
        q_row_ptr = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_row_ptr + d_offs * stride_qd, mask=d_offs < d_head, other=0.0)

        # online softmax accumulators
        m_i = tl.full((1,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((1,), 0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # indices/mask: (B, H, T, k_sel) — access with pid_b + pid_h
        idx_row_ptr = (
            IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        )
        mask_row_ptr = (
            MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq
        )
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs

            idx_load_mask = k_block_offs < k_selected
            qi_indices = tl.load(
                idx_row_ptr + k_block_offs * stride_ik, mask=idx_load_mask, other=0
            )
            qi_mask_val = tl.load(
                mask_row_ptr + k_block_offs * stride_mk, mask=idx_load_mask, other=0.0
            )
            qi_mask = (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            # gather K, V via indirect load
            k_ptrs = (
                k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            )
            v_ptrs = (
                v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            )
            kv_load_mask = qi_mask[:, None] & (d_offs[None, :] < d_head)
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

            # dot-product scores
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            valid = idx_load_mask & qi_mask
            scores = tl.where(valid, scores, float("-inf"))

            # online softmax update
            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)

            # Prevent NaN when m_new is -inf
            alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
            beta = tl.where(m_new == float("-inf"), 0.0, tl.exp(scores - m_new))

            l_i = alpha * l_i + tl.sum(beta, axis=0)
            acc = alpha * acc + tl.sum(beta[:, None] * v_vals, axis=0)
            m_i = m_new

        # normalise
        l_i_safe = tl.where(l_i == 0.0, 1.0, l_i)
        acc = acc / l_i_safe

        # store output
        out_row_ptr = (
            OUT_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        )
        tl.store(out_row_ptr + d_offs * stride_od, acc, mask=d_offs < d_head)

        # store LSE  [batch, n_heads, seq_q]
        lse_ptr = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + pid_q
        tl.store(lse_ptr + tl.arange(0, 1), m_i + tl.log(l_i))


# ═══════════════════════════════════════════════════════════════════════
# Backward kernels
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    @triton.jit
    def _sparse_attn_bwd_preprocess(
        O_ptr,
        DO_ptr,
        DELTA_ptr,
        seq_len,
        n_heads,
        d_head,
        stride_ob,
        stride_oq,
        stride_oh,
        stride_od,
        stride_dob,
        stride_doq,
        stride_doh,
        stride_dod,
        BLOCK_D: tl.constexpr,
    ):
        """
        Compute delta[b,h,q] = sum_d( O[b,q,h,d] * dO[b,q,h,d] ).
        Grid: (B*H, T)
        """
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        o_base = O_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh

        o_vals = tl.load(o_base + d_offs * stride_od, mask=d_mask, other=0.0).to(
            tl.float32
        )
        do_vals = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(
            tl.float32
        )

        delta = tl.sum(o_vals * do_vals, axis=0)

        # delta layout: [B, H, T]  (same as LSE)
        delta_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        tl.store(DELTA_ptr + delta_offset, delta)

    @triton.jit
    def _sparse_attn_bwd_dq_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        IDX_ptr,
        MASK_ptr,
        LSE_ptr,
        DELTA_ptr,
        DQ_ptr,
        seq_len,
        seq_kv,
        n_heads,
        d_head,
        k_selected,
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vh,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_doh,
        stride_dod,
        stride_ib,
        stride_ih,
        stride_iq,
        stride_ik,
        stride_mb,
        stride_mh,
        stride_mq,
        stride_mk,
        stride_dqb,
        stride_dqq,
        stride_dqh,
        stride_dqd,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Compute dQ for each query — local accumulate, no atomics.
        Grid: (B*H, T)

        Math:
          dS[q,ki] = P[q,ki] * ( dO[q]·V[ki] − δ[q] )
          dQ[q]    = scale * Σ_i  dS[q,ki] * K[ki]

        P is recomputed from saved LSE:  P[q,ki] = exp(S[q,ki] − LSE[q])
        """
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        d_mask = d_offs < d_head

        # ── Load Q[q] and dO[q] ────────────────────────────────────
        q_base = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(
            tl.float32
        )

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(
            tl.float32
        )

        # ── Load scalar LSE[q] and delta[q] ────────────────────────
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)
        delta_i = tl.load(DELTA_ptr + ld_offset)

        # ── Row pointers for indices / mask ────────────────────────
        idx_row = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq

        # ── K, V base pointers ─────────────────────────────────────
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        # ── Accumulate dQ ──────────────────────────────────────────
        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            qi_indices = tl.load(
                idx_row + k_block_offs * stride_ik, mask=idx_load_mask, other=0
            )
            qi_mask_val = tl.load(
                mask_row + k_block_offs * stride_mk, mask=idx_load_mask, other=0.0
            )
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            # Gather K[ki] and V[ki]
            k_ptrs = (
                k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            )
            v_ptrs = (
                v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            )
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # Recompute scores → attention weights
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale  # [BLOCK_K]
            scores = tl.where(valid, scores, float("-inf"))
            p_i = tl.exp(scores - lse_i)  # [BLOCK_K]
            p_i = tl.where(valid, p_i, 0.0)

            # dO · V[ki]  per selected key
            do_v = tl.sum(do_i[None, :] * v_vals, axis=1)  # [BLOCK_K]

            # dS = P * (dO·V − δ)
            ds_i = p_i * (do_v - delta_i)  # [BLOCK_K]

            # dQ += scale * Σ_i  dS[i] * K[ki]
            dq_acc += tl.sum(ds_i[:, None] * k_vals, axis=0) * scale

        # ── Store dQ ───────────────────────────────────────────────
        dq_base = DQ_ptr + pid_b * stride_dqb + pid_q * stride_dqq + pid_h * stride_dqh
        tl.store(dq_base + d_offs * stride_dqd, dq_acc, mask=d_mask)

    @triton.jit
    def _sparse_attn_bwd_dkdv_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        IDX_ptr,
        MASK_ptr,
        LSE_ptr,
        DELTA_ptr,
        DK_ptr,
        DV_ptr,
        seq_len,
        seq_kv,
        n_heads,
        d_head,
        k_selected,
        stride_qb,
        stride_qq,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kh,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vh,
        stride_vd,
        stride_dob,
        stride_doq,
        stride_doh,
        stride_dod,
        stride_ib,
        stride_ih,
        stride_iq,
        stride_ik,
        stride_mb,
        stride_mh,
        stride_mq,
        stride_mk,
        stride_dkb,
        stride_dkk,
        stride_dkh,
        stride_dkd,
        stride_dvb,
        stride_dvk,
        stride_dvh,
        stride_dvd,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Compute dK/dV via atomic scatter.
        Grid: (B*H, T)

        For each query q, iterates over its k_sel selected keys and
        atomically accumulates:
          dK[ki] += scale * dS[q,ki] * Q[q]
          dV[ki] += P[q,ki]  * dO[q]
        """
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        d_mask = d_offs < d_head

        # ── Load Q[q] and dO[q] ────────────────────────────────────
        q_base = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(
            tl.float32
        )

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(
            tl.float32
        )

        # ── Load scalar LSE[q] and delta[q] ────────────────────────
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)
        delta_i = tl.load(DELTA_ptr + ld_offset)

        # ── Row pointers for indices / mask ────────────────────────
        idx_row = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq

        # ── Base pointers ──────────────────────────────────────────
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        dk_base = DK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
        dv_base = DV_ptr + pid_b * stride_dvb + pid_h * stride_dvh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            qi_indices = tl.load(
                idx_row + k_block_offs * stride_ik, mask=idx_load_mask, other=0
            )
            qi_mask_val = tl.load(
                mask_row + k_block_offs * stride_mk, mask=idx_load_mask, other=0.0
            )
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            # Gather K[ki] and V[ki]
            k_ptrs = (
                k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            )
            v_ptrs = (
                v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            )
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # Recompute scores → attention weights
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            scores = tl.where(valid, scores, float("-inf"))
            p_i = tl.exp(scores - lse_i)
            p_i = tl.where(valid, p_i, 0.0)

            # dO · V[ki]
            do_v = tl.sum(do_i[None, :] * v_vals, axis=1)

            # dS = P * (dO·V − δ)
            ds_i = p_i * (do_v - delta_i)

            # ── Atomic scatter dK[ki] += scale * dS * Q[q] ────────
            dk_contrib = (ds_i[:, None] * q_i[None, :]) * scale  # [BLOCK_K, BLOCK_D]
            dk_ptrs = (
                dk_base
                + qi_indices[:, None] * stride_dkk
                + d_offs[None, :] * stride_dkd
            )
            scatter_mask = valid[:, None] & d_mask[None, :]
            tl.atomic_add(dk_ptrs, dk_contrib, mask=scatter_mask)

            # ── Atomic scatter dV[ki] += P * dO[q] ────────────────
            dv_contrib = p_i[:, None] * do_i[None, :]  # [BLOCK_K, BLOCK_D]
            dv_ptrs = (
                dv_base
                + qi_indices[:, None] * stride_dvk
                + d_offs[None, :] * stride_dvd
            )
            tl.atomic_add(dv_ptrs, dv_contrib, mask=scatter_mask)


# ═══════════════════════════════════════════════════════════════════════
# torch.autograd.Function wrapper
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:

    class TritonSparseAttnFn(torch.autograd.Function):
        """
        Fused sparse attention with Triton forward + backward.

        Forward:   online softmax over k_sel gathered keys  →  O, LSE
        Backward:  FlashAttention-style recompute of P from LSE  →  dQ, dK, dV

        Numerically equivalent to pytorch_sparse_attention below.
        """

        @staticmethod
        def forward(ctx, q, k, v, indices, mask, scale):
            """
            Args:
                q, k, v:  [B, T, H, D]  (any dtype, computed in fp32)
                indices:  [B, H, T, k_sel]  int64
                mask:     [B, H, T, k_sel]  float32
                scale:    float
            Returns:
                out:      [B, T, H, D]  same dtype as q
            """
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

            _sparse_attn_fwd_kernel[grid](
                q,
                k,
                v,
                indices,
                mask,
                out,
                lse,
                B,
                T,
                T,
                H,
                D,
                k_sel,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                indices.stride(3),
                mask.stride(0),
                mask.stride(1),
                mask.stride(2),
                mask.stride(3),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out.stride(3),
                scale,
                BLOCK_K=BLOCK_K,
                BLOCK_D=BLOCK_D,
            )

            out_typed = out.to(q.dtype)

            # Save for backward — keep fp32 out for delta computation
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

            # Ensure grad_output is contiguous and float32
            do = grad_output.contiguous().to(torch.float32)

            # ── Step 1: delta[b,h,q] = sum_d(O * dO) ──────────────
            delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

            _sparse_attn_bwd_preprocess[grid](
                out_fp32,
                do,
                delta,
                T,
                H,
                D,
                out_fp32.stride(0),
                out_fp32.stride(1),
                out_fp32.stride(2),
                out_fp32.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                BLOCK_D=BLOCK_D,
            )

            # ── Step 2: dQ (local accumulate) ──────────────────────
            dq = torch.empty_like(q, dtype=torch.float32)

            _sparse_attn_bwd_dq_kernel[grid](
                q,
                k,
                v,
                do,
                indices,
                mask,
                lse,
                delta,
                dq,
                T,
                T_kv,
                H,
                D,
                k_sel,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                indices.stride(3),
                mask.stride(0),
                mask.stride(1),
                mask.stride(2),
                mask.stride(3),
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                scale,
                BLOCK_K=BLOCK_K,
                BLOCK_D=BLOCK_D,
            )

            # ── Step 3: dK/dV (atomic scatter) ─────────────────────
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)

            _sparse_attn_bwd_dkdv_kernel[grid](
                q,
                k,
                v,
                do,
                indices,
                mask,
                lse,
                delta,
                dk,
                dv,
                T,
                T_kv,
                H,
                D,
                k_sel,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                k.stride(0),
                k.stride(1),
                k.stride(2),
                k.stride(3),
                v.stride(0),
                v.stride(1),
                v.stride(2),
                v.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                indices.stride(3),
                mask.stride(0),
                mask.stride(1),
                mask.stride(2),
                mask.stride(3),
                dk.stride(0),
                dk.stride(1),
                dk.stride(2),
                dk.stride(3),
                dv.stride(0),
                dv.stride(1),
                dv.stride(2),
                dv.stride(3),
                scale,
                BLOCK_K=BLOCK_K,
                BLOCK_D=BLOCK_D,
            )

            # Cast gradients back to input dtype
            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

# Module-level toggle: set to False to use PyTorch autograd backward
# instead of fused Triton backward (useful for debugging / comparison).
USE_TRITON_BACKWARD = True


def triton_sparse_attention(
    q: torch.Tensor,  # [B, T, H, D]
    k: torch.Tensor,  # [B, T, H, D]
    v: torch.Tensor,  # [B, T, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,  # [B, H, T, k_sel]
    scale: float,
    use_triton_backward: bool = None,
) -> torch.Tensor:
    """
    Triton sparse attention — fully differentiable via custom backward.

    Args:
        use_triton_backward: If True, uses fused Triton backward kernels.
            If False, uses PyTorch autograd backward (slower but proven).
            If None (default), uses the module-level USE_TRITON_BACKWARD flag.
    """
    if not HAS_TRITON:
        raise ImportError("Triton is required for triton_sparse_attention")

    # Indices must be int64 for Triton pointer arithmetic
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    # Mask must be float32 for comparison
    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)

    # Resolve toggle
    _use_triton = (
        use_triton_backward if use_triton_backward is not None else USE_TRITON_BACKWARD
    )

    # Claude's Sanitization: Assert validity of Masked-In tokens, Clamp Masked-Out tokens
    T_kv = k.shape[1]
    bool_mask = mask > 0.5
    bad = ((indices < 0) | (indices >= T_kv)) & bool_mask

    if bad.any():
        raise ValueError(
            "Critical Error: GSA generated masked-in sparse indices that are out of bounds!"
        )

    safe_indices = torch.where(
        bool_mask, indices.clamp(0, T_kv - 1), torch.zeros_like(indices)
    )

    if _use_triton:
        return TritonSparseAttnFn.apply(q, k, v, safe_indices, mask, scale)
    else:
        # PyTorch autograd backward (slower but proven correct)
        return pytorch_sparse_attention(q, k, v, safe_indices, mask, scale)


def pytorch_sparse_attention(
    q: torch.Tensor,  # [B, T, H, D]
    k: torch.Tensor,  # [B, T, H, D]
    v: torch.Tensor,  # [B, T, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,  # [B, H, T, k_sel]
    scale: float,
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    Memory-efficient PyTorch sparse attention fallback.

    Gathers the k_sel keys/values per query using advanced indexing,
    then runs a small chunked softmax — O(T*k) instead of O(T^2).
    Fully differentiable (autograd-friendly).

    Kept as a reference implementation for gradient correctness testing.
    """
    B, T, H, _ = q.shape

    k_bh = k.permute(0, 2, 1, 3)  # (B, H, T_kv, D)
    v_bh = v.permute(0, 2, 1, 3)  # (B, H, T_kv, D)
    q_bh = q.permute(0, 2, 1, 3)  # (B, H, T,    D)

    output = torch.empty_like(q_bh)  # (B, H, T, D)

    bh_idx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
    h_idx = torch.arange(H, device=q.device).view(1, H, 1, 1)

    for i in range(0, T, chunk_size):
        end = min(i + chunk_size, T)

        idx_chunk = indices[:, :, i:end, :]
        mask_chunk = mask[:, :, i:end, :]
        q_chunk = q_bh[:, :, i:end, :]

        # Convert mask to bool for masked_fill (handles both bool and float inputs)
        bool_mask = mask_chunk > 0.5 if mask_chunk.dtype != torch.bool else mask_chunk

        # Clamp indices for PyTorch fallback so it doesn't crash on device-side asserts
        idx_chunk_clamped = torch.clamp(idx_chunk, 0, v_bh.size(2) - 1)

        k_gathered = k_bh[bh_idx, h_idx, idx_chunk_clamped]
        v_gathered = v_bh[bh_idx, h_idx, idx_chunk_clamped]

        scores = torch.einsum("bhqd,bhqkd->bhqk", q_chunk, k_gathered) * scale
        scores = scores.masked_fill(~bool_mask, float("-inf"))

        attn_w = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_w = attn_w.masked_fill(~bool_mask, 0.0)

        out_chunk = torch.einsum("bhqk,bhqkd->bhqd", attn_w, v_gathered)
        output[:, :, i:end, :] = out_chunk

    # (B, H, T, D) -> (B, T, H, D)
    return output.permute(0, 2, 1, 3)
