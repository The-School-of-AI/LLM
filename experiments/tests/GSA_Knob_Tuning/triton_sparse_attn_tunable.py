"""
Triton Sparse Attention Kernel
==============================

Computes attention only over selected token indices (from the GSA indexer),
achieving O(L*k) complexity instead of O(L^2).

Each program instance handles one query row for one (batch, head) pair.
Online softmax is used to accumulate the output in a single pass over
the k_selected keys, keeping register pressure low.

Numerical stability (bf16 training, AdamW eps 1e-10, no weight decay):
- Forward: Q/K/V upcast to fp32 for scores and online softmax; accumulators
  and output in fp32. LSE uses eps=1e-10 to avoid log(0); all-masked rows
  store LSE sentinel -1e4 so backward never sees -inf.
- Backward: exp(scores - LSE) clamped to exp(50) to avoid inf when LSE is
  the sentinel. Division uses max(l_i, 1e-10) for normalisation.

Includes:
- Triton JIT forward kernel with online softmax
- Triton JIT backward kernels (dQ, dK/dV) with FlashAttention-style recomputation
- torch.autograd.Function wrapper for end-to-end differentiability
- PyTorch chunked fallback (for testing / debugging / gradient reference)
"""

import torch
import torch.nn.functional as F
from typing import Optional

import triton
import triton.language as tl

# ==============================================================================
# Global Tuning Knobs (Used for Autotuning script)
# ==============================================================================
KNOBS = {
    "fwd_BLOCK_Q": 2,
    "fwd_num_warps": 4,
    "fwd_num_stages": 2,
    
    "bwd_dq_BLOCK_K": 64,
    "bwd_dq_num_warps": 4,
    "bwd_dq_num_stages": 2,
    
    "bwd_dkdv_BLOCK_K": 32,
    "bwd_dkdv_num_warps": 4,
    "bwd_dkdv_num_stages": 2,
}
HAS_TRITON = True


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel (unchanged)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_fwd_kernel(
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
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Sparse attention forward kernel with BLOCK_Q tiling.
        Since each query has a unique set of k_sel keys, the math executes as a
        batched vector-matrix operation. Grouping queries into blocks of size BLOCK_Q
        improves memory bandwidth utilization and instruction throughput relative to BLOCK_Q=1.
        
        Grid: (batch_size * n_heads, triton.cdiv(seq_q, BLOCK_Q))
        """
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh  % n_heads

        q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)

        q_mask = q_offs < seq_q

        # load query matrix: [BLOCK_Q, BLOCK_D] (fp32 for numerical stability)
        q_ptrs = (Q_ptr
                  + pid_b * stride_qb
                  + q_offs[:, None] * stride_qq
                  + pid_h * stride_qh
                  + d_offs[None, :] * stride_qd)
        q_i = tl.load(q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < d_head), other=0.0).to(tl.float32)

        # online softmax accumulators: [BLOCK_Q]
        m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
        l_i = tl.full((BLOCK_Q,), 0.0,           dtype=tl.float32)
        # accumulator: [BLOCK_Q, BLOCK_D]
        acc = tl.zeros((BLOCK_Q, BLOCK_D),       dtype=tl.float32)
        EPS = 1e-10

        # bases
        idx_base = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih
        mask_base = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            # Load indices and mask for the entire BLOCK_Q: [BLOCK_Q, BLOCK_K]
            idx_ptrs = idx_base + q_offs[:, None] * stride_iq + k_block_offs[None, :] * stride_ik
            mask_ptrs = mask_base + q_offs[:, None] * stride_mq + k_block_offs[None, :] * stride_mk
            
            # Mask loading with bounds protection
            q_k_mask = q_mask[:, None] & idx_load_mask[None, :]
            qi_indices = tl.load(idx_ptrs, mask=q_k_mask, other=0)
            qi_mask_val = tl.load(mask_ptrs, mask=q_k_mask, other=0.0)
            
            # Final token validity mask
            qi_mask = (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            # gather K, V via indirect load: [BLOCK_Q, BLOCK_K, BLOCK_D]
            k_ptrs = k_base + qi_indices[:, :, None] * stride_kk + d_offs[None, None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, :, None] * stride_vk + d_offs[None, None, :] * stride_vd
            
            kv_load_mask = qi_mask[:, :, None] & (d_offs[None, None, :] < d_head) & q_mask[:, None, None]
            
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)

            # Element-wise math since each query in BLOCK_Q has a uniquely gathered k_sel buffer
            # [BLOCK_Q, BLOCK_D] * [BLOCK_Q, BLOCK_K, BLOCK_D] -> sum over D -> [BLOCK_Q, BLOCK_K]
            scores = tl.sum(q_i[:, None, :] * k_vals, axis=2) * scale
            
            valid = q_mask[:, None] & idx_load_mask[None, :] & qi_mask
            scores = tl.where(valid, scores, float('-inf'))

            # online softmax update per query in the block
            block_max = tl.max(scores, axis=1)  # [BLOCK_Q]
            m_new = tl.maximum(m_i, block_max)  # [BLOCK_Q]
            
            alpha = tl.where(m_new == float('-inf'), 0.0, tl.exp(m_i - m_new))  # [BLOCK_Q]
            # Broadcast the [BLOCK_Q] condition to [BLOCK_Q, 1] to match the [BLOCK_Q, BLOCK_K] score tensor
            is_inf_mask = (m_new == float('-inf'))[:, None]
            beta  = tl.where(is_inf_mask, 0.0, tl.exp(scores - m_new[:, None])) # [BLOCK_Q, BLOCK_K]

            l_i = alpha * l_i + tl.sum(beta, axis=1) # [BLOCK_Q]
            # [BLOCK_Q, BLOCK_D] + sum_over_K( [BLOCK_Q, BLOCK_K, 1] * [BLOCK_Q, BLOCK_K, BLOCK_D] )
            acc = alpha[:, None] * acc + tl.sum(beta[:, :, None] * v_vals, axis=1) # [BLOCK_Q, BLOCK_D]
            m_i = m_new

        # normalise: [BLOCK_Q, BLOCK_D]
        l_i_safe = tl.where(l_i == 0.0, 1.0, tl.maximum(l_i, EPS))
        acc = acc / l_i_safe[:, None]

        # store output
        out_row_ptr = (OUT_ptr
                       + pid_b * stride_ob
                       + q_offs[:, None] * stride_oq
                       + pid_h * stride_oh)
        
        out_mask = q_mask[:, None] & (d_offs[None, :] < d_head)
        tl.store(out_row_ptr + d_offs[None, :] * stride_od, acc, mask=out_mask)

        # store LSE: [BLOCK_Q]
        lse_val = tl.where(l_i > 0.0, m_i + tl.log(tl.maximum(l_i, EPS)), -1e4)
        lse_ptr = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + q_offs
        tl.store(lse_ptr, lse_val, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════════
# Backward kernels
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_bwd_preprocess(
        O_ptr, DO_ptr, DELTA_ptr,
        seq_len, n_heads, d_head,
        stride_ob, stride_oq, stride_oh, stride_od,
        stride_dob, stride_doq, stride_doh, stride_dod,
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

        o_vals = tl.load(o_base + d_offs * stride_od, mask=d_mask, other=0.0).to(tl.float32)
        do_vals = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        delta = tl.sum(o_vals * do_vals, axis=0)

        # delta layout: [B, H, T]  (same as LSE)
        delta_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        tl.store(DELTA_ptr + delta_offset, delta)

    @triton.jit
    def _sparse_attn_bwd_dq_kernel(
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
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        # ── Load scalar LSE[q] and delta[q] ────────────────────────
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)
        
        # Mask out sentinel rows (entirely masked queries) instead of illegal Python `if` returns
        row_active = lse_i > -1e3
        
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

            qi_indices = tl.load(idx_row + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)
            valid = valid & row_active

            # Gather K[ki] and V[ki]
            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # Recompute scores → attention weights (fp32)
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale   # [BLOCK_K]
            scores = tl.where(valid, scores, float('-inf'))
            # Clamp exponent to avoid inf when LSE is sentinel (-1e4) for all-masked rows
            p_i = tl.exp(tl.minimum(scores - lse_i, 50.0))
            p_i = tl.where(valid, p_i, 0.0)

            # dO · V[ki]  per selected key
            do_v = tl.sum(do_i[None, :] * v_vals, axis=1)             # [BLOCK_K]

            # dS = P * (dO·V − δ)
            ds_i = p_i * (do_v - delta_i)                              # [BLOCK_K]

            # dQ += scale * Σ_i  dS[i] * K[ki]
            dq_acc += tl.sum(ds_i[:, None] * k_vals, axis=0) * scale

        # ── Store dQ ───────────────────────────────────────────────
        dq_base = DQ_ptr + pid_b * stride_dqb + pid_q * stride_dqq + pid_h * stride_dqh
        tl.store(dq_base + d_offs * stride_dqd, dq_acc, mask=d_mask)

    @triton.jit
    def _sparse_attn_bwd_dkdv_kernel(
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
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(tl.float32)

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        # ── Load scalar LSE[q] and delta[q] ────────────────────────
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        lse_i = tl.load(LSE_ptr + ld_offset)

        # Mask out sentinel rows instead of illegal Python JIT returning 
        row_active = lse_i > -1e3

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

            qi_indices = tl.load(idx_row + k_block_offs * stride_ik,
                                 mask=idx_load_mask, other=0)
            qi_mask_val = tl.load(mask_row + k_block_offs * stride_mk,
                                  mask=idx_load_mask, other=0.0)
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv)
            valid = valid & row_active

            # Gather K[ki] and V[ki]
            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_mask = valid[:, None] & d_mask[None, :]

            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            # Recompute scores → attention weights (fp32)
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            scores = tl.where(valid, scores, float('-inf'))
            # Clamp exponent to avoid inf when LSE is sentinel (-1e4) for all-masked rows
            p_i = tl.exp(tl.minimum(scores - lse_i, 50.0))
            p_i = tl.where(valid, p_i, 0.0)

            # dO · V[ki]
            do_v = tl.sum(do_i[None, :] * v_vals, axis=1)

            # dS = P * (dO·V − δ)
            ds_i = p_i * (do_v - delta_i)

            # ── Atomic scatter dK[ki] += scale * dS * Q[q] ────────
            dk_contrib = (ds_i[:, None] * q_i[None, :]) * scale       # [BLOCK_K, BLOCK_D]
            dk_ptrs = dk_base + qi_indices[:, None] * stride_dkk + d_offs[None, :] * stride_dkd
            scatter_mask = valid[:, None] & d_mask[None, :]
            tl.atomic_add(dk_ptrs, dk_contrib, mask=scatter_mask)

            # ── Atomic scatter dV[ki] += P * dO[q] ────────────────
            dv_contrib = p_i[:, None] * do_i[None, :]                 # [BLOCK_K, BLOCK_D]
            dv_ptrs = dv_base + qi_indices[:, None] * stride_dvk + d_offs[None, :] * stride_dvd
            tl.atomic_add(dv_ptrs, dv_contrib, mask=scatter_mask)


# ==============================================================================
# 5. Backward dK, dV Scatter Add (Custom Triton Kernel to solve PyTorch CPU Overhead)
# ==============================================================================
if HAS_TRITON:
    @triton.jit
    def _chunked_scatter_add_kernel(
        DK_C_ptr, DV_C_ptr, IDX_C_ptr,
        DK_ptr, DV_ptr,
        n_heads, Cq, k_sel, T_kv, D,
        stride_c_b, stride_c_h, stride_c_cq, stride_c_k, stride_c_d,
        stride_i_b, stride_i_h, stride_i_cq, stride_i_k,
        stride_out_b, stride_out_h, stride_out_t, stride_out_d,
        BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        """
        Takes the densely computed [B, H, Cq, k_sel, D] blocks and scatters them back
        into the full [B, H, T_kv, D] gradient arrays in one launch per chunk.
        """
        pid_bh = tl.program_id(0)
        pid_cq = tl.program_id(1)
        
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads
        
        k_offs = tl.arange(0, BLOCK_K)
        d_offs = tl.arange(0, BLOCK_D)
        
        d_mask = d_offs < D
        
        # Base pointers for this B, H, Cq chunk
        idx_base = IDX_C_ptr + pid_b * stride_i_b + pid_h * stride_i_h + pid_cq * stride_i_cq
        dk_c_base = DK_C_ptr + pid_b * stride_c_b + pid_h * stride_c_h + pid_cq * stride_c_cq
        dv_c_base = DV_C_ptr + pid_b * stride_c_b + pid_h * stride_c_h + pid_cq * stride_c_cq
        
        # Target bases
        out_base_dk = DK_ptr + pid_b * stride_out_b + pid_h * stride_out_h
        out_base_dv = DV_ptr + pid_b * stride_out_b + pid_h * stride_out_h
        
        for k_block in range(0, k_sel, BLOCK_K):
            k_idxs = k_block + k_offs
            k_mask = k_idxs < k_sel
            
            # Load the target T_kv index for this k_sel
            target_t_kv = tl.load(idx_base + k_idxs * stride_i_k, mask=k_mask, other=0)
            
            # Boundary mapping to avoid silent OOB memory corruption
            valid_t = (target_t_kv >= 0) & (target_t_kv < T_kv)
            target_t_kv = tl.where(valid_t, target_t_kv, 0)
            
            # Load the gradient values to scatter: [BLOCK_K, BLOCK_D]
            k_ptrs_c = dk_c_base + k_idxs[:, None] * stride_c_k + d_offs[None, :] * stride_c_d
            v_ptrs_c = dv_c_base + k_idxs[:, None] * stride_c_k + d_offs[None, :] * stride_c_d
            
            load_mask = k_mask[:, None] & d_mask[None, :] & valid_t[:, None]
            
            dk_vals = tl.load(k_ptrs_c, mask=load_mask, other=0.0)
            dv_vals = tl.load(v_ptrs_c, mask=load_mask, other=0.0)
            
            # Calculate destination pointers
            out_ptrs_dk = out_base_dk + target_t_kv[:, None] * stride_out_t + d_offs[None, :] * stride_out_d
            out_ptrs_dv = out_base_dv + target_t_kv[:, None] * stride_out_t + d_offs[None, :] * stride_out_d
            
            # Scatter add into main DK / DV arrays.
            # Note: Since different queries in the same chunk might select the same key,
            # we MUST use atomic_add here. However, this relies on L2 cache and only executes 1 launch per chunk.
            tl.atomic_add(out_ptrs_dk, dk_vals, mask=load_mask)
            tl.atomic_add(out_ptrs_dv, dv_vals, mask=load_mask)


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
            T_kv = k.shape[1]
            k_sel = indices.size(-1)

            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if mask.dtype != torch.float32:
                mask = mask.to(torch.float32)

            out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
            lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

            BLOCK_Q = KNOBS["fwd_BLOCK_Q"]
            BLOCK_K = triton.next_power_of_2(min(128, k_sel))  # FIX-PERF-03a: Raised from min(64) for A100 memory throughput
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * H, triton.cdiv(T, BLOCK_Q))

            _sparse_attn_fwd_kernel[grid](
                q, k, v, indices, mask,
                out, lse,
                B, T, T_kv, H, D, k_sel,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                scale,
                BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                num_warps=KNOBS["fwd_num_warps"], num_stages=KNOBS["fwd_num_stages"]
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

            # ── Step 1: delta[b,h,q] = sum_d(O * dO) ──────────────────
            delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
            _sparse_attn_bwd_preprocess[grid](
                out_fp32, do, delta,
                T, H, D,
                out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2), out_fp32.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                BLOCK_D=BLOCK_D,
            )

            # ── Step 2: dQ (local accumulate — fast, no atomics) ──────
            dq = torch.empty_like(q, dtype=torch.float32)
            _sparse_attn_bwd_dq_kernel[grid](
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
                BLOCK_K=KNOBS["bwd_dq_BLOCK_K"], BLOCK_D=BLOCK_D,
                num_warps=KNOBS["bwd_dq_num_warps"], num_stages=KNOBS["bwd_dq_num_stages"]
            )

            # ── Step 3: dK/dV ──
            # Fully Triton backward for dK/dV to eliminate PyTorch materialization HBM overhead
            dk = torch.zeros(B, T_kv, H, D, device=q.device, dtype=torch.float32)
            dv = torch.zeros(B, T_kv, H, D, device=q.device, dtype=torch.float32)

            FAST_DKDV_TRITON = True
            
            if FAST_DKDV_TRITON:
                # Optimized for k_sel=64, D=128 on A100.
                BLOCK_K_DKDV = KNOBS["bwd_dkdv_BLOCK_K"]
                
                _sparse_attn_bwd_dkdv_kernel[grid](
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
                    BLOCK_K=BLOCK_K_DKDV, BLOCK_D=BLOCK_D,
                    num_warps=KNOBS["bwd_dkdv_num_warps"], num_stages=KNOBS["bwd_dkdv_num_stages"]
                )
            else:
                # Fallback: Previous Python chunked orchestration
                q_bh  = q.permute(0, 2, 1, 3).contiguous().float()   # [B, H, T,   D]
                k_bh  = k.permute(0, 2, 1, 3).contiguous().float()   # [B, H, T_kv,D]
                v_bh  = v.permute(0, 2, 1, 3).contiguous().float()   # [B, H, T_kv,D]
                do_bh = do.permute(0, 2, 1, 3).contiguous()          # [B, H, T,   D]

                dk_bh = torch.zeros(B, H, T_kv, D, device=q.device, dtype=torch.float32)
                dv_bh = torch.zeros(B, H, T_kv, D, device=q.device, dtype=torch.float32)

                FAST_4K_MODE = True
                if FAST_4K_MODE:
                    C = T  # Process in a single massive chunk
                else:
                    C = 64

                for q_start in range(0, T, C):
                    q_end = min(q_start + C, T)
                    Cq = q_end - q_start   # actual chunk size

                    idx_c_raw  = indices[:, :, q_start:q_end, :]   # [B, H, Cq, k_sel]
                    msk_c_raw  = mask[:, :, q_start:q_end, :]      # [B, H, Cq, k_sel]
                    
                    DO_SORT = False
                    if DO_SORT:
                        idx_c, sort_idx = torch.sort(idx_c_raw, dim=-1)
                        msk_c = torch.gather(msk_c_raw, dim=-1, index=sort_idx)
                    else:
                        idx_c = idx_c_raw
                        msk_c = msk_c_raw

                    q_c    = q_bh[:, :, q_start:q_end, :]      # [B, H, Cq, D]
                    do_c   = do_bh[:, :, q_start:q_end, :]     # [B, H, Cq, D]
                    lse_c  = lse[:, :, q_start:q_end]          # [B, H, Cq]
                    delta_c = delta[:, :, q_start:q_end]       # [B, H, Cq]

                    idx_e = idx_c.unsqueeze(-1).expand(B, H, Cq, k_sel, D)
                    k_sel_c = torch.gather(k_bh, dim=2, index=idx_e)
                    v_sel_c = torch.gather(v_bh, dim=2, index=idx_e)

                    S_c = torch.matmul(q_c.unsqueeze(-2), k_sel_c.transpose(-1, -2)).squeeze(-2) * scale
                    P_c = torch.exp((S_c - lse_c.unsqueeze(-1)).clamp(max=50.0))
                    P_c = P_c * (msk_c > 0.5).float()

                    do_v_c = torch.matmul(do_c.unsqueeze(-2), v_sel_c.transpose(-1, -2)).squeeze(-2)

                    dS_c = P_c * (do_v_c - delta_c.unsqueeze(-1))           # [B, H, Cq, k_sel]

                    dk_c = (dS_c.unsqueeze(-1) * q_c.unsqueeze(-2)) * scale
                    dv_c = (P_c.unsqueeze(-1) * do_c.unsqueeze(-2))

                    dk_c = dk_c.contiguous()
                    dv_c = dv_c.contiguous()
                    idx_c = idx_c.contiguous()

                    scatter_grid = (B * H, Cq)
                    BLOCK_K_SCATTER = triton.next_power_of_2(min(128, k_sel))
                    
                    _chunked_scatter_add_kernel[scatter_grid](
                        dk_c, dv_c, idx_c,
                        dk_bh, dv_bh,
                        H, Cq, k_sel, T_kv, D,
                        dk_c.stride(0), dk_c.stride(1), dk_c.stride(2), dk_c.stride(3), dk_c.stride(4),
                        idx_c.stride(0), idx_c.stride(1), idx_c.stride(2), idx_c.stride(3),
                        dk_bh.stride(0), dk_bh.stride(1), dk_bh.stride(2), dk_bh.stride(3),
                        BLOCK_K=BLOCK_K_SCATTER, BLOCK_D=BLOCK_D,
                        num_warps=4, num_stages=2
                    )

                    del k_sel_c, v_sel_c, S_c, P_c, do_v_c, dS_c, dk_c, dv_c

                dk = dk_bh.permute(0, 2, 1, 3).contiguous()
                dv = dv_bh.permute(0, 2, 1, 3).contiguous()

            dk = dk.to(k.dtype)
            dv = dv.to(v.dtype)

            return dq.to(q.dtype), dk, dv, None, None, None

# ═══════════════════════════════════════════════════════════════════════
# ==============================================================================
# Public API
# ==============================================================================

# Module-level toggle: no longer used. Triton is mathematically required.
USE_TRITON_BACKWARD = True

def triton_sparse_attention(
    q: torch.Tensor,        # [B, T, H, D]
    k: torch.Tensor,        # [B, T, H, D]
    v: torch.Tensor,        # [B, T, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,     # [B, H, T, k_sel]
    scale: float,
    use_triton_backward: bool = None,
) -> torch.Tensor:
    """
    Triton sparse attention — fully differentiable via custom backward.
    """
    if not HAS_TRITON:
        raise ImportError("CRITICAL ERROR: Triton is absolutely required for triton_sparse_attention. PyTorch fallback has been disabled to prevent out-of-memory crashes on D-loop scatters.")

    # Indices must be int64 for Triton pointer arithmetic
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    # Mask must be float32 for comparison
    if mask.dtype != torch.float32:
        mask = mask.to(torch.float32)

    # Claude's Sanitization: Assert validity of Masked-In tokens, Clamp Masked-Out tokens
    T_kv = k.shape[1]
    bool_mask = mask > 0.5
    bad = ((indices < 0) | (indices >= T_kv)) & bool_mask
    
    if bad.any():
        raise ValueError("Critical Error: GSA generated masked-in sparse indices that are out of bounds!")
        
    safe_indices = torch.where(bool_mask, indices.clamp(0, T_kv - 1), torch.zeros_like(indices))

    # Guardrails explicitly requested by AI Code Review (Systems & Fragmentation safety)
    if not safe_indices.is_contiguous():
        safe_indices = safe_indices.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()

    out = TritonSparseAttnFn.apply(q, k, v, safe_indices, mask, scale)

    # Tripwire: Catch any NaN corruptions out of the forward pass immediately
    if out.requires_grad:
        sample = out.flatten()[::max(1, out.numel() // 1024)]
        if not torch.isfinite(sample).all():
            raise RuntimeError("Triton Sparse Attention Forward pass produced a NaN/Inf output early in the sequence!")

    return out
