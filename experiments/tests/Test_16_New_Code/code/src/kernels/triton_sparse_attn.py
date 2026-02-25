"""
Triton Sparse Attention Kernel V2-Light — Atomic-Route Edition (Refined)
========================================================================

Optimized for 4K-16K context lengths.
- 100% Zero Host-Device Synchronization (No .item(), no .sum() allocation)
- O(N) Inverse Indexing (Atomic Routing groups connections without sorting)
- Specialized Forward Kernel (BLOCK_Q=1 for reduced register pressure)
- High-Occupancy dK/dV (Register accumulation with zero-atomics)
"""

import torch
from typing import Optional

import triton
import triton.language as tl

HAS_TRITON = True

# ═══════════════════════════════════════════════════════════════════════
# Forward kernel (BLOCK_Q=1 Specialization)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        IDX_ptr, MASK_ptr,
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
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        # Guard for seq_q bounds
        if pid_q >= seq_q:
            return

        # Specializing for BLOCK_Q=1 allows using 1D tensors for registers
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        
        # Load Q: [BLOCK_D]
        q_row_ptr = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_row_ptr + d_offs * stride_qd, mask=d_offs < d_head, other=0.0).to(tl.float32)

        m_i = float('-inf')
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        EPS = 1e-10

        # Base pointers for K and V
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        # Index and Mask row pointers: [B, H, Q, K_SEL]
        idx_row_ptr_base = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row_ptr_base = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq

        for k_step in range(0, k_selected, BLOCK_K):
            ks = k_step + k_offs
            in_range = ks < k_selected
            
            # Load selected key indices: [BLOCK_K]
            qi_indices = tl.load(idx_row_ptr_base + ks * stride_ik, mask=in_range, other=0).to(tl.int32)
            qi_mask_val = tl.load(mask_row_ptr_base + ks * stride_mk, mask=in_range, other=0.0)
            
            # Combine validity checks explicitly bounding >= 0
            qi_mask = in_range & (qi_mask_val > 0.5) & (qi_indices >= 0) & (qi_indices < seq_kv)

            # Load Keys and Values: [BLOCK_K, BLOCK_D]
            k_p = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_p = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            
            kv_mask = qi_mask[:, None] & (d_offs[None, :] < d_head)
            k_vals = tl.load(k_p, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_p, mask=kv_mask, other=0.0).to(tl.float32)

            # Compute Attention: [BLOCK_K]
            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            scores = tl.where(qi_mask, scores, float('-inf'))

            # Online Softmax update
            block_max = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(m_new == float('-inf'), 0.0, tl.exp(m_i - m_new))
            beta = tl.where(scores == float('-inf'), 0.0, tl.exp(scores - m_new))
            
            l_i = alpha * l_i + tl.sum(beta, axis=0)
            acc = alpha * acc + tl.sum(beta[:, None] * v_vals, axis=0)
            m_i = m_new

        # Finalize Output
        l_i_safe = tl.where(l_i == 0.0, 1.0, tl.maximum(l_i, EPS))
        acc = acc / l_i_safe

        out_ptr = OUT_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        tl.store(out_ptr + d_offs * stride_od, acc, mask=d_offs < d_head)

        lse_val = tl.where(l_i == 0.0, -1e4, m_i + tl.log(l_i_safe))
        lse_ptr_final = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + pid_q
        tl.store(lse_ptr_final, lse_val)

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
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        o_base = O_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        o_i = tl.load(o_base + d_offs * stride_od, mask=d_mask, other=0.0).to(tl.float32)
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(tl.float32)

        delta_i = tl.sum(o_i * do_i)
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        tl.store(DELTA_ptr + ld_offset, delta_i)

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
        row_active = lse_i > -1e3
        delta_i = tl.load(DELTA_ptr + ld_offset)

        idx_row = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh + pid_q * stride_mq
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected
            
            qi_indices = tl.load(idx_row + k_block_offs * stride_ik, mask=idx_load_mask, other=0).to(tl.int32)
            qi_mask_val = tl.load(mask_row + k_block_offs * stride_mk, mask=idx_load_mask, other=0.0)
            valid = idx_load_mask & (qi_mask_val > 0.5) & (qi_indices >= 0) & (qi_indices < seq_kv) & row_active

            k_ptrs = k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            kv_mask = valid[:, None] & d_mask[None, :]
            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            scores = tl.where(valid, scores, float('-inf'))
            p_i = tl.exp(tl.minimum(scores - lse_i, 50.0))
            p_i = tl.where(valid, p_i, 0.0)

            do_v = tl.sum(do_i[None, :] * v_vals, axis=1)
            ds_i = p_i * (do_v - delta_i)
            dq_acc += tl.sum(ds_i[:, None] * k_vals, axis=0) * scale

        dq_base = DQ_ptr + pid_b * stride_dqb + pid_q * stride_dqq + pid_h * stride_dqh
        tl.store(dq_base + d_offs * stride_dqd, dq_acc, mask=d_mask)

    @triton.jit
    def _sparse_attn_bwd_dkdv_keymajor_kernel(
        Q_ptr, K_ptr, V_ptr, DO_ptr,
        LSE_ptr, DELTA_ptr,
        DK_ptr, DV_ptr,
        INV_QUERIES_ptr, INV_COUNT_ptr, INV_OFFSET_ptr,
        seq_len, seq_kv, n_heads, d_head,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_dob, stride_doq, stride_doh, stride_dod,
        stride_dkb, stride_dkk, stride_dkh, stride_dkd,
        stride_dvb, stride_dvk, stride_dvh, stride_dvd,
        scale,
        BLOCK_Q_INNER: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_ki = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        # Load fan-in count and absolute offset for this key position
        fan_in = tl.load(INV_COUNT_ptr + pid_bh * seq_kv + pid_ki).to(tl.int32)
        base_off = tl.load(INV_OFFSET_ptr + pid_bh * seq_kv + pid_ki).to(tl.int32)

        # Load K[b, ki, h, :] and V[b, ki, h, :] once
        k_base = K_ptr + pid_b * stride_kb + pid_ki * stride_kk + pid_h * stride_kh
        k_vec = tl.load(k_base + d_offs * stride_kd, mask=d_mask, other=0.0).to(tl.float32)

        v_base = V_ptr + pid_b * stride_vb + pid_ki * stride_vk + pid_h * stride_vh
        v_vec = tl.load(v_base + d_offs * stride_vd, mask=d_mask, other=0.0).to(tl.float32)

        # Accumulators in registers
        dk_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        inv_ptr_base = INV_QUERIES_ptr + base_off
        q_batch_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_batch_base = DO_ptr + pid_b * stride_dob + pid_h * stride_doh
        lse_base = pid_b * n_heads * seq_len + pid_h * seq_len

        q_inner_offs = tl.arange(0, BLOCK_Q_INNER)

        for q_start in range(0, fan_in, BLOCK_Q_INNER):
            q_block_offs = q_start + q_inner_offs
            q_valid = q_block_offs < fan_in

            # Load which queries selected this key
            q_ids = tl.load(inv_ptr_base + q_block_offs, mask=q_valid, other=0)
            
            q_ptrs = q_batch_base + q_ids[:, None] * stride_qq + d_offs[None, :] * stride_qd
            qd_mask = q_valid[:, None] & d_mask[None, :]
            q_vals = tl.load(q_ptrs, mask=qd_mask, other=0.0).to(tl.float32)

            do_ptrs = do_batch_base + q_ids[:, None] * stride_doq + d_offs[None, :] * stride_dod
            do_vals = tl.load(do_ptrs, mask=qd_mask, other=0.0).to(tl.float32)

            # Load attention stats
            lse_vals = tl.load(LSE_ptr + lse_base + q_ids, mask=q_valid, other=-1e4)
            delta_vals = tl.load(DELTA_ptr + lse_base + q_ids, mask=q_valid, other=0.0)

            active = q_valid & (lse_vals > -1e3)
            
            # Recompute Attention Scores
            scores = tl.sum(q_vals * k_vec[None, :], axis=1) * scale
            scores = tl.where(active, scores, float('-inf'))
            p_i = tl.exp(tl.minimum(scores - lse_vals, 50.0))
            p_i = tl.where(active, p_i, 0.0)

            # dS = P * (dO·V - delta)
            do_v = tl.sum(do_vals * v_vec[None, :], axis=1)
            ds_i = p_i * (do_v - delta_vals)

            dk_acc += tl.sum(ds_i[:, None] * q_vals, axis=0) * scale
            dv_acc += tl.sum(p_i[:, None] * do_vals, axis=0)

        # Single store to global DK/DV
        dk_base = DK_ptr + pid_b * stride_dkb + pid_ki * stride_dkk + pid_h * stride_dkh
        tl.store(dk_base + d_offs * stride_dkd, dk_acc, mask=d_mask)

        dv_base = DV_ptr + pid_b * stride_dvb + pid_ki * stride_dvk + pid_h * stride_dvh
        tl.store(dv_base + d_offs * stride_dvd, dv_acc, mask=d_mask)

# ═══════════════════════════════════════════════════════════════════════
# Inverse Index Builder (Zero-Sync O(N) Atomic Route)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _route_inverse_index_kernel(
        IDX_ptr, MASK_ptr,
        WRITE_OFF_ptr,       # Mutated by atomic_add to yield write indices
        INV_QUERIES_ptr,     # Final flat output array
        total_elements, B, H, T, k_sel, T_kv,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid_thread = offs < total_elements
        
        # Linear offset -> (b, h, t, k_sel_idx)
        k_idx = offs % k_sel
        tmp = offs // k_sel
        t = tmp % T
        tmp = tmp // T
        h = tmp % H
        b = tmp // H
        
        # Load index and mask
        idx_p = IDX_ptr + b * stride_ib + h * stride_ih + t * stride_iq + k_idx * stride_ik
        mask_p = MASK_ptr + b * stride_mb + h * stride_mh + t * stride_mq + k_idx * stride_mk
        
        idx_val = tl.load(idx_p, mask=valid_thread, other=-1).to(tl.int32)
        mask_val = tl.load(mask_p, mask=valid_thread, other=0.0)
        
        # Connection validity check
        is_valid = valid_thread & (mask_val > 0.5) & (idx_val >= 0) & (idx_val < T_kv)
        
        # Zero out invalid indices before math to prevent PTX memory faults
        idx_safe = tl.where(is_valid, idx_val, 0)
        
        # Atomic counter per (bh, key)
        bh_ki = (b * H + h) * T_kv + idx_safe
        write_off_p = WRITE_OFF_ptr + bh_ki
        
        # Atomic add 1 returns the unique write index for query t
        write_idx = tl.atomic_add(write_off_p, 1, mask=is_valid)
        
        # Drop query index into the flat array
        tl.store(INV_QUERIES_ptr + write_idx, t, mask=is_valid)

def _build_inverse_index_v2_atomic(indices, mask, T_kv):
    """
    Groups queries by their selected key indices. 
    100% Zero-Sync implementation using Atomic Routing.
    """
    B, H, T, k_sel = indices.shape
    device = indices.device
    BH = B * H

    # Pass 1: Count fan-in per key (Zero-Sync Pattern A)
    valid = (mask > 0.5) & (indices >= 0) & (indices < T_kv)
    bh_range = torch.arange(BH, device=device).view(B, H, 1, 1).expand_as(indices)
    flat_keys = (bh_range * T_kv + indices.clamp(0, T_kv - 1)).flatten()
    add_vals = valid.flatten().to(torch.int32) # No sum(), No sync
    
    inv_count = torch.zeros(BH * T_kv, device=device, dtype=torch.int32)
    inv_count.scatter_add_(0, flat_keys, add_vals)
    
    # Pass 2: Base Offsets via Cumsum
    inv_offset = torch.zeros(BH * T_kv, device=device, dtype=torch.int32)
    inv_offset[1:] = inv_count[:-1].cumsum(0).int()
    
    # Pass 3: Atomic Route to fill grouped query IDs
    # Pre-allocate theoretical max size to avoid synchronization
    # Using torch.empty to avoid the memset overhead of torch.zeros
    total_elements = B * H * T * k_sel
    inv_queries_routed = torch.empty(total_elements, device=device, dtype=torch.int32)
    write_offsets = inv_offset.clone() 
    
    # Higher BLOCK_SIZE and num_warps to hide memory latency during atomic scatter
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    _route_inverse_index_kernel[grid](
        indices, mask,
        write_offsets,
        inv_queries_routed,
        total_elements, B, H, T, k_sel, T_kv,
        indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3),
        mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8
    )
    
    return inv_queries_routed, inv_count, inv_offset

# ═══════════════════════════════════════════════════════════════════════
# torch.autograd.Function wrapper
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    class TritonSparseAttnFnV2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, indices, mask, scale):
            B, T, H, D = q.shape
            T_kv = k.shape[1]
            k_sel = indices.size(-1)

            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if mask.dtype != torch.float32:
                mask = mask.to(torch.float32)

            out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
            lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

            BLOCK_Q = 1   
            BLOCK_K = 64 # Improved occupancy for 4k-16k
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * H, T)

            _sparse_attn_fwd_kernel[grid](
                q, k, v,
                indices, mask,
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
                num_warps=4, num_stages=2,
            )

            ctx.save_for_backward(q, k, v, indices, mask, out.to(torch.float32), lse)
            ctx.scale = scale
            ctx.BLOCK_K = BLOCK_K
            ctx.BLOCK_D = BLOCK_D
            return out.to(q.dtype)

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
            delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
            
            _sparse_attn_bwd_preprocess[grid](
                out_fp32, do, delta,
                T, H, D,
                out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2), out_fp32.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                BLOCK_D=BLOCK_D,
            )

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
                BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
            )

            # Step 3: Zero-Sync Inverse Indexer (Atomic Route)
            inv_queries_routed, inv_count, inv_offset = _build_inverse_index_v2_atomic(
                indices, mask, T_kv
            )

            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            
            # High-Occupancy dK/dV Configuration
            grid_dkdv = (B * H, T_kv)
            _sparse_attn_bwd_dkdv_keymajor_kernel[grid_dkdv](
                q, k, v, do,
                lse, delta,
                dk, dv,
                inv_queries_routed, inv_count, inv_offset,
                T, T_kv, H, D,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                scale,
                BLOCK_Q_INNER=32, BLOCK_D=BLOCK_D,
                num_warps=8, num_stages=1,
            )

            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None

def triton_sparse_attention_v2(q, k, v, indices, mask, scale, use_triton_backward=True):
    return TritonSparseAttnFnV2.apply(q, k, v, indices, mask, scale)
