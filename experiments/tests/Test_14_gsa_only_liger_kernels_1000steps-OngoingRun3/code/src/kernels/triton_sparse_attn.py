"""
Triton Sparse Attention Kernel V2 — Key-Major dK/dV Backward
=============================================================

Changes from V1:
- Forward kernel: UNCHANGED
- Backward dQ: UNCHANGED
- Backward dK/dV: REWRITTEN with key-major algorithm
  - Builds inverse index: for each key, finds which queries selected it
  - Each thread block owns one (batch, head, key_position) tuple
  - Accumulates dK/dV in registers — ZERO atomics
  - Eliminates L2 cache serialization bottleneck

NOTE (perf + correctness):
- V2 key-major dK/dV (one block per (b,h,ki), register accumulate, single store) is the correct cure for the atomic/L2-serialization wall.
- HOWEVER, the current data plumbing is not scalable:
  (1) Dense [B,H,T,k_max] indices+mask is a padding tax (wasted gather + wasted math) and becomes structurally wrong for long-context tuning.
  (2) Any CPU/Python loop (per-batch argsort) and any allocation shaped like O(T*k_max) is an OOM bomb at T=256k.
  (3) Building inverse index from indices[:,0] silently assumes head-sharing; if heads diverge, gradients become wrong without crashing.
  (4) Dynamic fan_in loops inside Triton are risky; even if they compile, they invite divergence and unpredictable performance.
- Required fix: move to packed CSR (per (b,h,q): row_ptr + flat_indices) so kernels iterate exactly k_q (no padding),
  and build inverse index from packed COO/CSR (GPU-only) for key-major backward. Optional: block-CSR (block=4/8) to break gather BW limits.

Expected speedup for dK/dV: ~2.5x (from 3.8s → ~1.5s at full model scale)
"""

import torch
import torch.nn.functional as F
from typing import Optional

import triton
import triton.language as tl
HAS_TRITON = True

# Import profiling helpers
try:
    from ..profiler import kernel_region
except ImportError:
    # Fallback: no-op context manager
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


# ═══════════════════════════════════════════════════════════════════════
# GPU-Only CSR Builder (Data Plumbing)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def csr_row_count_kernel(
        IDX_ptr, MASK_ptr,
        COUNTS_ptr,  # int32 [B,H,T]
        T: tl.constexpr, K_MAX: tl.constexpr,
        stride_ib, stride_ih, stride_iq, stride_ik,
        stride_mb, stride_mh, stride_mq, stride_mk,
        stride_cb, stride_ch, stride_cq,
        BLOCK_K: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)

        k_offs = tl.arange(0, BLOCK_K)
        count = tl.zeros((), dtype=tl.int32)

        idx_row  = IDX_ptr  + pid_bh * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_bh * stride_mh + pid_q * stride_mq

        for k0 in tl.static_range(0, K_MAX, BLOCK_K):
            ks = k0 + k_offs
            in_range = ks < K_MAX

            idx = tl.load(idx_row + ks * stride_ik, mask=in_range, other=0).to(tl.int32)
            m   = tl.load(mask_row + ks * stride_mk, mask=in_range, other=0.0)

            v = (m > 0.5) & (idx >= 0)
            count += tl.sum(v.to(tl.int32), axis=0)

        out_ptr = COUNTS_ptr + pid_bh * stride_ch + pid_q * stride_cq
        tl.store(out_ptr, count)

    @triton.jit
    def csr_pack_kernel(
        IDX_ptr, MASK_ptr,
        ROWPTR_ptr,      # int32 [BH, T+1]
        CSR_IDX_ptr,     # int32 [total_nnz]
        CSR_QID_ptr,     # int32 [total_nnz] (optional)
        CSR_BHID_ptr,    # int32 [total_nnz] (optional)
        BH_BASE_ptr,     # int32 [BH] base offsets into CSR arrays
        T: tl.constexpr, K_MAX: tl.constexpr,
        stride_ih, stride_iq, stride_ik,
        stride_mh, stride_mq, stride_mk,
        stride_rh, stride_rq,  # rowptr strides: [BH, T+1]
        BLOCK_K: tl.constexpr,
        WRITE_QID: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)

        bh_base = tl.load(BH_BASE_ptr + pid_bh).to(tl.int32)
        row_start = tl.load(ROWPTR_ptr + pid_bh * stride_rh + pid_q * stride_rq).to(tl.int32)
        out_base = bh_base + row_start

        idx_row  = IDX_ptr  + pid_bh * stride_ih + pid_q * stride_iq
        mask_row = MASK_ptr + pid_bh * stride_mh + pid_q * stride_mq

        k_offs = tl.arange(0, BLOCK_K)
        running = tl.zeros((), dtype=tl.int32)

        for k0 in tl.static_range(0, K_MAX, BLOCK_K):
            ks = k0 + k_offs
            in_range = ks < K_MAX

            idx = tl.load(idx_row + ks * stride_ik, mask=in_range, other=0).to(tl.int32)
            m   = tl.load(mask_row + ks * stride_mk, mask=in_range, other=0.0)

            valid = (m > 0.5) & (idx >= 0)
            v_i32 = valid.to(tl.int32)

            prefix = tl.cumsum(v_i32, axis=0) - 1
            tile_n = tl.sum(v_i32, axis=0)

            pos = out_base + running + prefix

            tl.store(CSR_IDX_ptr + pos, idx, mask=valid)
            if WRITE_QID:
                tl.store(CSR_QID_ptr + pos, pid_q, mask=valid)
                tl.store(CSR_BHID_ptr + pos, pid_bh, mask=valid)

            running += tile_n

def build_csr_from_dense(indices: torch.Tensor, mask: torch.Tensor, K_MAX: int):
    """
    GPU-only CSR builder for dense [B,H,T,K_MAX] indices+mask.
    No Python loops, no argsort, no O(T*K_MAX) allocations.

    Returns:
      row_ptr:   [BH, T+1] int32
      bh_base:   [BH] int32 (base offset per BH into flat arrays)
      csr_idx:   [total_nnz] int32
      csr_qid:   [total_nnz] int32
      csr_bhid:  [total_nnz] int32
    """
    assert indices.is_cuda and mask.is_cuda
    B, H, T, K = indices.shape
    assert K == K_MAX, "Pass fixed K_MAX for stable compilation (pad upstream if needed)."

    BH = B * H
    idx = indices.reshape(BH, T, K_MAX).contiguous()
    msk = mask.reshape(BH, T, K_MAX).contiguous()

    counts = torch.empty((BH, T), device=idx.device, dtype=torch.int32)
    grid = (BH, T)
    csr_row_count_kernel[grid](
        idx, msk, counts,
        T=T, K_MAX=K_MAX,
        stride_ib=0, 
        stride_ih=idx.stride(0), stride_iq=idx.stride(1), stride_ik=idx.stride(2),
        stride_mb=0,
        stride_mh=msk.stride(0), stride_mq=msk.stride(1), stride_mk=msk.stride(2),
        stride_cb=0,
        stride_ch=counts.stride(0), stride_cq=counts.stride(1),
        BLOCK_K=128,
        num_warps=4,
    )

    row_ptr = torch.empty((BH, T + 1), device=idx.device, dtype=torch.int32)
    row_ptr[:, 0] = 0
    row_ptr[:, 1:] = torch.cumsum(counts, dim=1)

    bh_nnz = row_ptr[:, -1]
    bh_base = torch.empty((BH,), device=idx.device, dtype=torch.int32)
    bh_base[0] = 0
    bh_base[1:] = torch.cumsum(bh_nnz[:-1], dim=0)

    total_nnz = (bh_base[-1] + bh_nnz[-1]).item()
    csr_idx = torch.empty((total_nnz,), device=idx.device, dtype=torch.int32)
    csr_qid = torch.empty((total_nnz,), device=idx.device, dtype=torch.int32)
    csr_bhid = torch.empty((total_nnz,), device=idx.device, dtype=torch.int32)

    csr_pack_kernel[grid](
        idx, msk,
        row_ptr, csr_idx, csr_qid, csr_bhid, bh_base,
        T=T, K_MAX=K_MAX,
        stride_ih=idx.stride(0), stride_iq=idx.stride(1), stride_ik=idx.stride(2),
        stride_mh=msk.stride(0), stride_mq=msk.stride(1), stride_mk=msk.stride(2),
        stride_rh=row_ptr.stride(0), stride_rq=row_ptr.stride(1),
        BLOCK_K=128,
        WRITE_QID=True,
        num_warps=4,
        num_stages=2,
    )

    return row_ptr, bh_base, csr_idx, csr_qid, csr_bhid


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel (IDENTICAL to V1)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _sparse_attn_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        ROWPTR_ptr, BH_BASE_ptr, CSR_IDX_ptr,
        OUT_ptr, LSE_ptr,
        batch_size,
        seq_q, seq_kv, n_heads, d_head,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_rh, stride_rq,
        stride_ob, stride_oq, stride_oh, stride_od,
        scale,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh  % n_heads

        q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        q_mask = q_offs < seq_q

        q_ptrs = (Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qq
                  + pid_h * stride_qh + d_offs[None, :] * stride_qd)
        q_i = tl.load(q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < d_head), other=0.0).to(tl.float32)

        m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
        l_i = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        EPS = 1e-10

        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        # Load row start and end for each query in the block
        bh_base = tl.load(BH_BASE_ptr + pid_bh).to(tl.int32)
        q_start = tl.load(ROWPTR_ptr + pid_bh * stride_rh + q_offs * stride_rq, mask=q_mask, other=0).to(tl.int32)
        q_end = tl.load(ROWPTR_ptr + pid_bh * stride_rh + (q_offs + 1) * stride_rq, mask=q_mask, other=0).to(tl.int32)
        
        # Max iterations needed for this thread block
        max_k_t = tl.max(q_end - q_start)

        for k_step in range(0, max_k_t, BLOCK_K):
            k_block_offs = k_step + k_offs
            
            # Mask based on each query's individual count
            q_k_count = q_end - q_start
            idx_load_mask = k_block_offs[None, :] < q_k_count[:, None]
            
            # Global position in CSR_IDX
            pos = bh_base + q_start[:, None] + k_block_offs[None, :]
            
            q_k_mask = q_mask[:, None] & idx_load_mask
            
            qi_indices = tl.load(CSR_IDX_ptr + pos, mask=q_k_mask, other=0).to(tl.int32)
            
            qi_mask = q_k_mask & (qi_indices < seq_kv)

            k_ptrs = k_base + qi_indices[:, :, None] * stride_kk + d_offs[None, None, :] * stride_kd
            v_ptrs = v_base + qi_indices[:, :, None] * stride_vk + d_offs[None, None, :] * stride_vd
            
            kv_load_mask = qi_mask[:, :, None] & (d_offs[None, None, :] < d_head)
            
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q_i[:, None, :] * k_vals, axis=2) * scale
            scores = tl.where(qi_mask, scores, float('-inf'))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(m_new == float('-inf'), 0.0, tl.exp(m_i - m_new))
            is_inf_mask = (m_new == float('-inf'))[:, None]
            beta = tl.where(is_inf_mask, 0.0, tl.exp(scores - m_new[:, None]))
            
            l_i = alpha * l_i + tl.sum(beta, axis=1)
            acc = alpha[:, None] * acc + tl.sum(beta[:, :, None] * v_vals, axis=1)
            m_i = m_new

        l_i_safe = tl.where(l_i == 0.0, 1.0, tl.maximum(l_i, EPS))
        acc = acc / l_i_safe[:, None]

        out_row_ptr = (OUT_ptr + pid_b * stride_ob + q_offs[:, None] * stride_oq + pid_h * stride_oh)
        out_mask = q_mask[:, None] & (d_offs[None, :] < d_head)
        tl.store(out_row_ptr + d_offs[None, :] * stride_od, acc, mask=out_mask)

        lse_vals = tl.where(l_i == 0.0, -1e4, m_i + tl.log(l_i_safe))
        lse_ptrs = LSE_ptr + pid_b * n_heads * seq_q + pid_h * seq_q + q_offs
        tl.store(lse_ptrs, lse_vals, mask=q_mask)


# ═══════════════════════════════════════════════════════════════════════
# Backward kernels — preprocess and dQ (IDENTICAL to V1)
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


if HAS_TRITON:
    @triton.jit
    def _sparse_attn_bwd_dq_kernel(
        Q_ptr, K_ptr, V_ptr, DO_ptr,
        ROWPTR_ptr, BH_BASE_ptr, CSR_IDX_ptr,
        LSE_ptr, DELTA_ptr,
        DQ_ptr,
        seq_len, seq_kv, n_heads, d_head,
        stride_qb, stride_qq, stride_qh, stride_qd,
        stride_kb, stride_kk, stride_kh, stride_kd,
        stride_vb, stride_vk, stride_vh, stride_vd,
        stride_dob, stride_doq, stride_doh, stride_dod,
        stride_rh, stride_rq,
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

        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        dq_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        bh_base = tl.load(BH_BASE_ptr + pid_bh).to(tl.int32)
        q_start = tl.load(ROWPTR_ptr + pid_bh * stride_rh + pid_q * stride_rq).to(tl.int32)
        q_end = tl.load(ROWPTR_ptr + pid_bh * stride_rh + (pid_q + 1) * stride_rq).to(tl.int32)
        q_k_count = q_end - q_start

        for k_step in range(0, q_k_count, BLOCK_K):
            k_block_offs = k_step + k_offs
            idx_load_mask = k_block_offs < q_k_count
            pos = bh_base + q_start + k_block_offs
            
            qi_indices = tl.load(CSR_IDX_ptr + pos, mask=idx_load_mask, other=0).to(tl.int32)
            valid = idx_load_mask & (qi_indices < seq_kv) & row_active

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


# ═══════════════════════════════════════════════════════════════════════
# NEW: Key-Major dK/dV Backward Kernel (V2 — zero atomics)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
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
        stride_inv_bh,  # inv_queries: [BH, max_entries] OR contiguous 1D with base_off
        stride_cnt_bh,  # inv_count:   [BH, T_kv]
        stride_off_bh,  # inv_offset:  [BH, T_kv]
        scale,
        BLOCK_Q_INNER: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Key-Major dK/dV backward — ZERO atomics.

        Grid: (B * n_heads, T_kv)
          pid_bh → which (batch, head)
          pid_ki → which key position

        For each key position ki:
          1. Load K[ki] and V[ki] once
          2. Iterate over all queries that selected ki (from inverse index)
          3. Recompute attention weight P from saved LSE
          4. Accumulate dK and dV in registers
          5. Single write to global memory
        """
        pid_bh = tl.program_id(0)
        pid_ki = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        # ── Load fan-in count and offset (PER HEAD) ────────
        fan_in = tl.load(INV_COUNT_ptr + pid_bh * stride_cnt_bh + pid_ki).to(tl.int32)
        base_off = tl.load(INV_OFFSET_ptr + pid_bh * stride_off_bh + pid_ki).to(tl.int32)

        # ── Load K[b, ki, h, :] and V[b, ki, h, :] once ──────────────
        k_base = K_ptr + pid_b * stride_kb + pid_ki * stride_kk + pid_h * stride_kh
        k_vec = tl.load(k_base + d_offs * stride_kd, mask=d_mask, other=0.0).to(tl.float32)

        v_base = V_ptr + pid_b * stride_vb + pid_ki * stride_vk + pid_h * stride_vh
        v_vec = tl.load(v_base + d_offs * stride_vd, mask=d_mask, other=0.0).to(tl.float32)

        # ── Accumulators (register-local) ─────────────────────────────
        dk_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # ── Inverse index base pointer ────────────────────────────────
        inv_base = INV_QUERIES_ptr + pid_bh * stride_inv_bh + base_off

        # ── Q / dO base pointers ──────────────────────────────────────
        q_batch_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_batch_base = DO_ptr + pid_b * stride_dob + pid_h * stride_doh

        # ── LSE / delta base (layout: [B, H, T]) ─────────────────────
        lse_base = pid_b * n_heads * seq_len + pid_h * seq_len

        q_inner_offs = tl.arange(0, BLOCK_Q_INNER)

        # ── Main loop: iterate ONLY over this key's fan-in (not global max) ─
        for q_start in range(0, fan_in, BLOCK_Q_INNER):
            q_block_offs = q_start + q_inner_offs
            q_valid = q_block_offs < fan_in

            # Load query IDs from inverse index
            q_ids = tl.load(inv_base + q_block_offs, mask=q_valid, other=0)

            # Load Q[b, q_id, h, :]: [BLOCK_Q_INNER, BLOCK_D]
            q_ptrs = q_batch_base + q_ids[:, None] * stride_qq + d_offs[None, :] * stride_qd
            qd_mask = q_valid[:, None] & d_mask[None, :]
            q_vals = tl.load(q_ptrs, mask=qd_mask, other=0.0).to(tl.float32)

            # Load dO[b, q_id, h, :]: [BLOCK_Q_INNER, BLOCK_D]
            do_ptrs = do_batch_base + q_ids[:, None] * stride_doq + d_offs[None, :] * stride_dod
            do_vals = tl.load(do_ptrs, mask=qd_mask, other=0.0).to(tl.float32)

            # Load LSE[b, h, q_id] and delta[b, h, q_id]: [BLOCK_Q_INNER]
            lse_vals = tl.load(LSE_ptr + lse_base + q_ids, mask=q_valid, other=-1e4)
            delta_vals = tl.load(DELTA_ptr + lse_base + q_ids, mask=q_valid, other=0.0)

            # Row-active check (sentinel LSE = -1e4 for fully-masked queries)
            row_active = lse_vals > -1e3
            active = q_valid & row_active

            # ── Recompute attention scores and weights ────────────────
            # score = Q[q] · K[ki] * scale
            scores = tl.sum(q_vals * k_vec[None, :], axis=1) * scale  # [BLOCK_Q_INNER]
            scores = tl.where(active, scores, float('-inf'))

            # P = exp(score - LSE[q]), clamped for numerical safety
            p_i = tl.exp(tl.minimum(scores - lse_vals, 50.0))
            p_i = tl.where(active, p_i, 0.0)  # [BLOCK_Q_INNER]

            # ── Compute dS = P * (dO·V - delta) ──────────────────────
            do_v = tl.sum(do_vals * v_vec[None, :], axis=1)  # [BLOCK_Q_INNER]
            ds_i = p_i * (do_v - delta_vals)                  # [BLOCK_Q_INNER]

            # ── Accumulate dK += scale * dS * Q[q] ───────────────────
            dk_acc += tl.sum(ds_i[:, None] * q_vals, axis=0) * scale  # [BLOCK_D]

            # ── Accumulate dV += P * dO[q] ───────────────────────────
            dv_acc += tl.sum(p_i[:, None] * do_vals, axis=0)  # [BLOCK_D]

        # ── Single store to global memory (no atomics!) ───────────────
        dk_base = DK_ptr + pid_b * stride_dkb + pid_ki * stride_dkk + pid_h * stride_dkh
        tl.store(dk_base + d_offs * stride_dkd, dk_acc, mask=d_mask)

        dv_base = DV_ptr + pid_b * stride_dvb + pid_ki * stride_dvk + pid_h * stride_dvh
        tl.store(dv_base + d_offs * stride_dvd, dv_acc, mask=d_mask)


# ═══════════════════════════════════════════════════════════════════════
# Inverse Index Builder (GPU COO -> CSR)
# ═══════════════════════════════════════════════════════════════════════

def _build_inverse_index_from_csr(csr_idx, csr_qid, csr_bhid, B, H, T_kv):
    """
    Build per-head inverse index from purely packed 1D CSR representations.
    Solves the head-sharing bug and eliminates python argsort overhead entirely.
    """
    device = csr_idx.device
    BH = B * H

    # 1. We create a combined sort key to sort keys globally but bounded by BH segment
    # key = bh_id * T_kv + csr_idx
    sort_keys = (csr_bhid.long() * T_kv) + csr_idx.long()
    
    # 2. Global GPU sort on 1D tensor of exact nonzero size
    order = sort_keys.argsort(stable=True)
    inv_queries_flat = csr_qid[order]  # [total_nnz]
    
    # 3. Dense scatter-add to count per (bh, k)
    inv_count = torch.zeros(BH * T_kv, device=device, dtype=torch.int32)
    inv_count.scatter_add_(0, sort_keys, torch.ones_like(sort_keys, dtype=torch.int32))
    inv_count = inv_count.view(BH, T_kv)
    
    # 4. Offsets (Prefix sum)
    inv_offset = torch.zeros(BH, T_kv, device=device, dtype=torch.int32)
    inv_offset[:, 1:] = inv_count[:, :-1].cumsum(dim=1).int()

    # The array is actually purely dense 1D logically from the sort... 
    # But Triton expects 2D strides. We can allocate a [BH, total_nnz] array, OR
    # tightly bound the offset lookups because `inv_queries_flat` is fully dense and packed!
    # Let's allocate a 2D array [BH, max_per_bh] to keep Triton simple, or just use 1D.
    # Actually, we can return inv_queries_flat and stride_inv_bh=0 since inv_queries_flat
    # already groups everything perfectly by base_off! 
    # Wait, `base_off` needs to point to the absolute offset in the 1D flat array.
    
    total_per_bh = inv_count.sum(dim=1) # [BH]
    bh_global_offsets = torch.zeros(BH, device=device, dtype=torch.int32)
    bh_global_offsets[1:] = total_per_bh[:-1].cumsum(dim=0).int()
    
    # Add absolute BH offset to our relative inv_offset
    inv_offset += bh_global_offsets.unsqueeze(1)
    
    return inv_queries_flat, inv_count, inv_offset

# ═══════════════════════════════════════════════════════════════════════
# torch.autograd.Function wrapper (V2)
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    class TritonSparseAttnFnV2(torch.autograd.Function):
        """
        Fused sparse attention with Triton forward + Key-Major backward.

        Forward:   IDENTICAL to V1 (online softmax, saves LSE)
        Backward:  dQ via query-major (no atomics, same as V1)
                   dK/dV via KEY-MAJOR (inverse index, ZERO atomics)
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
            with kernel_region("sparse_attn_csr.fwd_total"):
                B, T, H, D = q.shape
                T_kv = k.shape[1]
                k_sel = indices.size(-1)

                if indices.dtype != torch.int64:
                    indices = indices.to(torch.int64)
                if mask.dtype != torch.float32:
                    mask = mask.to(torch.float32)

                # Build purely packed CSR index representations
                with kernel_region("sparse_attn_csr.fwd_build_csr"):
                    row_ptr, bh_base, csr_idx, csr_qid, csr_bhid = build_csr_from_dense(indices, mask, k_sel)

                with kernel_region("sparse_attn_csr.fwd_alloc"):
                    out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
                    lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

                BLOCK_Q = 2   # 18,990 tok/sec baseline: BQ=2
                BLOCK_K = 128 # Usually stable, now bounds max loops
                BLOCK_D = triton.next_power_of_2(D)
                grid = (B * H, triton.cdiv(T, BLOCK_Q))

                with kernel_region("sparse_attn_csr.fwd_kernel"):
                    _sparse_attn_fwd_kernel[grid](
                        q, k, v,
                        row_ptr, bh_base, csr_idx,
                        out, lse,
                        B, T, T_kv, H, D,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        row_ptr.stride(0), row_ptr.stride(1),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        scale,
                        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                        num_warps=4, num_stages=2,
                    )

                with kernel_region("sparse_attn_csr.fwd_convert"):
                    out_typed = out.to(q.dtype)
                
                # Save compact tensors, completely bypassing padding masks for backward pass!
                ctx.save_for_backward(q, k, v, row_ptr, bh_base, csr_idx, csr_qid, csr_bhid, out, lse)
                ctx.scale = scale
                ctx.BLOCK_K = BLOCK_K
                ctx.BLOCK_D = BLOCK_D

                return out_typed

        @staticmethod
        def backward(ctx, grad_output):
            with kernel_region("sparse_attn_csr.bwd_total"):
                q, k, v, row_ptr, bh_base, csr_idx, csr_qid, csr_bhid, out_fp32, lse = ctx.saved_tensors
                scale = ctx.scale
                BLOCK_K = ctx.BLOCK_K
                BLOCK_D = ctx.BLOCK_D

                B, T, H, D = q.shape
                T_kv = k.shape[1]
                k_sel = indices.size(-1)
                grid = (B * H, T)

                with kernel_region("sparse_attn_csr.bwd_convert_do"):
                    do = grad_output.contiguous().to(torch.float32)

                # ── Step 1: delta[b,h,q] = sum_d(O * dO) ─────────────────
                with kernel_region("sparse_attn_csr.bwd_preprocess"):
                    delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
                    _sparse_attn_bwd_preprocess[grid](
                        out_fp32, do, delta,
                        T, H, D,
                        out_fp32.stride(0), out_fp32.stride(1), out_fp32.stride(2), out_fp32.stride(3),
                        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                        BLOCK_D=BLOCK_D,
                    )

                # ── Step 2: dQ (query-major, using CSR indices!) ─────
                with kernel_region("sparse_attn_csr.bwd_dq"):
                    dq = torch.empty_like(q, dtype=torch.float32)
                    _sparse_attn_bwd_dq_kernel[grid](
                        q, k, v, do,
                        row_ptr, bh_base, csr_idx,
                        lse, delta,
                        dq,
                        T, T_kv, H, D,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                        row_ptr.stride(0), row_ptr.stride(1),
                        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                        scale,
                        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
                    )

                # ── Step 3: Fast GPU Inverse Index from packed CSR ──────────
                with kernel_region("sparse_attn_csr.bwd_inv_index"):
                    inv_queries_flat, inv_count, inv_offset = _build_inverse_index_from_csr(
                        csr_idx, csr_qid, csr_bhid, B, H, T_kv
                    )

                # ── Step 4: dK/dV via KEY-MAJOR kernel (ZERO atomics!) ───
                with kernel_region("sparse_attn_csr.bwd_dkdv"):
                    dk = torch.zeros(B, T_kv, H, D, device=q.device, dtype=torch.float32)
                    dv = torch.zeros(B, T_kv, H, D, device=q.device, dtype=torch.float32)

                    BLOCK_Q_INNER = 4  # Process 4 queries per inner loop iteration

                    grid_dkdv = (B * H, T_kv)
                    _sparse_attn_bwd_dkdv_keymajor_kernel[grid_dkdv](
                        q, k, v, do,
                        lse, delta,
                        dk, dv,
                        inv_queries_flat, inv_count, inv_offset,
                        T, T_kv, H, D,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                        0,                      # stride_inv_bh (inv_queries logcially 1D)
                        inv_count.stride(0),    # stride_cnt_bh
                        inv_offset.stride(0),   # stride_off_bh
                        scale,
                        BLOCK_Q_INNER=BLOCK_Q_INNER, BLOCK_D=BLOCK_D,
                        num_warps=4, num_stages=1,
                    )

                with kernel_region("sparse_attn_csr.bwd_convert"):
                    dq_out = dq.to(q.dtype)
                    dk_out = dk.to(k.dtype)
                    dv_out = dv.to(v.dtype)

                return dq_out, dk_out, dv_out, None, None, None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

USE_TRITON_BACKWARD = True


def triton_sparse_attention_v2(
    q: torch.Tensor,        # [B, T, H, D]
    k: torch.Tensor,        # [B, T_kv, H, D]
    v: torch.Tensor,        # [B, T_kv, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,     # [B, H, T, k_sel]
    scale: float,
    use_triton_backward: bool = True,
) -> torch.Tensor:
    """
    Sparse attention with Key-Major backward (V2).

    Identical forward pass to V1. Backward uses inverse-index key-major
    dK/dV kernel that eliminates all atomic operations.

    Args:
        q, k, v:  [B, T, H, D] query/key/value tensors
        indices:  [B, H, T, k_sel] selected key indices per query
        mask:     [B, H, T, k_sel] validity mask (1.0 = valid)
        scale:    attention scale factor (typically 1/sqrt(d_head))
        use_triton_backward: ignored (always uses V2 backward)

    Returns:
        out: [B, T, H, D] attention output
    """
    return TritonSparseAttnFnV2.apply(q, k, v, indices, mask, scale)
