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

Expected speedup for dK/dV: ~2.5x (from 3.8s → ~1.5s at full model scale)
"""

import torch
import triton
import triton.language as tl

HAS_TRITON = True


# ═══════════════════════════════════════════════════════════════════════
# Forward kernel (IDENTICAL to V1)
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
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        q_mask = q_offs < seq_q

        q_ptrs = (
            Q_ptr
            + pid_b * stride_qb
            + q_offs[:, None] * stride_qq
            + pid_h * stride_qh
            + d_offs[None, :] * stride_qd
        )
        q_i = tl.load(
            q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < d_head), other=0.0
        ).to(tl.float32)

        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
        EPS = 1e-10

        idx_base = IDX_ptr + pid_b * stride_ib + pid_h * stride_ih
        mask_base = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

        for k_block in range(0, k_selected, BLOCK_K):
            k_block_offs = k_block + k_offs
            idx_load_mask = k_block_offs < k_selected

            idx_ptrs = (
                idx_base
                + q_offs[:, None] * stride_iq
                + k_block_offs[None, :] * stride_ik
            )
            mask_ptrs = (
                mask_base
                + q_offs[:, None] * stride_mq
                + k_block_offs[None, :] * stride_mk
            )
            q_k_mask = q_mask[:, None] & idx_load_mask[None, :]
            qi_indices = tl.load(idx_ptrs, mask=q_k_mask, other=0)
            qi_mask_val = tl.load(mask_ptrs, mask=q_k_mask, other=0.0)
            qi_mask = (qi_mask_val > 0.5) & (qi_indices < seq_kv)

            k_ptrs = (
                k_base
                + qi_indices[:, :, None] * stride_kk
                + d_offs[None, None, :] * stride_kd
            )
            v_ptrs = (
                v_base
                + qi_indices[:, :, None] * stride_vk
                + d_offs[None, None, :] * stride_vd
            )
            kv_load_mask = (
                qi_mask[:, :, None]
                & (d_offs[None, None, :] < d_head)
                & q_mask[:, None, None]
            )
            k_vals = tl.load(k_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_load_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q_i[:, None, :] * k_vals, axis=2) * scale
            valid = q_mask[:, None] & idx_load_mask[None, :] & qi_mask
            scores = tl.where(valid, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, block_max)
            alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
            is_inf_mask = (m_new == float("-inf"))[:, None]
            beta = tl.where(is_inf_mask, 0.0, tl.exp(scores - m_new[:, None]))
            l_i = alpha * l_i + tl.sum(beta, axis=1)
            acc = alpha[:, None] * acc + tl.sum(beta[:, :, None] * v_vals, axis=1)
            m_i = m_new

        l_i_safe = tl.where(l_i == 0.0, 1.0, tl.maximum(l_i, EPS))
        acc = acc / l_i_safe[:, None]

        out_row_ptr = (
            OUT_ptr
            + pid_b * stride_ob
            + q_offs[:, None] * stride_oq
            + pid_h * stride_oh
        )
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
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        o_base = O_ptr + pid_b * stride_ob + pid_q * stride_oq + pid_h * stride_oh
        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        o_i = tl.load(o_base + d_offs * stride_od, mask=d_mask, other=0.0).to(
            tl.float32
        )
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(
            tl.float32
        )

        delta_i = tl.sum(o_i * do_i)
        ld_offset = pid_b * n_heads * seq_len + pid_h * seq_len + pid_q
        tl.store(DELTA_ptr + ld_offset, delta_i)


if HAS_TRITON:

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
        pid_bh = tl.program_id(0)
        pid_q = tl.program_id(1)
        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        k_offs = tl.arange(0, BLOCK_K)
        d_mask = d_offs < d_head

        q_base = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + pid_h * stride_qh
        q_i = tl.load(q_base + d_offs * stride_qd, mask=d_mask, other=0.0).to(
            tl.float32
        )

        do_base = DO_ptr + pid_b * stride_dob + pid_q * stride_doq + pid_h * stride_doh
        do_i = tl.load(do_base + d_offs * stride_dod, mask=d_mask, other=0.0).to(
            tl.float32
        )

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
            qi_indices = tl.load(
                idx_row + k_block_offs * stride_ik, mask=idx_load_mask, other=0
            )
            qi_mask_val = tl.load(
                mask_row + k_block_offs * stride_mk, mask=idx_load_mask, other=0.0
            )
            valid = (
                idx_load_mask & (qi_mask_val > 0.5) & (qi_indices < seq_kv) & row_active
            )

            k_ptrs = (
                k_base + qi_indices[:, None] * stride_kk + d_offs[None, :] * stride_kd
            )
            v_ptrs = (
                v_base + qi_indices[:, None] * stride_vk + d_offs[None, :] * stride_vd
            )
            kv_mask = valid[:, None] & d_mask[None, :]
            k_vals = tl.load(k_ptrs, mask=kv_mask, other=0.0).to(tl.float32)
            v_vals = tl.load(v_ptrs, mask=kv_mask, other=0.0).to(tl.float32)

            scores = tl.sum(q_i[None, :] * k_vals, axis=1) * scale
            scores = tl.where(valid, scores, float("-inf"))
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
    def _sparse_attn_bwd_dkdv_keymajor_splitk_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        DO_ptr,
        LSE_ptr,
        DELTA_ptr,
        DK_workspace_ptr,
        DV_workspace_ptr,
        INV_QUERIES_ptr,
        INV_COUNT_ptr,
        INV_OFFSET_ptr,
        seq_len,
        seq_kv,
        n_heads,
        d_head,
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
        stride_dkw_s,
        stride_dkw_b,
        stride_dkw_k,
        stride_dkw_h,
        stride_dkw_d,
        stride_dvw_s,
        stride_dvw_b,
        stride_dvw_k,
        stride_dvw_h,
        stride_dvw_d,
        stride_inv_b,  # inv_queries: [B, max_entries]
        stride_cnt_b,  # inv_count:   [B, T_kv]
        stride_off_b,  # inv_offset:  [B, T_kv]
        scale,
        BLOCK_Q_INNER: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Key-Major dK/dV backward — ZERO atomics, Split-K scaled.

        Grid: (B * n_heads, T_kv, SPLIT_K)
          pid_bh → which (batch, head)
          pid_ki → which key position
          pid_sk → which split of the queries

        For each key position ki:
          1. Load fan_in and determine range for this pid_sk
          2. Iterates ONLY over its chunk of queries
          3. Accumulates dK and dV in registers
          4. Single write to workspace[pid_sk, b, ki, h, d]
        """
        pid_bh = tl.program_id(0)
        pid_ki = tl.program_id(1)
        pid_sk = tl.program_id(2)
        SPLIT_K = tl.num_programs(2)

        pid_b = pid_bh // n_heads
        pid_h = pid_bh % n_heads

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < d_head

        # ── Load fan-in count and offset (per batch×head) ────────────
        bh_idx = pid_b * n_heads + pid_h
        fan_in = tl.load(INV_COUNT_ptr + bh_idx * stride_cnt_b + pid_ki).to(tl.int32)
        base_off = tl.load(INV_OFFSET_ptr + bh_idx * stride_off_b + pid_ki).to(tl.int32)

        # ── Load K[b, ki, h, :] and V[b, ki, h, :] once ──────────────
        k_base = K_ptr + pid_b * stride_kb + pid_ki * stride_kk + pid_h * stride_kh
        k_vec = tl.load(k_base + d_offs * stride_kd, mask=d_mask, other=0.0).to(
            tl.float32
        )

        v_base = V_ptr + pid_b * stride_vb + pid_ki * stride_vk + pid_h * stride_vh
        v_vec = tl.load(v_base + d_offs * stride_vd, mask=d_mask, other=0.0).to(
            tl.float32
        )

        # ── Accumulators (register-local) ─────────────────────────────
        dk_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        dv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

        # ── Inverse index base pointer ────────────────────────────────
        inv_base = INV_QUERIES_ptr + bh_idx * stride_inv_b + base_off

        # ── Q / dO base pointers ──────────────────────────────────────
        q_batch_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        do_batch_base = DO_ptr + pid_b * stride_dob + pid_h * stride_doh

        # ── LSE / delta base (layout: [B, H, T]) ─────────────────────
        lse_base = pid_b * n_heads * seq_len + pid_h * seq_len

        q_inner_offs = tl.arange(0, BLOCK_Q_INNER)

        # ── Determine query bounds for this SPLIT_K block ─────────────
        chunk_size = (fan_in + SPLIT_K - 1) // SPLIT_K
        start_q_idx = pid_sk * chunk_size
        end_q_idx = tl.minimum(start_q_idx + chunk_size, fan_in)

        # ── Main loop: iterate over this block's chunk of fan-in ──────
        for q_start in range(start_q_idx, end_q_idx, BLOCK_Q_INNER):
            q_block_offs = q_start + q_inner_offs
            q_valid = q_block_offs < end_q_idx

            # Load query IDs from inverse index
            q_ids = tl.load(inv_base + q_block_offs, mask=q_valid, other=0)

            # Load Q[b, q_id, h, :]: [BLOCK_Q_INNER, BLOCK_D]
            q_ptrs = (
                q_batch_base + q_ids[:, None] * stride_qq + d_offs[None, :] * stride_qd
            )
            qd_mask = q_valid[:, None] & d_mask[None, :]
            q_vals = tl.load(q_ptrs, mask=qd_mask, other=0.0).to(tl.float32)

            # Load dO[b, q_id, h, :]: [BLOCK_Q_INNER, BLOCK_D]
            do_ptrs = (
                do_batch_base
                + q_ids[:, None] * stride_doq
                + d_offs[None, :] * stride_dod
            )
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
            scores = tl.where(active, scores, float("-inf"))

            # P = exp(score - LSE[q]), clamped for numerical safety
            p_i = tl.exp(tl.minimum(scores - lse_vals, 50.0))
            p_i = tl.where(active, p_i, 0.0)  # [BLOCK_Q_INNER]

            # ── Compute dS = P * (dO·V - delta) ──────────────────────
            do_v = tl.sum(do_vals * v_vec[None, :], axis=1)  # [BLOCK_Q_INNER]
            ds_i = p_i * (do_v - delta_vals)  # [BLOCK_Q_INNER]

            # ── Accumulate dK += scale * dS * Q[q] ───────────────────
            dk_acc += tl.sum(ds_i[:, None] * q_vals, axis=0) * scale  # [BLOCK_D]

            # ── Accumulate dV += P * dO[q] ───────────────────────────
            dv_acc += tl.sum(p_i[:, None] * do_vals, axis=0)  # [BLOCK_D]

        # ── Single store to SPLIT_K workspace ─────────────────────────
        dk_base = (
            DK_workspace_ptr
            + pid_sk * stride_dkw_s
            + pid_b * stride_dkw_b
            + pid_ki * stride_dkw_k
            + pid_h * stride_dkw_h
        )
        tl.store(dk_base + d_offs * stride_dkw_d, dk_acc, mask=d_mask)

        dv_base = (
            DV_workspace_ptr
            + pid_sk * stride_dvw_s
            + pid_b * stride_dvw_b
            + pid_ki * stride_dvw_k
            + pid_h * stride_dvw_h
        )
        tl.store(dv_base + d_offs * stride_dvw_d, dv_acc, mask=d_mask)


# ═══════════════════════════════════════════════════════════════════════
# Inverse Index Builder (PyTorch — runs on GPU, ~1ms)
# ═══════════════════════════════════════════════════════════════════════


def _build_inverse_index(indices, mask, T_kv):
    """
    Build inverse index: for each key position, which queries selected it.
    Per (batch, head) pair — each head has its own index selections.
    Fully vectorized — single batch sort, no Python for-loops.

    Args:
        indices: [B, H, T, k_sel] int64
        mask:    [B, H, T, k_sel] float32
        T_kv:    int — number of key/value positions

    Returns:
        inv_queries: [B*H, max_entries] int32 — query IDs sorted by key position
        inv_count:   [B*H, T_kv] int32 — fan-in per key per (batch, head)
        inv_offset:  [B*H, T_kv] int32 — offset into inv_queries per key
    """
    B, H, T, k_sel = indices.shape
    device = indices.device
    BH = B * H

    # Reshape to [B*H, T, k_sel] — process all (batch, head) pairs
    idx = indices.reshape(BH, T, k_sel).long()
    msk = mask.reshape(BH, T, k_sel)

    # Valid entries: mask active AND index in bounds
    valid = (msk > 0.5) & (idx >= 0) & (idx < T_kv)  # [BH, T, k_sel]

    # Query position broadcast
    q_pos = torch.arange(T, device=device, dtype=torch.int32)
    q_pos = q_pos.view(1, T, 1).expand(BH, T, k_sel)  # [BH, T, k_sel]

    # Count per key per (batch, head) using scatter_add
    idx_clamped = idx.clamp(0, T_kv - 1)
    valid_int = valid.int().reshape(BH, -1)
    idx_flat = idx_clamped.reshape(BH, -1)
    inv_count = torch.zeros(BH, T_kv, device=device, dtype=torch.int32)
    inv_count.scatter_add_(1, idx_flat, valid_int)

    # Offsets: exclusive prefix sum
    inv_offset = torch.zeros(BH, T_kv, device=device, dtype=torch.int32)
    inv_offset[:, 1:] = inv_count[:, :-1].cumsum(dim=1).int()

    # Vectorized batch sort: pad invalid entries with large key so they sort to end
    # This replaces the Python for-loop with a single batched sort
    LARGE_KEY = T_kv  # invalid entries sort after all valid keys [0..T_kv-1]
    key_for_sort = torch.where(
        valid.reshape(BH, -1),
        idx_clamped.reshape(BH, -1),
        torch.full((BH, T * k_sel), LARGE_KEY, device=device, dtype=idx_clamped.dtype),
    )
    _, sort_order = key_for_sort.sort(dim=1, stable=True)
    inv_queries = q_pos.reshape(BH, -1).gather(1, sort_order.long()).int()

    return inv_queries, inv_count, inv_offset


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
            B, T, H, D = q.shape
            T_kv = k.shape[1]
            k_sel = indices.size(-1)

            if indices.dtype != torch.int64:
                indices = indices.to(torch.int64)
            if mask.dtype != torch.float32:
                mask = mask.to(torch.float32)

            out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
            lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)

            BLOCK_Q = 16  # exp 6: increase from 2 to 16
            BLOCK_K = triton.next_power_of_2(min(128, k_sel))  # FIX-PERF-03a
            BLOCK_D = triton.next_power_of_2(D)
            grid = (B * H, triton.cdiv(T, BLOCK_Q))

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
                BLOCK_Q=BLOCK_Q,
                BLOCK_K=BLOCK_K,
                BLOCK_D=BLOCK_D,
                num_warps=2,
                num_stages=2,  # exp 30: try 2 warps
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

            # ── Step 1: delta[b,h,q] = sum_d(O * dO) ─────────────────
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

            # ── Step 2: dQ (query-major, no atomics — same as V1) ─────
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
                BLOCK_D=BLOCK_D,  # 18,990 baseline: BK=128 (from fwd)
            )

            # ── Step 3: Build inverse index (key → queries) ──────────
            inv_queries, inv_count, inv_offset = _build_inverse_index(
                indices, mask, T_kv
            )

            # ── Step 4: dK/dV via KEY-MAJOR kernel (ZERO atomics!) ───
            SPLIT_K = 16  # Distribute hot keys across up to 16 thread blocks

            # Workspace shape: [SPLIT_K, B, T_kv, H, D]
            dk_workspace = torch.zeros(
                SPLIT_K, B, T_kv, H, D, device=q.device, dtype=torch.float32
            )
            dv_workspace = torch.zeros(
                SPLIT_K, B, T_kv, H, D, device=q.device, dtype=torch.float32
            )

            BLOCK_Q_INNER = 8  # Process 8 queries per inner loop iteration

            grid_dkdv = (B * H, T_kv, SPLIT_K)
            _sparse_attn_bwd_dkdv_keymajor_splitk_kernel[grid_dkdv](
                q,
                k,
                v,
                do,
                lse,
                delta,
                dk_workspace,
                dv_workspace,
                inv_queries,
                inv_count,
                inv_offset,
                T,
                T_kv,
                H,
                D,
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
                dk_workspace.stride(0),
                dk_workspace.stride(1),
                dk_workspace.stride(2),
                dk_workspace.stride(3),
                dk_workspace.stride(4),
                dv_workspace.stride(0),
                dv_workspace.stride(1),
                dv_workspace.stride(2),
                dv_workspace.stride(3),
                dv_workspace.stride(4),
                inv_queries.stride(0),  # stride_inv_b
                inv_count.stride(0),  # stride_cnt_b
                inv_offset.stride(0),  # stride_off_b
                scale,
                BLOCK_Q_INNER=BLOCK_Q_INNER,
                BLOCK_D=BLOCK_D,
                num_warps=4,
                num_stages=1,
            )

            # ── Step 5: Reduce SPLIT_K workspace ───────────────────────
            # High-speed PyTorch C++ reduction (memory bound, <2ms)
            dk = dk_workspace.sum(dim=0)
            dv = dv_workspace.sum(dim=0)

            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

USE_TRITON_BACKWARD = True


def triton_sparse_attention_v2(
    q: torch.Tensor,  # [B, T, H, D]
    k: torch.Tensor,  # [B, T_kv, H, D]
    v: torch.Tensor,  # [B, T_kv, H, D]
    indices: torch.Tensor,  # [B, H, T, k_sel]
    mask: torch.Tensor,  # [B, H, T, k_sel]
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
