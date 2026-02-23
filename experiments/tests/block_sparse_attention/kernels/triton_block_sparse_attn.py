import torch
import triton
import triton.language as tl

@triton.jit
def _block_sparse_attn_fwd_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, BLOCK_IDX_ptr, MASK_ptr,
    OUT_ptr, LSE_ptr,
    # Shapes
    batch_size, seq_q, seq_kv, n_heads, d_head, num_blocks_selected,
    # Strides
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kh, stride_kd,
    stride_vb, stride_vk, stride_vh, stride_vd,
    stride_ib, stride_ih, stride_iq, stride_iblocks,
    stride_mb, stride_mh, stride_mq, stride_mblocks,
    stride_ob, stride_oq, stride_oh, stride_od,
    # Meta
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads
    
    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)
    
    q_mask = q_offs < seq_q
    
    q_ptrs = (Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qq
              + pid_h * stride_qh + d_offs[None, :] * stride_qd)
    q_i = tl.load(q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < d_head), other=0.0).to(tl.float32)
    
    m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    l_i = tl.full((BLOCK_Q,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)
    EPS = 1e-10
    
    idx_base = BLOCK_IDX_ptr + pid_b * stride_ib + pid_h * stride_ih
    mask_base = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh
    
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    
    for block_step in range(0, num_blocks_selected):
        idx_ptr = idx_base + q_offs * stride_iq + block_step * stride_iblocks
        mask_ptr = mask_base + q_offs * stride_mq + block_step * stride_mblocks
        
        block_id = tl.load(idx_ptr, mask=q_mask, other=-1)
        valid_block = tl.load(mask_ptr, mask=q_mask, other=0.0) > 0.5
        
        base_k_idx = block_id * CHUNK_SIZE
        chunk_offs = tl.arange(0, CHUNK_SIZE)
        
        k_indices = base_k_idx[:, None] + chunk_offs[None, :]
        k_mask = valid_block[:, None] & (k_indices < seq_kv) & (k_indices >= 0)
        
        k_ptrs = k_base + k_indices[:, :, None] * stride_kk + d_offs[None, None, :] * stride_kd
        v_ptrs = v_base + k_indices[:, :, None] * stride_vk + d_offs[None, None, :] * stride_vd
        
        fetch_mask = k_mask[:, :, None] & (d_offs[None, None, :] < d_head) & q_mask[:, None, None]
        
        k_vals = tl.load(k_ptrs, mask=fetch_mask, other=0.0).to(tl.float32)
        v_vals = tl.load(v_ptrs, mask=fetch_mask, other=0.0).to(tl.float32)
        
        scores = tl.sum(q_i[:, None, :] * k_vals, axis=2) * scale
        scores = tl.where(k_mask & q_mask[:, None], scores, float('-inf'))
        
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

@triton.jit
def _block_sparse_bwd_preprocess(
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
def _block_sparse_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    BLOCK_IDX_ptr, MASK_ptr,
    LSE_ptr, DELTA_ptr,
    DQ_ptr,
    seq_len, seq_kv, n_heads, d_head, num_blocks_selected,
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kh, stride_kd,
    stride_vb, stride_vk, stride_vh, stride_vd,
    stride_dob, stride_doq, stride_doh, stride_dod,
    stride_ib, stride_ih, stride_iq, stride_iblocks,
    stride_mb, stride_mh, stride_mq, stride_mblocks,
    stride_dqb, stride_dqq, stride_dqh, stride_dqd,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads
    
    q_offs = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offs = tl.arange(0, BLOCK_D)
    q_mask = q_offs < seq_len
    d_mask = d_offs < d_head

    # Pointers
    q_ptrs = Q_ptr + pid_b * stride_qb + q_offs[:, None] * stride_qq + pid_h * stride_qh + d_offs[None, :] * stride_qd
    do_ptrs = DO_ptr + pid_b * stride_dob + q_offs[:, None] * stride_doq + pid_h * stride_doh + d_offs[None, :] * stride_dod
    
    q_i = tl.load(q_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    do_i = tl.load(do_ptrs, mask=q_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    
    lse_i = tl.load(LSE_ptr + pid_b * n_heads * seq_len + pid_h * seq_len + q_offs, mask=q_mask, other=0.0)
    delta_i = tl.load(DELTA_ptr + pid_b * n_heads * seq_len + pid_h * seq_len + q_offs, mask=q_mask, other=0.0)

    dq_acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

    idx_base = BLOCK_IDX_ptr + pid_b * stride_ib + pid_h * stride_ih
    mask_base = MASK_ptr + pid_b * stride_mb + pid_h * stride_mh
    
    k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh

    for block_step in range(0, num_blocks_selected):
        idx_ptr = idx_base + q_offs * stride_iq + block_step * stride_iblocks
        mask_ptr = mask_base + q_offs * stride_mq + block_step * stride_mblocks
        
        block_id = tl.load(idx_ptr, mask=q_mask, other=-1)
        valid_block = tl.load(mask_ptr, mask=q_mask, other=0.0) > 0.5
        
        base_k_idx = block_id * CHUNK_SIZE
        chunk_offs = tl.arange(0, CHUNK_SIZE)
        
        k_indices = base_k_idx[:, None] + chunk_offs[None, :]
        k_mask = valid_block[:, None] & (k_indices < seq_kv) & (k_indices >= 0)
        
        k_ptrs = k_base + k_indices[:, :, None] * stride_kk + d_offs[None, None, :] * stride_kd
        v_ptrs = v_base + k_indices[:, :, None] * stride_vk + d_offs[None, None, :] * stride_vd
        fetch_mask = k_mask[:, :, None] & (d_offs[None, None, :] < d_head) & q_mask[:, None, None]
        
        k_vals = tl.load(k_ptrs, mask=fetch_mask, other=0.0).to(tl.float32)
        v_vals = tl.load(v_ptrs, mask=fetch_mask, other=0.0).to(tl.float32)
        
        scores = tl.sum(q_i[:, None, :] * k_vals, axis=2) * scale
        scores = tl.where(k_mask & q_mask[:, None], scores, float('-inf'))
        
        p = tl.exp(scores - lse_i[:, None])
        
        # dQ Math
        V_term = tl.sum(do_i[:, None, :] * v_vals, axis=2)
        v_term = tl.where(k_mask & q_mask[:, None], p * scale * (V_term - delta_i[:, None]), 0.0)
        
        dq_acc += tl.sum(v_term[:, :, None] * k_vals, axis=1)

    dq_out_ptrs = DQ_ptr + pid_b * stride_dqb + q_offs[:, None] * stride_dqq + pid_h * stride_dqh + d_offs[None, :] * stride_dqd
    tl.store(dq_out_ptrs, dq_acc, mask=q_mask[:, None] & d_mask[None, :])


@triton.jit
def _block_sparse_bwd_dkdv_blockmajor_kernel(
    Q_ptr, K_ptr, V_ptr, DO_ptr,
    LSE_ptr, DELTA_ptr,
    DK_workspace_ptr, DV_workspace_ptr,
    INV_QUERIES_ptr, INV_COUNT_ptr, INV_OFFSET_ptr,
    seq_len, seq_kv, n_heads, d_head,
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kh, stride_kd,
    stride_vb, stride_vk, stride_vh, stride_vd,
    stride_dob, stride_doq, stride_doh, stride_dod,
    stride_dkw_s, stride_dkw_b, stride_dkw_k, stride_dkw_h, stride_dkw_d,
    stride_dvw_s, stride_dvw_b, stride_dvw_k, stride_dvw_h, stride_dvw_d,
    stride_inv_b,   # [B, max_entries]
    stride_cnt_b,   # [B, num_chunks]
    stride_off_b,   # [B, num_chunks]
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_Q_INNER: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_id = tl.program_id(1)
    pid_sk = tl.program_id(2)
    SPLIT_K = tl.num_programs(2)

    pid_b = pid_bh // n_heads
    pid_h = pid_bh % n_heads

    fan_in = tl.load(INV_COUNT_ptr + pid_b * stride_cnt_b + chunk_id)
    if fan_in == 0:
        return

    chunk_size = (fan_in + SPLIT_K - 1) // SPLIT_K
    start_q_idx = pid_sk * chunk_size
    end_q_idx = tl.minimum(start_q_idx + chunk_size, fan_in)

    # Offset to contiguous keys
    k_offs = chunk_id * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    d_offs = tl.arange(0, BLOCK_D)
    
    k_mask = (k_offs[:, None] < seq_kv) & (d_offs[None, :] < d_head)
    k_ptrs = K_ptr + pid_b * stride_kb + k_offs[:, None] * stride_kk + pid_h * stride_kh + d_offs[None, :] * stride_kd
    v_ptrs = V_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vk + pid_h * stride_vh + d_offs[None, :] * stride_vd

    k_chunk = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
    v_chunk = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    dk_acc = tl.zeros((CHUNK_SIZE, BLOCK_D), dtype=tl.float32)
    dv_acc = tl.zeros((CHUNK_SIZE, BLOCK_D), dtype=tl.float32)

    inv_base = INV_QUERIES_ptr + pid_b * stride_inv_b + tl.load(INV_OFFSET_ptr + pid_b * stride_off_b + chunk_id)
    q_inner_offs = tl.arange(0, BLOCK_Q_INNER)
    d_mask = d_offs < d_head

    for q_start in range(start_q_idx, end_q_idx, BLOCK_Q_INNER):
        q_block_offs = q_start + q_inner_offs
        q_valid = q_block_offs < end_q_idx
        
        q_ids = tl.load(inv_base + q_block_offs, mask=q_valid, other=0)
        
        q_ptrs = Q_ptr + pid_b * stride_qb + q_ids[:, None] * stride_qq + pid_h * stride_qh + d_offs[None, :] * stride_qd
        do_ptrs = DO_ptr + pid_b * stride_dob + q_ids[:, None] * stride_doq + pid_h * stride_doh + d_offs[None, :] * stride_dod
        
        load_q_mask = q_valid[:, None] & d_mask[None, :]
        q_i = tl.load(q_ptrs, mask=load_q_mask, other=0.0).to(tl.float32)
        do_i = tl.load(do_ptrs, mask=load_q_mask, other=0.0).to(tl.float32)
        
        lse_ptrs = LSE_ptr + pid_b * n_heads * seq_len + pid_h * seq_len + q_ids
        delta_ptrs = DELTA_ptr + pid_b * n_heads * seq_len + pid_h * seq_len + q_ids
        
        lse_i = tl.load(lse_ptrs, mask=q_valid, other=float('inf'))
        delta_i = tl.load(delta_ptrs, mask=q_valid, other=0.0)

        # Dot product: q_i [BQ, D] * k_chunk [CHUNK, D] -> [BQ, CHUNK]
        scores = tl.sum(q_i[:, None, :] * k_chunk[None, :, :], axis=2) * scale
        
        # Valid mask combining q_valid and chunk items < seq_kv
        chunk_valid = k_offs < seq_kv
        valid = q_valid[:, None] & chunk_valid[None, :]
        scores = tl.where(valid, scores, float('-inf'))

        p = tl.exp(scores - lse_i[:, None])  # [BQ, CHUNK]
        
        # dK / dV math
        dp = p * scale * (tl.sum(do_i[:, None, :] * v_chunk[None, :, :], axis=2) - delta_i[:, None])
        
        dk_acc += tl.sum(dp[:, :, None] * q_i[:, None, :], axis=0)
        dv_acc += tl.sum(p[:, :, None] * do_i[:, None, :], axis=0)

    # Store workspace
    dk_out_ptrs = DK_workspace_ptr + pid_sk * stride_dkw_s + pid_b * stride_dkw_b + k_offs[:, None] * stride_dkw_k + pid_h * stride_dkw_h + d_offs[None, :] * stride_dkw_d
    dv_out_ptrs = DV_workspace_ptr + pid_sk * stride_dvw_s + pid_b * stride_dvw_b + k_offs[:, None] * stride_dvw_k + pid_h * stride_dvw_h + d_offs[None, :] * stride_dvw_d
    
    out_mask = (k_offs[:, None] < seq_kv) & d_mask[None, :]
    tl.store(dk_out_ptrs, dk_acc, mask=out_mask)
    tl.store(dv_out_ptrs, dv_acc, mask=out_mask)


def _build_chunk_inverse_index(block_indices, mask, num_chunks):
    """
    Inverse index mapping chunk_ids to list of query_ids.
    """
    B, H, T, k_limit = block_indices.shape
    device = block_indices.device

    idx = block_indices[:, 0].long() # [B, T, k_limit] - head 0 uses shared blocks
    msk = mask[:, 0]

    valid = (msk > 0.5) & (idx < num_chunks) & (idx >= 0)
    valid_int = valid.to(torch.int32)
    idx_flat = idx.reshape(B, -1).clamp(0, num_chunks - 1)

    inv_count = torch.zeros(B, num_chunks, device=device, dtype=torch.int32)
    inv_count.scatter_add_(1, idx_flat, valid_int)

    inv_offset = torch.zeros(B, num_chunks, device=device, dtype=torch.int32)
    inv_offset[:, 1:] = torch.cumsum(inv_count[:, :-1], dim=1)

    max_fan_in = int(inv_count.max().item())
    max_fan_in = min(max_fan_in, T * k_limit)
    
    inv_queries = torch.zeros(B, max_fan_in * num_chunks, device=device, dtype=torch.int32)

    q_pos = torch.arange(T, device=device).view(1, T, 1).expand(B, T, k_limit)
    for b in range(B):
        v = valid[b].reshape(-1)
        ci = idx[b].reshape(-1)[v]
        qi = q_pos[b].reshape(-1)[v]
        
        order = ci.argsort(stable=True)
        qi_sorted = qi[order]
        
        n_val = qi_sorted.shape[0]
        if n_val > 0:
            inv_queries[b, :n_val] = qi_sorted.int()

    return inv_queries, inv_count, inv_offset

class TritonBlockSparseAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_indices, mask, scale, chunk_size):
        B, T, H, D = q.shape
        T_kv = k.shape[1]
        num_blocks_selected = block_indices.size(-1)
        
        out = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
        lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        
        BLOCK_Q = min(128, triton.next_power_of_2(T)) 
        BLOCK_D = triton.next_power_of_2(D)
        
        grid = (B * H, triton.cdiv(T, BLOCK_Q))
        
        _block_sparse_attn_fwd_kernel[grid](
            q, k, v, block_indices, mask,
            out, lse,
            B, T, T_kv, H, D, num_blocks_selected,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scale,
            CHUNK_SIZE=chunk_size,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
        )
        
        ctx.save_for_backward(q, k, v, block_indices, mask, out, lse)
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.BLOCK_Q = BLOCK_Q
        ctx.BLOCK_D = BLOCK_D
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, do):
        q, k, v, block_indices, mask, out, lse = ctx.saved_tensors
        scale = ctx.scale
        chunk_size = ctx.chunk_size
        BLOCK_D = ctx.BLOCK_D
        BLOCK_Q = ctx.BLOCK_Q
        
        B, T, H, D = q.shape
        T_kv = k.shape[1]
        num_blocks_selected = block_indices.size(-1)
        
        if do.dtype != torch.float32:
            do = do.to(torch.float32)
            
        # 1. Preprocess
        delta = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        grid_pp = (B * H, triton.cdiv(T, 16))
        _block_sparse_bwd_preprocess[grid_pp](
            out, do, delta,
            T, H, D,
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            BLOCK_D=BLOCK_D
        )
        
        # 2. dQ
        dq = torch.empty_like(q, dtype=torch.float32)
        grid_dq = (B * H, triton.cdiv(T, BLOCK_Q))
        _block_sparse_bwd_dq_kernel[grid_dq](
            q, k, v, do,
            block_indices, mask,
            lse, delta,
            dq,
            T, T_kv, H, D, num_blocks_selected,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            block_indices.stride(0), block_indices.stride(1), block_indices.stride(2), block_indices.stride(3),
            mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            scale,
            CHUNK_SIZE=chunk_size,
            BLOCK_Q=BLOCK_Q,
            BLOCK_D=BLOCK_D,
        )
        
        # 3. dK / dV Workspace
        num_chunks = (T_kv + chunk_size - 1) // chunk_size
        inv_queries, inv_count, inv_offset = _build_chunk_inverse_index(block_indices, mask, num_chunks)
        
        SPLIT_K = 16
        dk_workspace = torch.zeros(SPLIT_K, B, T_kv, H, D, device=q.device, dtype=torch.float32)
        dv_workspace = torch.zeros(SPLIT_K, B, T_kv, H, D, device=q.device, dtype=torch.float32)
        
        BLOCK_Q_INNER = 8
        grid_dkdv = (B * H, num_chunks, SPLIT_K)
        _block_sparse_bwd_dkdv_blockmajor_kernel[grid_dkdv](
            q, k, v, do,
            lse, delta,
            dk_workspace, dv_workspace,
            inv_queries, inv_count, inv_offset,
            T, T_kv, H, D,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk_workspace.stride(0), dk_workspace.stride(1), dk_workspace.stride(2), dk_workspace.stride(3), dk_workspace.stride(4),
            dv_workspace.stride(0), dv_workspace.stride(1), dv_workspace.stride(2), dv_workspace.stride(3), dv_workspace.stride(4),
            inv_queries.stride(0),
            inv_count.stride(0),
            inv_offset.stride(0),
            scale,
            CHUNK_SIZE=chunk_size,
            BLOCK_Q_INNER=BLOCK_Q_INNER,
            BLOCK_D=BLOCK_D,
        )
        
        dk = dk_workspace.sum(dim=0)
        dv = dv_workspace.sum(dim=0)
        
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None, None, None

def triton_block_sparse_attention(q, k, v, block_indices, mask, scale, chunk_size=64):
    return TritonBlockSparseAttnFn.apply(q, k, v, block_indices, mask, scale, chunk_size)
