import torch

import triton
import triton.language as tl

HAS_TRITON = True

@triton.jit
def _block_indexer_fwd_kernel(
    # Pointers
    Q_ptr, K_ptr, W_ptr, B_ptr, OUT_ptr,
    # Dimensions
    batch_size, seq_q, seq_kv, n_heads, d_idx,
    num_blocks, block_size,
    # Strides
    stride_qb, stride_qq, stride_qh, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_wb, stride_wq, stride_wh,
    stride_ob, stride_oq, stride_ok,
    # Scales
    scale,
    # Meta
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Block-Sparse Gated Indexer.
    
    Q is shaped [B, seq_q, H, d_idx].
    K is shaped [B, num_blocks, d_idx] -> Pre-pooled keys!
    
    Outputs a score for each BLOCK, rather than each individual token.
    """
    pid_b = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    d_offs = tl.arange(0, BLOCK_D)
    kb_offs = tl.arange(0, BLOCK_K) # Block indices
    
    if pid_q < seq_q:
        # We loop over blocks instead of individual tokens
        out_row_ptr = OUT_ptr + pid_b * stride_ob + pid_q * stride_oq
        
        for k_idx in range(0, num_blocks, BLOCK_K):
            k_block = k_idx + kb_offs
            k_mask = (k_block < num_blocks) & (d_offs[:, None] < d_idx)
            
            # Load K blocks
            k_ptrs = K_ptr + pid_b * stride_kb + k_block[None, :] * stride_kk + d_offs[:, None] * stride_kd
            # Shape is [BLOCK_D, BLOCK_K] instead of [BLOCK_K, BLOCK_D] for easier dot
            k_vals = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32) 
            
            acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
            
            for h in range(n_heads):
                # Load w and b
                w_ptr = W_ptr + pid_b * stride_wb + pid_q * stride_wq + h * stride_wh
                w_val = tl.load(w_ptr).to(tl.float32)
                w_sigmoid = tl.sigmoid(w_val)
                b = tl.load(B_ptr + h).to(tl.float32)
                
                # Load Q
                q_ptrs = Q_ptr + pid_b * stride_qb + pid_q * stride_qq + h * stride_qh + d_offs * stride_qd
                q_val = tl.load(q_ptrs, mask=d_offs < d_idx, other=0.0).to(tl.float32)
                
                # compute block scores
                scores = tl.sum(q_val[:, None] * k_vals, axis=0) * scale 
                gated = tl.sigmoid(scores + b)
                
                acc += w_sigmoid * gated
                
            # Causal mask - approximate at block level
            q_block_id = pid_q // block_size
            causal_mask = q_block_id >= k_block
            acc = tl.where(causal_mask, acc, float('-inf'))
                
            out_ptrs = out_row_ptr + k_block * stride_ok
            out_mask = k_block < num_blocks
            tl.store(out_ptrs, acc, mask=out_mask)


def block_sparse_gated_indexer(
    q: torch.Tensor,   # [batch, seq_q, n_heads, d_idx]
    k: torch.Tensor,   # [batch, seq_kv, d_idx]
    w: torch.Tensor,   # [batch, seq_q, n_heads]
    b: torch.Tensor,   # [n_heads]
    block_size: int = 64,
    scale: float = 1.0,
):
    """
    Computes importance scores for chunks of keys instead of individual tokens.
    Automatically mean-pools the Keys into blocks before scoring.
    
    Output shape: [batch, seq_q, num_blocks]
    """
    B, T_q, H, d_idx = q.shape
    _, T_kv, _ = k.shape
    
    # ── 1. Pool Keys into Blocks ── (Extremely fast, memory bound)
    num_blocks = (T_kv + block_size - 1) // block_size
    
    # Pad k if necessary
    k_padded = k
    pad_len = (block_size - (T_kv % block_size)) % block_size
    if pad_len > 0:
        k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
        
    k_blocks = k_padded.view(B, num_blocks, block_size, d_idx).mean(dim=2) # [B, num_blocks, d_idx]
    
    # ── 2. Run Block Indexer ──
    out = torch.empty(B, T_q, num_blocks, device=q.device, dtype=torch.float32)
    
    BLOCK_K = min(64, triton.next_power_of_2(num_blocks))
    BLOCK_D = triton.next_power_of_2(d_idx)
    grid = (B, T_q)
    
    _block_indexer_fwd_kernel[grid](
        q.contiguous(), k_blocks.contiguous(), w.contiguous(), b.contiguous(), out,
        B, T_q, T_kv, H, d_idx,
        num_blocks, block_size,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_blocks.stride(0), k_blocks.stride(1), k_blocks.stride(2),
        w.stride(0), w.stride(1), w.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        scale,
        BLOCK_K=BLOCK_K, BLOCK_D=BLOCK_D,
    )
    
    return out
