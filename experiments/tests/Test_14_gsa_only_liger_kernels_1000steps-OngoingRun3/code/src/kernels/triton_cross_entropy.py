"""
<<<<<<< HEAD
triton_cross_entropy.py — Ultimate Production V3.3-prod (Fixed Grids)
====================================================================

This is the "True Ship It" version:
- Corrected Grid Logic: Grids now dynamically depend on autotune 'meta' blocks.
- Forward: 1D over B-blocks, streams vocab blocks, stable online LSE + ylog extraction.
- Backward: 2D grid over (B-block, V-block), computes logits tile ONCE (no H² trap).
- Runtime-B safe (no recompiles from varying token counts).
- Scale passed as a device tensor (no .item()).
- Safe fallback handles non-standard shapes via torch.autograd.grad.

Assumed production shapes: H=4096, V=131072.
"""

from __future__ import annotations
=======
triton_cross_entropy_turbo_v2.py
================================

Streaming fused Linear + Cross Entropy, optimized for large vocab (e.g. 131072)
without materializing [B, V] logits.

Forward (Pass 1): stream vocab blocks
  - compute per-row max m
  - compute per-row sumexp s (stable online update)
  - compute per-row target logit y

Backward (Pass 2): recompute vocab blocks
  - compute p = exp(logits - lse)
  - dlogits = (p - onehot(y)) / N
  - accumulate grad_x and grad_w (grad_w via atomic adds)

Notes:
- grad_w is accumulated in fp32, returned as fp16/bf16 to match weight dtype.
- This is "fastest-class" style. Final speed depends on tuning (BLOCK_V, BLOCK_H, warps, stages).
"""

from __future__ import annotations

import math
>>>>>>> b8a8ddd9c0670decd0b5f1959f6dddc829b8e7dd
import torch
import triton
import triton.language as tl


<<<<<<< HEAD
# -----------------------------------------------------------------------------
# Forward (1D grid over B blocks, streams vocab tiles)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 16, "BLOCK_V": 8192,  "BLOCK_H": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_B": 32, "BLOCK_V": 16384, "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_B": 16, "BLOCK_V": 16384, "BLOCK_H": 256}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_B": 32, "BLOCK_V": 8192,  "BLOCK_H": 256}, num_warps=8, num_stages=3),
    ],
    key=["V", "H"],
)
@triton.jit
def _fwd_stream_kernel(
    X_ptr, W_ptr, Y_ptr,
    LSE_ptr, YLOG_ptr,
    B,                       # runtime
    H: tl.constexpr,         # constexpr
    V: tl.constexpr,         # constexpr
    ignore_index: tl.constexpr,
    stride_xm: tl.constexpr, stride_xh: tl.constexpr,
    stride_wv: tl.constexpr, stride_wh: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    row0 = pid * BLOCK_B
    rows = row0 + tl.arange(0, BLOCK_B)
    row_mask = rows < B

    y = tl.load(Y_ptr + rows, mask=row_mask, other=ignore_index).to(tl.int32)
    active = row_mask & (y != ignore_index)

    m = tl.full((BLOCK_B,), -float("inf"), tl.float32)
    s = tl.zeros((BLOCK_B,), tl.float32)
    ylog = tl.zeros((BLOCK_B,), tl.float32)

    for v0 in tl.static_range(0, V, BLOCK_V):
        v_ids = v0 + tl.arange(0, BLOCK_V)
        v_mask = v_ids < V

        logits = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)

        for h0 in tl.static_range(0, H, BLOCK_H):
            h_ids = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_ids < H

            x = tl.load(
                X_ptr + rows[:, None] * stride_xm + h_ids[None, :] * stride_xh,
                mask=active[:, None] & h_mask[None, :],
                other=0.0
            ).to(tl.float16)

=======
# -------------------------
# Utilities
# -------------------------

def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 2 ** (int(x - 1).bit_length())


# -------------------------
# Pass 1: Streaming LSE + y_logit (no logits materialization)
# One program = one row.
# Streams vocab in blocks of BLOCK_V.
# Computes logits in tiles of BLOCK_H.
# -------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 8192,  "BLOCK_H": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 16384, "BLOCK_H": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192,  "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 16384, "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 4096,  "BLOCK_H": 128}, num_warps=4, num_stages=4),
    ],
    key=["H", "V"],
)
@triton.jit
def _fwd_lse_y_kernel(
    X_ptr, W_ptr, Y_ptr,
    M_ptr, S_ptr, YLOG_ptr,
    H: tl.constexpr, V: tl.constexpr,
    ignore_index: tl.constexpr,
    stride_xm: tl.constexpr, stride_xh: tl.constexpr,
    stride_wh: tl.constexpr, stride_wv: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int32)
    y = tl.load(Y_ptr + row).to(tl.int32)
    ignore = y == ignore_index

    # Load x[row, :]
    # We'll reuse x chunks in dot products.
    # Shape: [H]
    # For dot, we stream H in BLOCK_H.
    # (Loading x repeatedly is okay; it stays hot in L2/SMEM depending on cache.)
    # Initialize running stats
    m = tl.full((), -float("inf"), tl.float32)
    s = tl.full((), 0.0, tl.float32)
    ylog = tl.full((), 0.0, tl.float32)

    # For ignored rows, we store sentinel m=-1e4 and s=1 (so lse finite if needed),
    # and ylog=0, but loss masking is handled in Python.
    # We still do minimal work.
    if tl.constexpr(True):
        # Triton doesn't allow branching on runtime tensors safely.
        # We'll just mask all later updates.
        pass

    # Stream vocab blocks
    v0 = 0
    while v0 < V:
        v_ids = v0 + tl.arange(0, BLOCK_V)
        v_mask = v_ids < V

        # Compute logits for this vocab block: [BLOCK_V]
        # logits[v] = dot(x[row, :], W[v, :])
        # We compute in fp32 accumulation.
        acc = tl.zeros((BLOCK_V,), tl.float32)

        h0 = 0
        while h0 < H:
            h_ids = h0 + tl.arange(0, BLOCK_H)
            h_mask = h_ids < H

            x = tl.load(X_ptr + row * stride_xm + h_ids * stride_xh, mask=h_mask, other=0.0).to(tl.float32)
            # Load W block: W[v_ids, h_ids]
            # W layout assumed [V, H] contiguous-ish; strides passed in.
            # We load as [BLOCK_V, BLOCK_H] then dot with x.
>>>>>>> b8a8ddd9c0670decd0b5f1959f6dddc829b8e7dd
            w = tl.load(
                W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
                mask=v_mask[:, None] & h_mask[None, :],
                other=0.0
<<<<<<< HEAD
            ).to(tl.float16)

            logits += tl.dot(x, tl.trans(w)).to(tl.float32)

        block_max = tl.max(logits, axis=1)
        m_new = tl.maximum(m, block_max)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new[:, None]), axis=1)
        m = m_new

        in_range = active & (y >= v0) & (y < (v0 + BLOCK_V))
        is_tgt = in_range[:, None] & (v_ids[None, :] == y[:, None])
        y_val = tl.sum(tl.where(is_tgt, logits, 0.0), axis=1)
        ylog = tl.where(in_range, y_val, ylog)

    s = tl.maximum(s, 1e-20)
    lse = m + tl.log(s)
    lse = tl.where(active, lse, -1e4)
    ylog = tl.where(active, ylog, 0.0)

    tl.store(LSE_ptr + rows, lse, mask=row_mask)
    tl.store(YLOG_ptr + rows, ylog, mask=row_mask)


# -----------------------------------------------------------------------------
# Backward (2D grid over B x V)
# -----------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_B": 16, "BLOCK_V": 8192, "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_B": 16, "BLOCK_V": 4096, "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_B": 8,  "BLOCK_V": 8192, "BLOCK_H": 256}, num_warps=8, num_stages=3),
    ],
    key=["V", "H"],
)
@triton.jit
def _bwd_tile_kernel(
    X_ptr, W_ptr, Y_ptr, LSE_ptr,
    dX_ptr, dW_ptr,
    grad_scale_ptr,
    B,
    H: tl.constexpr,
    V: tl.constexpr,
    ignore_index: tl.constexpr,
    stride_xm: tl.constexpr, stride_xh: tl.constexpr,
    stride_wv: tl.constexpr, stride_wh: tl.constexpr,
    stride_dxm: tl.constexpr, stride_dxh: tl.constexpr,
    stride_dwv: tl.constexpr, stride_dwh: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_V: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0).to(tl.int64)
    pid_v = tl.program_id(1).to(tl.int64)

    row0 = pid_b * BLOCK_B
    v0 = pid_v * BLOCK_V

    rows = row0 + tl.arange(0, BLOCK_B)
    v_ids = v0 + tl.arange(0, BLOCK_V)

    row_mask = rows < B
    v_mask = v_ids < V

    y = tl.load(Y_ptr + rows, mask=row_mask, other=ignore_index).to(tl.int32)
    lse = tl.load(LSE_ptr + rows, mask=row_mask, other=-1e4).to(tl.float32)
    active = row_mask & (y != ignore_index)

    scale = tl.load(grad_scale_ptr).to(tl.float32)

    logits = tl.zeros((BLOCK_B, BLOCK_V), tl.float32)
    for h0 in tl.static_range(0, H, BLOCK_H):
        h_ids = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_ids < H

        x = tl.load(
            X_ptr + rows[:, None] * stride_xm + h_ids[None, :] * stride_xh,
            mask=active[:, None] & h_mask[None, :],
            other=0.0
        ).to(tl.float16)

=======
            ).to(tl.float32)

            # Dot: (BLOCK_V, BLOCK_H) @ (BLOCK_H)
            acc += tl.sum(w * x[None, :], axis=1)
            h0 += BLOCK_H

        # Stable online update for this block
        block_max = tl.max(acc, axis=0)
        m_new = tl.maximum(m, block_max)
        # s = s*exp(m-m_new) + sum(exp(acc - m_new))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(acc - m_new), axis=0)
        m = m_new

        # Extract target logit if y in [v0, v0+BLOCK_V)
        in_range = (y >= v0) & (y < (v0 + BLOCK_V)) & (~ignore)
        # compute index inside block
        y_idx = y - v0
        # gather acc[y_idx] safely
        y_val = tl.load(acc + y_idx, mask=in_range, other=0.0)
        ylog = tl.where(in_range, y_val, ylog)

        v0 += BLOCK_V

    # Apply ignore sentinel
    m_out = tl.where(ignore, -1e4, m)
    s_out = tl.where(ignore, 0.0, s)
    ylog_out = tl.where(ignore, 0.0, ylog)

    tl.store(M_ptr + row, m_out)
    tl.store(S_ptr + row, s_out)
    tl.store(YLOG_ptr + row, ylog_out)


# -------------------------
# Pass 2: Backward streaming
# - recompute logits per vocab block
# - p = exp(logits - lse)
# - d = (p - onehot) / N
# - grad_x += d @ W_block
# - grad_w += outer(d, x)  (atomic adds)
# One program = one (row, vocab_block)
# -------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_V": 4096,  "BLOCK_H": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 8192,  "BLOCK_H": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_V": 4096,  "BLOCK_H": 256}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_V": 8192,  "BLOCK_H": 256}, num_warps=8, num_stages=3),
    ],
    key=["H", "V"],
)
@triton.jit
def _bwd_stream_kernel(
    X_ptr, W_ptr, Y_ptr,
    LSE_ptr,
    dX_ptr, dW_ptr,   # dW is fp32
    n_non_ignore_f: tl.constexpr,
    H: tl.constexpr, V: tl.constexpr,
    ignore_index: tl.constexpr,
    stride_xm: tl.constexpr, stride_xh: tl.constexpr,
    stride_wh: tl.constexpr, stride_wv: tl.constexpr,
    stride_dxm: tl.constexpr, stride_dxh: tl.constexpr,
    stride_dwh: tl.constexpr, stride_dwv: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    # 2D grid folded into 1D: pid = row * nblocks + block_id
    nblocks = tl.cdiv(V, BLOCK_V)
    row = (pid // nblocks).to(tl.int32)
    b = (pid % nblocks).to(tl.int32)
    v0 = b * BLOCK_V

    y = tl.load(Y_ptr + row).to(tl.int32)
    ignore = y == ignore_index

    lse = tl.load(LSE_ptr + row).to(tl.float32)
    inv_n = 1.0 / n_non_ignore_f

    v_ids = v0 + tl.arange(0, BLOCK_V)
    v_mask = v_ids < V

    # Recompute logits for this (row, vocab_block): acc[BLOCK_V]
    acc = tl.zeros((BLOCK_V,), tl.float32)
    h0 = 0
    while h0 < H:
        h_ids = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_ids < H

        x = tl.load(X_ptr + row * stride_xm + h_ids * stride_xh, mask=h_mask, other=0.0).to(tl.float32)
>>>>>>> b8a8ddd9c0670decd0b5f1959f6dddc829b8e7dd
        w = tl.load(
            W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
            mask=v_mask[:, None] & h_mask[None, :],
            other=0.0
<<<<<<< HEAD
        ).to(tl.float16)

        logits += tl.dot(x, tl.trans(w)).to(tl.float32)

    p = tl.exp(logits - lse[:, None])
    is_tgt = active[:, None] & (v_ids[None, :] == y[:, None])
    dlogits = (p - tl.where(is_tgt, 1.0, 0.0)) * scale
    dlogits = tl.where(active[:, None], dlogits, 0.0)

    for h0 in tl.static_range(0, H, BLOCK_H):
        h_ids = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_ids < H

        xh = tl.load(
            X_ptr + rows[:, None] * stride_xm + h_ids[None, :] * stride_xh,
            mask=active[:, None] & h_mask[None, :],
            other=0.0
        ).to(tl.float16)

        wh = tl.load(
            W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
            mask=v_mask[:, None] & h_mask[None, :],
            other=0.0
        ).to(tl.float16)

        dW_tile = tl.dot(tl.trans(dlogits).to(tl.float16), xh).to(tl.float32)
        tl.atomic_add(
            dW_ptr + v_ids[:, None] * stride_dwv + h_ids[None, :] * stride_dwh,
            dW_tile,
            mask=v_mask[:, None] & h_mask[None, :]
        )

        dX_tile = tl.dot(dlogits.to(tl.float16), wh).to(tl.float32)
        tl.atomic_add(
            dX_ptr + rows[:, None] * stride_dxm + h_ids[None, :] * stride_dxh,
            dX_tile,
            mask=active[:, None] & h_mask[None, :]
        )


# -----------------------------------------------------------------------------
# Autograd wrapper
# -----------------------------------------------------------------------------

class _UltimateFusedLinearCE_V33(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, target, ignore_index, reduction):
        B, H = x.shape
        V, _ = weight.shape

        if H != 4096 or V != 131072:
            ctx.use_torch = True
            ctx.ignore_index, ctx.reduction = ignore_index, reduction
            ctx.save_for_backward(x, weight, target)
            logits = x @ weight.t()
            return torch.nn.functional.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)

        ctx.use_torch = False
        ctx.ignore_index, ctx.reduction = ignore_index, reduction
        lse = torch.empty((B,), device=x.device, dtype=torch.float32)
        ylog = torch.empty((B,), device=x.device, dtype=torch.float32)

        def grid_fwd(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]),)

        _fwd_stream_kernel[grid_fwd](
            x, weight, target, lse, ylog,
            B, H, V, ignore_index,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
        )

        active = (target != ignore_index)
        n_non_ignore = max(int(active.sum().item()), 1)
        loss_vec = (lse - ylog) * active
        loss = (loss_vec.sum() / n_non_ignore) if reduction == "mean" else loss_vec.sum()

        ctx.save_for_backward(x, weight, target, lse)
        ctx.n_non_ignore = n_non_ignore
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        if getattr(ctx, "use_torch", False):
            x, weight, target = ctx.saved_tensors
            logits = x @ weight.t()
            loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=ctx.ignore_index, reduction=ctx.reduction)
            dx, dw = torch.autograd.grad(loss, (x, weight), grad_outputs=grad_output)
            return dx, dw, None, None, None

        x, weight, target, lse = ctx.saved_tensors
        B, H = x.shape
        V, _ = weight.shape

        dX32 = torch.zeros((B, H), device=x.device, dtype=torch.float32)
        dW32 = torch.zeros((V, H), device=weight.device, dtype=torch.float32)

        grad_out = grad_output.to(torch.float32)
        scale = (grad_out / float(ctx.n_non_ignore)) if ctx.reduction == "mean" else grad_out
        scale_tensor = scale.reshape(1).contiguous()

        def grid_bwd(meta):
            return (triton.cdiv(B, meta["BLOCK_B"]), triton.cdiv(V, meta["BLOCK_V"]))

        _bwd_tile_kernel[grid_bwd](
            x, weight, target, lse, dX32, dW32, scale_tensor,
            B, H, V, ctx.ignore_index,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            dX32.stride(0), dX32.stride(1),
            dW32.stride(0), dW32.stride(1),
        )

        return dX32.to(x.dtype), dW32.to(weight.dtype), None, None, None
=======
        ).to(tl.float32)

        acc += tl.sum(w * x[None, :], axis=1)
        h0 += BLOCK_H

    # softmax prob
    p = tl.exp(acc - lse)
    # onehot subtraction
    is_tgt = (v_ids == y) & (~ignore)
    d = (p - tl.where(is_tgt, 1.0, 0.0)) * inv_n
    d = tl.where(ignore, 0.0, d)

    # grad_x[row, :] += d @ W_block
    # Compute in H tiles:
    h0 = 0
    while h0 < H:
        h_ids = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_ids < H

        # Load W_block[:, h_ids] and reduce over vocab
        w = tl.load(
            W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
            mask=v_mask[:, None] & h_mask[None, :],
            other=0.0
        ).to(tl.float32)

        gx = tl.sum(w * d[:, None], axis=0)  # [BLOCK_H]
        # atomic add into dX (fp32 or fp16? we'll write into fp32 buffer generally)
        tl.atomic_add(
            dX_ptr + row * stride_dxm + h_ids * stride_dxh,
            gx,
            mask=h_mask
        )

        h0 += BLOCK_H

    # grad_w[v_ids, :] += d[:,None] * x[None,:]
    # This is the heavy part; we do atomic adds in fp32.
    h0 = 0
    while h0 < H:
        h_ids = h0 + tl.arange(0, BLOCK_H)
        h_mask = h_ids < H

        x = tl.load(X_ptr + row * stride_xm + h_ids * stride_xh, mask=h_mask, other=0.0).to(tl.float32)
        # outer: (BLOCK_V, BLOCK_H)
        dw = d[:, None] * x[None, :]

        tl.atomic_add(
            dW_ptr + v_ids[:, None] * stride_dwv + h_ids[None, :] * stride_dwh,
            dw,
            mask=v_mask[:, None] & h_mask[None, :]
        )

        h0 += BLOCK_H


# -------------------------
# Autograd Function
# -------------------------

class _StreamingLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor,
                ignore_index: int = -100, reduction: str = "mean"):
        assert x.is_cuda and w.is_cuda and y.is_cuda
        assert x.ndim == 2 and w.ndim == 2 and y.ndim == 1
        B, H = x.shape
        V, H2 = w.shape
        assert H == H2
        assert y.shape[0] == B
        assert reduction in ("mean", "sum")

        # Pass 1 buffers
        m = torch.empty((B,), device=x.device, dtype=torch.float32)
        s = torch.empty((B,), device=x.device, dtype=torch.float32)
        ylog = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid = (B,)
        _fwd_lse_y_kernel[grid](
            x, w, y,
            m, s, ylog,
            H=H, V=V,
            ignore_index=ignore_index,
            stride_xm=x.stride(0), stride_xh=x.stride(1),
            stride_wh=w.stride(1), stride_wv=w.stride(0),
        )

        # lse and loss
        # s can be 0 for ignored rows; clamp for safety
        s_clamped = torch.clamp(s, min=1e-20)
        lse = m + torch.log(s_clamped)

        active = (y != ignore_index)
        n_non_ignore = int(active.sum().item())
        if n_non_ignore <= 0:
            n_non_ignore = 1

        loss_vec = (lse - ylog) * active
        if reduction == "mean":
            loss = loss_vec.sum() / n_non_ignore
        else:
            loss = loss_vec.sum()

        ctx.save_for_backward(x, w, y, lse, active)
        ctx.ignore_index = ignore_index
        ctx.reduction = reduction
        ctx.n_non_ignore = float(n_non_ignore)
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w, y, lse, active = ctx.saved_tensors
        B, H = x.shape
        V, _ = w.shape

        # We accumulate dX and dW in fp32 for stability/perf, cast at the end.
        dX32 = torch.zeros((B, H), device=x.device, dtype=torch.float32)
        dW32 = torch.zeros((V, H), device=w.device, dtype=torch.float32)

        n_non_ignore_f = ctx.n_non_ignore

        # grid over (row, vocab_block)
        # program_id = row * nblocks + block
        nblocks = triton.cdiv(V, 8192)  # just for sizing; autotune will pick real BLOCK_V
        # safer: compute max blocks using largest BLOCK_V in configs = 16384 forward, 8192 backward configs
        # We'll compute using 4096 as worst-case upper bound.
        nblocks = triton.cdiv(V, 4096)
        grid = (B * nblocks,)

        _bwd_stream_kernel[grid](
            x, w, y,
            lse,
            dX32, dW32,
            n_non_ignore_f=n_non_ignore_f,
            H=H, V=V,
            ignore_index=ctx.ignore_index,
            stride_xm=x.stride(0), stride_xh=x.stride(1),
            stride_wh=w.stride(1), stride_wv=w.stride(0),
            stride_dxm=dX32.stride(0), stride_dxh=dX32.stride(1),
            stride_dwh=dW32.stride(1), stride_dwv=dW32.stride(0),
        )

        # Apply upstream grad (scalar) always
        # grad_out is typically 1.0; this is cheap and correct.
        dX32 *= grad_out.to(torch.float32)
        dW32 *= grad_out.to(torch.float32)

        # Cast to match input/weight dtypes
        dX = dX32.to(dtype=x.dtype)
        dW = dW32.to(dtype=w.dtype)

        return dX, dW, None, None, None

>>>>>>> b8a8ddd9c0670decd0b5f1959f6dddc829b8e7dd

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

<<<<<<< HEAD
    def forward(self, hidden_states, weight, target):
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            target = target.reshape(-1)
        return _UltimateFusedLinearCE_V33.apply(hidden_states, weight, target, self.ignore_index, self.reduction)
=======
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, target: torch.Tensor):
        return _StreamingLinearCE.apply(hidden_states, weight, target, self.ignore_index, self.reduction)
>>>>>>> b8a8ddd9c0670decd0b5f1959f6dddc829b8e7dd
