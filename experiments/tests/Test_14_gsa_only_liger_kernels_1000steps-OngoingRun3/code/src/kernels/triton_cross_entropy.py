"""
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
import torch
import triton
import triton.language as tl


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

            w = tl.load(
                W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
                mask=v_mask[:, None] & h_mask[None, :],
                other=0.0
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

        w = tl.load(
            W_ptr + v_ids[:, None] * stride_wv + h_ids[None, :] * stride_wh,
            mask=v_mask[:, None] & h_mask[None, :],
            other=0.0
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

class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, hidden_states, weight, target):
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            target = target.reshape(-1)
        return _UltimateFusedLinearCE_V33.apply(hidden_states, weight, target, self.ignore_index, self.reduction)
