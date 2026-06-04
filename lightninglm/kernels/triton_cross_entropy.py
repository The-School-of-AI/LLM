"""
triton_cross_entropy.py — Fused Linear + CE with in-kernel softcap

Optimized: bf16 logits (no float32 conversion), pre-allocated buffer,
larger chunks to reduce iteration overhead.
"""

import torch
import triton
import triton.language as tl

_MAX_FUSED_SIZE = 8192


@triton.jit
def _tanh(x):
    """Numerically stable tanh: 1 - 2/(exp(2x)+1). Works for all float32."""
    return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)


@triton.jit
def _liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    softcap,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)
    if y == ignore_index:
        if HAS_GRADIENTS:
            X_ptr += program_id * X_stride
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return
    X_ptr += program_id * X_stride
    loss_ptr += program_id * loss_stride
    m = float("-inf")
    d = 0.0

    ori_X_y = tl.load(X_ptr + y).to(tl.float32)
    if HAS_SOFTCAP:
        ori_X_y = softcap * _tanh(ori_X_y / softcap)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = X_offsets < n_cols
        X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).to(
            tl.float32
        )

        if HAS_SOFTCAP:
            X_block = softcap * _tanh(X_block / softcap)

        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)

    if HAS_GRADIENTS:
        inv_d = 1.0 / d
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = X_offsets < n_cols
            X_block = tl.load(X_ptr + X_offsets, mask=mask, other=float("-inf")).to(
                tl.float32
            )

            if HAS_SOFTCAP:
                tanh_val = _tanh(X_block / softcap)
                sc_block = softcap * tanh_val
                chain_factor = 1.0 - tanh_val * tanh_val
            else:
                sc_block = X_block

            grad = tl.exp(sc_block - m) * inv_d
            grad = tl.where(X_offsets == y, grad - 1.0, grad)

            if HAS_SOFTCAP:
                grad = grad * chain_factor

            if reduction == "mean":
                grad = grad / n_non_ignore

            tl.store(X_ptr + X_offsets, grad, mask=mask)

    loss = lse - ori_X_y
    if reduction == "mean":
        loss = loss / n_non_ignore
    tl.store(loss_ptr, loss)


def _fused_linear_ce_forward(
    _input, weight, target, ignore_index, reduction, max_chunk_bytes, softcap=0.0
):
    BT, H = _input.shape
    V = weight.shape[0]

    # With bf16 logits (instead of float32), each row uses half the memory
    # so the same byte budget supports 2x more rows per chunk
    elem_size = _input.element_size()  # 2 for bf16 vs 4 for float32
    max_elems = max_chunk_bytes // (V * elem_size)
    chunk_size = max(1, min(BT, int(max_elems)))

    _cs = 1
    while _cs * 2 <= chunk_size:
        _cs *= 2
    chunk_size = min(_cs, BT)

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))
    n_non_ignore = max(int((target != ignore_index).sum().item()), 1)

    grad_input = torch.zeros_like(_input) if _input.requires_grad else None
    grad_weight = torch.zeros_like(weight) if weight.requires_grad else None
    loss_accum = torch.zeros(1, device=_input.device, dtype=torch.float32)

    weight_T = weight.t()  # View, no copy (cuBLAS handles transposed B natively)
    need_grad = grad_input is not None or grad_weight is not None
    has_softcap = softcap > 0

    # Pre-allocate reusable buffers
    logits_buf = torch.empty(chunk_size, V, device=_input.device, dtype=_input.dtype)
    loss_buf = torch.empty(chunk_size, dtype=torch.float32, device=_input.device)

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        C = end - start

        h_chunk = _input[start:end]
        t_chunk = target[start:end]

        # bf16 matmul directly into pre-allocated buffer
        logits_chunk = logits_buf[:C]
        torch.mm(h_chunk, weight_T, out=logits_chunk)

        loss_1d = loss_buf[:C].zero_()

        _liger_cross_entropy_kernel[(C,)](
            logits_chunk,
            logits_chunk.stride(-2),
            t_chunk,
            t_chunk.stride(-1),
            loss_1d,
            loss_1d.stride(-1),
            V,
            n_non_ignore,
            ignore_index,
            softcap,
            reduction,
            BLOCK_SIZE,
            need_grad,
            has_softcap,
        )

        loss_accum += loss_1d.sum()

        # logits_chunk now contains gradients (written by CE kernel)
        # Already in bf16 - no dtype conversion needed
        if grad_input is not None:
            grad_input[start:end].add_(logits_chunk @ weight)
        if grad_weight is not None:
            grad_weight.add_(logits_chunk.t() @ h_chunk)

    return loss_accum.squeeze(), grad_input, grad_weight


class _FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, _input, weight, target, ignore_index, reduction, max_chunk_gb, softcap
    ):
        max_chunk_bytes = int(max_chunk_gb * 1024 * 1024 * 1024)
        loss, grad_input, grad_weight = _fused_linear_ce_forward(
            _input, weight, target, ignore_index, reduction, max_chunk_bytes, softcap
        )
        ctx.save_for_backward(grad_input, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            if grad_input is not None:
                grad_input *= grad_output
            if grad_weight is not None:
                grad_weight *= grad_output
        return grad_input, grad_weight, None, None, None, None, None


class FusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self, ignore_index=-100, reduction="mean", max_chunk_gb=8.0, softcap=0.0
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.max_chunk_gb = max_chunk_gb
        self.softcap = softcap

    def forward(self, hidden_states, weight, target):
        return _FusedLinearCEFunction.apply(
            hidden_states,
            weight,
            target,
            self.ignore_index,
            self.reduction,
            self.max_chunk_gb,
            self.softcap,
        )
