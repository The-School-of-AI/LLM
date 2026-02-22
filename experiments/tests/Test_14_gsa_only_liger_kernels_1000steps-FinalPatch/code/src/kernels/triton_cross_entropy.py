"""
triton_cross_entropy.py — Self-contained Liger cross-entropy kernel.

Ported from https://github.com/linkedin/Liger-Kernel (Apache-2.0 licence).
All liger_kernel package imports replaced with local equivalents so that
no `pip install liger-kernel` is required on the training server.

Key properties:
- One Triton program per token row  — never materialises [N, V] FP32 tensor.
- Online softmax (2 passes over V) — numerically identical to PyTorch CE.
- Gradient stored in-place in the input buffer (further saves memory).
- Supports BF16/FP16/FP32 input.  ignore_index=-100 by default.
- `reduction='mean'` divides by the number of non-ignored tokens (matches
  PyTorch CrossEntropyLoss default).

Usage:
    from src.kernels.triton_cross_entropy import LigerCrossEntropyLoss
    ce = LigerCrossEntropyLoss(ignore_index=-100, reduction='mean')
    loss = ce(logits.view(-1, vocab_size), targets.view(-1))
"""

from typing import Optional

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Platform helpers (inlined so we have zero external imports)
# ---------------------------------------------------------------------------


def _is_hip() -> bool:
    return torch.version.hip is not None


# MAX_FUSED_SIZE: largest BLOCK_SIZE the Triton kernel will use for the
# vocab dimension.  32 768 covers vocab ≤ 32 768; for larger vocabs the
# kernel loops in strides of BLOCK_SIZE.
_MAX_FUSED_SIZE = 32768


# ---------------------------------------------------------------------------
# Triton utility kernel: element-wise multiply (used in backward)
# ---------------------------------------------------------------------------

@triton.jit
def _element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    X_ptr += program_id * X_stride
    grad_output = tl.load(grad_output_ptr)
    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


# ---------------------------------------------------------------------------
# Main fused CE kernel
# Adapted from:
#   https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py
# Apache-2.0 licence.  Modifications: removed XPU/NPU/HIP special-cases,
# removed z_loss, label-smoothing, weight, softcap, token-accuracy branches
# (all features we don't use) to keep the kernel compact.
# ---------------------------------------------------------------------------

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
    reduction: tl.constexpr,   # "mean" or "sum"
    BLOCK_SIZE: tl.constexpr,
    HAS_GRADIENTS: tl.constexpr,
):
    """
    Fused online-softmax cross-entropy.

    Grid: (N,) where N = number of token rows (B * T).
    Each program handles exactly one row.
    Gradient is stored back into X_ptr in-place.
    """
    program_id = tl.program_id(0).to(tl.int64)

    # Load target; early-exit for ignored positions
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

    # ── Pass 1: find max and log-sum-exp ────────────────────────────────────
    m = float("-inf")
    d = 0.0
    ori_X_y = tl.load(X_ptr + y).to(tl.float32)   # score at true class

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(
            X_ptr + X_offsets,
            mask=X_offsets < n_cols,
            other=float("-inf"),
        ).to(tl.float32)
        block_max = tl.max(X_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    lse = m + tl.log(d)          # log Σ exp(x_i)

    # ── Pass 2: write gradients in-place ────────────────────────────────────
    if HAS_GRADIENTS:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(
                X_ptr + X_offsets,
                mask=X_offsets < n_cols,
                other=float("-inf"),
            ).to(tl.float32)
            # gradient = softmax(x_i) for i != y; softmax(x_y) - 1 for i == y
            grad = tl.exp(X_block - m) / d
            grad = tl.where(X_offsets == y, grad - 1.0, grad)
            if reduction == "mean":
                grad = grad / n_non_ignore
            tl.store(X_ptr + X_offsets, grad, mask=X_offsets < n_cols)

    tl.debug_barrier()   # ensure grad writes are visible before loss store

    # ── Loss = lse - x_y ────────────────────────────────────────────────────
    loss = lse - ori_X_y
    if reduction == "mean":
        loss = loss / n_non_ignore

    tl.store(loss_ptr, loss)


# ---------------------------------------------------------------------------
# Python-side dispatch (forward + backward helpers)
# ---------------------------------------------------------------------------

def _cross_entropy_forward(_input, target, ignore_index, reduction):
    """
    Args:
        _input:  [N, V]  fp16/bf16/fp32, must be contiguous or will be made so.
        target:  [N]     int64
        ignore_index: int (default -100)
        reduction: "mean" | "sum"
    Returns:
        (scalar_loss, _input_with_grad_stored)
    """
    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    N, V = _input.shape

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(N, dtype=_input.dtype, device=_input.device)

    n_non_ignore = (target != ignore_index).sum().item()
    n_non_ignore = max(n_non_ignore, 1)   # avoid div-by-zero

    _liger_cross_entropy_kernel[(N,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_GRADIENTS=_input.requires_grad,
        num_warps=32 if not _is_hip() else 16,
    )

    loss = loss_1d.sum()    # each row already divided by n_non_ignore for 'mean'
    return loss, _input


def _cross_entropy_backward(_input, grad_output):
    """Scale in-place gradients that were written into _input during forward."""
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        return _input           # grad = 1 → no scaling needed

    if grad_output.ndim > 0:
        # reduction='none' path
        _input = _input * grad_output.unsqueeze(1)
    else:
        # scalar grad (mean/sum reduction)
        N, V = _input.shape
        BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))
        _element_mul_kernel[(N,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32 if not _is_hip() else 16,
        )
    return _input


# ---------------------------------------------------------------------------
# torch.autograd.Function
# ---------------------------------------------------------------------------

class _LigerCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, target, ignore_index, reduction):
        input_requires_grad = _input.requires_grad
        loss, _input_with_grad = _cross_entropy_forward(
            _input, target, ignore_index, reduction
        )
        if input_requires_grad:
            ctx.save_for_backward(_input_with_grad.detach())
        ctx.reduction = reduction
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (_input,) = ctx.saved_tensors
        _input = _cross_entropy_backward(_input, grad_output)
        return _input, None, None, None


# ---------------------------------------------------------------------------
# Drop-in replacement for torch.nn.CrossEntropyLoss / liger_kernel LigerCrossEntropyLoss
# ---------------------------------------------------------------------------

class LigerCrossEntropyLoss(torch.nn.Module):
    """
    Drop-in replacement for torch.nn.CrossEntropyLoss backed by the Liger
    fused Triton kernel.

    - Never materialises a full [N, V] FP32 probability tensor.
    - Gradient is accumulated in-place inside the kernel (2× memory saving
      vs PyTorch CE on BF16 inputs).
    - Numerically equivalent to PyTorch CE (online softmax, same formula).

    Args:
        ignore_index (int): Positions where target == ignore_index contribute
            zero loss and zero gradient.  Default: -100.
        reduction (str): "mean" or "sum".  Default: "mean".
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, _input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            _input:  [N, V]  (logits, any float dtype).
            target:  [N]     (token ids, int64).
        Returns:
            scalar loss.
        """
        return _LigerCEFunction.apply(_input, target, self.ignore_index, self.reduction)


# ---------------------------------------------------------------------------
# FusedLinearCrossEntropyLoss
# Fuses lm_head matmul + CE loss into one chunked kernel.
# Never materialises [B*T, vocab] logits.
#
# Ported/adapted from:
#   https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py
# Apache-2.0 licence.
#
# Algorithm:
#   1. Auto-size chunk_size so peak [C, vocab] ≤ 512 MB.
#   2. For each chunk:
#       a. logits_chunk = h_chunk @ W.T         (stays on GPU, [C, V])
#       b. Call CE Triton kernel (writes dlogits into logits_chunk in-place)
#       c. grad_h_chunk   += dlogits @ W        (accumulate grad wrt hidden)
#       d. grad_W         += h_chunk.T @ dlogits (accumulate grad wrt weight)
#       e. Discard logits_chunk                  (only [H, V] and [C, H] survives)
# ---------------------------------------------------------------------------

def _fused_linear_ce_forward(_input, weight, target, ignore_index, reduction):
    """
    Args:
        _input : [BT, H]  hidden states (bf16/fp16/fp32), contiguous
        weight : [V, H]   lm_head weight (same dtype)
        target : [BT]     token ids (int64)
    Returns:
        (scalar_loss, grad_input [BT, H], grad_weight [V, H])
        grad_input / grad_weight are None when requires_grad is False.
    """
    _input  = _input.contiguous()
    weight  = weight.contiguous()
    target  = target.contiguous()

    BT, H = _input.shape
    V     = weight.shape[0]

    # Auto chunk size: keep [chunk_size, V] float32 ≤ 512 MB
    max_chunk_bytes = 512 * 1024 * 1024
    elem_size = 4  # we accumulate in fp32
    max_elems  = max_chunk_bytes // (V * elem_size)
    chunk_size = max(1, min(BT, int(max_elems)))
    # Round to next power-of-2 for alignment, but don't exceed BT
    _cs = 1
    while _cs * 2 <= chunk_size:
        _cs *= 2
    chunk_size = min(_cs, BT)

    BLOCK_SIZE = min(_MAX_FUSED_SIZE, triton.next_power_of_2(V))

    n_non_ignore = max(int((target != ignore_index).sum().item()), 1)

    input_requires_grad  = _input.requires_grad
    weight_requires_grad = weight.requires_grad

    grad_input  = torch.zeros_like(_input)  if input_requires_grad  else None
    grad_weight = torch.zeros_like(weight)  if weight_requires_grad else None

    loss_accum = torch.zeros(1, device=_input.device, dtype=torch.float32)

    for start in range(0, BT, chunk_size):
        end = min(start + chunk_size, BT)
        C   = end - start

        h_chunk = _input[start:end]              # [C, H]
        t_chunk = target[start:end]              # [C]

        # Matmul in input dtype, then cast to float32 for the CE kernel
        logits_chunk = (h_chunk.float() @ weight.float().T).contiguous()  # [C, V] fp32

        t_chunk = t_chunk.contiguous()

        loss_1d = torch.zeros(C, dtype=torch.float32, device=_input.device)

        _liger_cross_entropy_kernel[(C,)](
            X_ptr      = logits_chunk,
            X_stride   = logits_chunk.stride(-2),
            Y_ptr      = t_chunk,
            Y_stride   = t_chunk.stride(-1),
            loss_ptr   = loss_1d,
            loss_stride= loss_1d.stride(-1),
            n_cols     = V,
            n_non_ignore = n_non_ignore,
            ignore_index = ignore_index,
            reduction  = reduction,
            BLOCK_SIZE = BLOCK_SIZE,
            HAS_GRADIENTS = (input_requires_grad or weight_requires_grad),
            num_warps  = 32 if not _is_hip() else 16,
        )

        loss_accum += loss_1d.sum()

        if input_requires_grad or weight_requires_grad:
            # logits_chunk now holds dL/dlogits (written in-place by CE kernel)
            dlogits = logits_chunk  # [C, V]
            if input_requires_grad:
                # dL/dh_chunk = dlogits @ W
                grad_input[start:end].add_(
                    (dlogits @ weight.float()).to(_input.dtype)
                )
            if weight_requires_grad:
                # dL/dW += h_chunk.T @ dlogits
                grad_weight.add_(
                    (h_chunk.float().T @ dlogits).to(weight.dtype)
                )

        del logits_chunk, loss_1d

    return loss_accum.squeeze(), grad_input, grad_weight


class _FusedLinearCEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, weight, target, ignore_index, reduction):
        loss, grad_input, grad_weight = _fused_linear_ce_forward(
            _input, weight, target, ignore_index, reduction
        )
        # Save grads computed during forward (Liger's write-in-place trick)
        ctx.save_for_backward(grad_input, grad_weight)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight = ctx.saved_tensors

        # Scale by upstream gradient (usually 1.0 if loss is the final scalar)
        if not torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            if grad_input  is not None: grad_input  = grad_input  * grad_output
            if grad_weight is not None: grad_weight = grad_weight * grad_output

        return grad_input, grad_weight, None, None, None


class FusedLinearCrossEntropyLoss(torch.nn.Module):
    """
    Fuses lm_head linear projection with cross-entropy loss.
    Never materialises the [B*T, vocab] logit tensor.

    Saves ~17 GB per step at B=4, T=4094, vocab=131075:
      - logits_ntp BF16 (4.3 GB) — never created
      - logits_mtp BF16 (4.3 GB) — never created
      - FP32 CE cast (8.6 GB)     — never created

    Zero fallback: if this kernel is unavailable, fail loudly.

    Args:
        ignore_index (int): Default: -100.
        reduction (str): "mean" | "sum". Default: "mean".

    Usage:
        fused_ce = FusedLinearCrossEntropyLoss()
        # Pass hidden states (NOT logits) + weight matrix:
        loss = fused_ce(h_ntp.view(-1, H), lm_head_weight, y_ntp.view(-1))
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(
        self,
        hidden_states: torch.Tensor,   # [N, H]
        weight:        torch.Tensor,   # [V, H]  (lm_head.weight)
        target:        torch.Tensor,   # [N]
    ) -> torch.Tensor:
        return _FusedLinearCEFunction.apply(
            hidden_states, weight, target, self.ignore_index, self.reduction
        )
