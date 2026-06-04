"""
Fused Sigmoid Gating kernels.

Pattern A: x * sigmoid(gate) — replaces 2 CUDA kernels with 1
Pattern B: scale * sigmoid(x) — fuses scalar multiply with sigmoid
Both with backward passes.

Called 7+ times per forward pass.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_fwd_kernel(
    X_ptr,
    Gate_ptr,
    Out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(Out_ptr + offset, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _sigmoid_gate_bwd_kernel(
    Grad_ptr,
    X_ptr,
    Gate_ptr,
    dX_ptr,
    dGate_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    grad = tl.load(Grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(Gate_ptr + offset, mask=mask, other=0.0).to(tl.float32)

    sig_g = tl.sigmoid(g)
    dx = grad * sig_g
    dg = grad * x * sig_g * (1.0 - sig_g)

    tl.store(dX_ptr + offset, dx.to(dX_ptr.dtype.element_ty), mask=mask)
    tl.store(dGate_ptr + offset, dg.to(dGate_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _scaled_sigmoid_fwd_kernel(
    X_ptr,
    Out_ptr,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    out = scale * tl.sigmoid(x)
    tl.store(Out_ptr + offset, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _scaled_sigmoid_bwd_kernel(
    Grad_ptr,
    X_ptr,
    dX_ptr,
    N,
    scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N

    grad = tl.load(Grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    sig_x = tl.sigmoid(x)
    dx = grad * scale * sig_x * (1.0 - sig_x)
    tl.store(dX_ptr + offset, dx.to(dX_ptr.dtype.element_ty), mask=mask)


BLOCK = 1024


class FusedSigmoidGateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gate):
        assert x.is_contiguous() and gate.is_contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        grid = (triton.cdiv(N, BLOCK),)
        _sigmoid_gate_fwd_kernel[grid](x, gate, out, N, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x, gate)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, gate = ctx.saved_tensors
        dx = torch.empty_like(x)
        dg = torch.empty_like(gate)
        N = x.numel()
        grad_output = grad_output.contiguous()
        grid = (triton.cdiv(N, BLOCK),)
        _sigmoid_gate_bwd_kernel[grid](
            grad_output, x, gate, dx, dg, N, BLOCK=BLOCK, num_warps=4
        )
        return dx, dg


class FusedScaledSigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale=2.0):
        assert x.is_contiguous()
        out = torch.empty_like(x)
        N = x.numel()
        grid = (triton.cdiv(N, BLOCK),)
        _scaled_sigmoid_fwd_kernel[grid](x, out, N, scale, BLOCK=BLOCK, num_warps=4)
        ctx.save_for_backward(x)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        N = x.numel()
        grad_output = grad_output.contiguous()
        grid = (triton.cdiv(N, BLOCK),)
        _scaled_sigmoid_bwd_kernel[grid](
            grad_output, x, dx, N, ctx.scale, BLOCK=BLOCK, num_warps=4
        )
        return dx, None


def fused_sigmoid_gate(x, gate):
    """x * sigmoid(gate) in one kernel."""
    return FusedSigmoidGateFn.apply(x, gate)


def fused_scaled_sigmoid(x, scale=2.0):
    """scale * sigmoid(x) in one kernel."""
    return FusedScaledSigmoidFn.apply(x, scale)


# References
def pytorch_sigmoid_gate(x, gate):
    return x * torch.sigmoid(gate)


def pytorch_scaled_sigmoid(x, scale=2.0):
    return scale * torch.sigmoid(x)
