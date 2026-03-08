"""
Fused SwiGLU Autograd Function (Unsloth-style)
================================================

Fuses the shared expert SwiGLU: gate_proj(x) + up_proj(x) + silu_mul
into a single autograd function.

Standard PyTorch (3 separate autograd nodes):
    gate_h = gate_proj(x)         # saves x for backward
    up_h = up_proj(x)             # saves x for backward (DUPLICATE)
    h = silu(gate_h) * up_h       # saves gate_h, up_h for backward

Fused (1 autograd node):
    h = silu(gate_proj(x)) * up_proj(x)
    Saves: x, W_gate, W_up (parameter refs, no extra alloc)
    Does NOT save: gate_h or up_h (recomputed in backward)
    => eliminates 2 full [B*T, d_hidden] activation tensors from autograd graph

This is the same Unsloth philosophy applied to the MLP: trade recompute for memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .triton_silu_mul import HAS_TRITON
    if HAS_TRITON:
        from .triton_silu_mul import _silu_mul_fwd_kernel, _silu_mul_bwd_kernel
except ImportError:
    HAS_TRITON = False

try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


class _FusedSwiGLUFunc(torch.autograd.Function):
    """
    Fused SwiGLU: silu(x @ W_gate^T) * (x @ W_up^T)

    Forward: compute gate_h and up_h, apply silu_mul, return result.
    Backward: recompute gate_h and up_h from saved x and weights.
    Saves: x, W_gate, W_up (all refs — W's are frozen or shared, x is needed anyway).
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, W_gate, W_up):
        gate_h = F.linear(x, W_gate)
        up_h = F.linear(x, W_up)

        if HAS_TRITON and gate_h.is_cuda:
            h = torch.empty_like(gate_h)
            n = gate_h.numel()
            BLOCK = 1024
            _silu_mul_fwd_kernel[(n + BLOCK - 1) // BLOCK,](
                h, gate_h.reshape(-1), up_h.reshape(-1), n, BLOCK_SIZE=BLOCK,
            )
        else:
            h = F.silu(gate_h) * up_h

        ctx.save_for_backward(x, W_gate, W_up)
        return h

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_h):
        x, W_gate, W_up = ctx.saved_tensors
        orig_shape = x.shape

        x_2d = x.reshape(-1, x.shape[-1])
        go_2d = grad_h.reshape(-1, grad_h.shape[-1])

        gate_h = x_2d @ W_gate.t()
        up_h = x_2d @ W_up.t()

        if HAS_TRITON and gate_h.is_cuda:
            # Single Triton kernel: zero Python temporaries for d_gate_h / d_up_h
            n = gate_h.numel()
            d_gate_h = torch.empty_like(gate_h)
            d_up_h = torch.empty_like(up_h)
            BLOCK = 1024
            _silu_mul_bwd_kernel[(n + BLOCK - 1) // BLOCK,](
                d_gate_h.reshape(-1), d_up_h.reshape(-1),
                go_2d.contiguous().reshape(-1),
                gate_h.reshape(-1), up_h.reshape(-1),
                n, BLOCK_SIZE=BLOCK,
            )
            del gate_h, up_h
        else:
            sigma = torch.sigmoid(gate_h)
            silu_gate = gate_h * sigma
            d_up_h = silu_gate * go_2d
            d_gate_h = sigma * (1.0 + gate_h * (1.0 - sigma)) * up_h * go_2d
            del gate_h, up_h, sigma, silu_gate

        grad_W_gate = d_gate_h.t() @ x_2d
        grad_W_up = d_up_h.t() @ x_2d

        grad_x_2d = d_gate_h @ W_gate
        grad_x_2d.addmm_(d_up_h, W_up)
        del d_gate_h, d_up_h

        return grad_x_2d.reshape(orig_shape), grad_W_gate, grad_W_up


class FusedSwiGLUForward(nn.Module):
    """
    Drop-in replacement for the shared expert gate+up+silu_mul pattern.

    Usage:
        # Before:
        shared_h = liger_silu_mul(self.shared_gate(x), self.shared_up(x))

        # After:
        self.fused_swiglu = FusedSwiGLUForward(self.shared_gate, self.shared_up)
        shared_h = self.fused_swiglu(x)
    """

    def __init__(self, gate_proj: nn.Linear, up_proj: nn.Linear):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return _FusedSwiGLUFunc.apply(x, self.gate_proj.weight, self.up_proj.weight)
        gate_h = self.gate_proj(x)
        up_h = self.up_proj(x)
        return F.silu(gate_h) * up_h
