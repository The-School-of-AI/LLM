"""
Triton Fused SiLU*Mul Kernel — Forward AND Backward
=====================================================

Fuses SiLU(gate) * up into a single Triton kernel for forward,
and fuses the backward into a single kernel that recomputes SiLU(gate)
rather than storing it.

Standard PyTorch:
    silu_gate = F.silu(gate)      # read gate, write silu_gate
    result = silu_gate * up       # read silu_gate + up, write result
    => 3 intermediate tensors, ~6 memory passes

Fused:
    result = silu(gate) * up      # read gate + up, write result
    => 0 intermediate tensors, 3 memory passes (~2x bandwidth reduction)

Backward math:
    sigma = sigmoid(gate)
    silu_gate = gate * sigma
    d_up = silu_gate * grad_output
    d_gate = (sigma + gate * sigma * (1 - sigma)) * up * grad_output
           = (sigma * (1 + gate * (1 - sigma))) * up * grad_output

Attribution:
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel (Apache-2.0)
- Unsloth: https://github.com/unslothai/unsloth (Apache-2.0)
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

try:
    from ..profiler import kernel_region
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def kernel_region(name: str):
        yield


# ═══════════════════════════════════════════════════════════════════════
# Triton Forward Kernel
# ═══════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    @triton.jit
    def _silu_mul_fwd_kernel(
        OUT_ptr, GATE_ptr, UP_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate_raw = tl.load(GATE_ptr + offsets, mask=mask)
        up_raw = tl.load(UP_ptr + offsets, mask=mask)
        out_dtype = gate_raw.dtype

        gate = gate_raw.to(tl.float32)
        up = up_raw.to(tl.float32)

        sigma = tl.sigmoid(gate)
        result = gate * sigma * up

        tl.store(OUT_ptr + offsets, result.to(out_dtype), mask=mask)

    @triton.jit
    def _silu_mul_bwd_kernel(
        D_GATE_ptr, D_UP_ptr,
        GRAD_ptr, GATE_ptr, UP_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        gate_raw = tl.load(GATE_ptr + offsets, mask=mask)
        out_dtype = gate_raw.dtype

        grad = tl.load(GRAD_ptr + offsets, mask=mask).to(tl.float32)
        gate = gate_raw.to(tl.float32)
        up = tl.load(UP_ptr + offsets, mask=mask).to(tl.float32)

        sigma = tl.sigmoid(gate)
        silu_gate = gate * sigma

        d_up = silu_gate * grad
        d_gate = sigma * (1.0 + gate * (1.0 - sigma)) * up * grad

        tl.store(D_GATE_ptr + offsets, d_gate.to(out_dtype), mask=mask)
        tl.store(D_UP_ptr + offsets, d_up.to(out_dtype), mask=mask)


# ═══════════════════════════════════════════════════════════════════════
# Autograd Function
# ═══════════════════════════════════════════════════════════════════════

class _FusedSiLUMulFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, gate, up):
        assert gate.shape == up.shape, f"Shape mismatch: gate {gate.shape} vs up {up.shape}"
        assert gate.is_contiguous() and up.is_contiguous()

        out = torch.empty_like(gate)
        n = gate.numel()
        BLOCK = 1024

        with kernel_region("silu_mul_fwd"):
            _silu_mul_fwd_kernel[(n + BLOCK - 1) // BLOCK,](
                out, gate, up, n, BLOCK_SIZE=BLOCK,
            )

        ctx.save_for_backward(gate, up)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        d_gate = torch.empty_like(gate)
        d_up = torch.empty_like(up)
        n = gate.numel()
        BLOCK = 1024

        with kernel_region("silu_mul_bwd"):
            _silu_mul_bwd_kernel[(n + BLOCK - 1) // BLOCK,](
                d_gate, d_up, grad_output, gate, up, n, BLOCK_SIZE=BLOCK,
            )

        return d_gate, d_up


# ═══════════════════════════════════════════════════════════════════════
# Public API (with PyTorch fallback)
# ═══════════════════════════════════════════════════════════════════════

def _pytorch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SiLU(gate) * up — Triton on CUDA, PyTorch fallback otherwise."""
    if HAS_TRITON and gate.is_cuda and gate.is_contiguous() and up.is_contiguous():
        return _FusedSiLUMulFunction.apply(gate, up)
    return _pytorch_silu_mul(gate, up)
