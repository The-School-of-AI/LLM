"""
Fused MoE expert block: gate + up + silu_mul + down in one (or few) kernel(s).

One launch per (expert, block of M rows). Recomputes gate/up/h in backward
(Unsloth-style) to avoid storing activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from .moe_grouped_gemm import moe_grouped_gemm
    from .triton_silu_mul import _silu_mul_fwd_kernel, _silu_mul_bwd_kernel
except ImportError:
    moe_grouped_gemm = None
    _silu_mul_fwd_kernel = None
    _silu_mul_bwd_kernel = None

import torch.nn.functional as F


# Block sizes for the fused kernel (tiled matmuls).
# sm_80+ (A100/H100): BLOCK_M=32, BLOCK_D=64, BLOCK_H=64
# sm_75 and below (T4): BLOCK_M=16, BLOCK_D=32, BLOCK_H=32
# All launches use num_stages=1, num_warps=4.
BLOCK_M_DEFAULT = 32
BLOCK_D_DEFAULT = 64
BLOCK_H_DEFAULT = 64
BLOCK_M_SMALL = 16
BLOCK_D_SMALL = 32
BLOCK_H_SMALL = 32

_USE_SMALL_ENV = __import__("os").environ.get("TRITON_USE_SMALL_BLOCKS", "").strip().lower() in ("1", "true", "yes")
_block_cache_moe: dict[int, tuple[int, int, int]] = {}


def _get_block_sizes_moe(device: torch.device) -> tuple[int, int, int]:
    """Pick block sizes based on GPU compute capability."""
    if _USE_SMALL_ENV:
        return BLOCK_M_SMALL, BLOCK_D_SMALL, BLOCK_H_SMALL
    if device.type != "cuda":
        return BLOCK_M_DEFAULT, BLOCK_D_DEFAULT, BLOCK_H_DEFAULT
    idx = device.index if device.index is not None else 0
    if idx in _block_cache_moe:
        return _block_cache_moe[idx]
    try:
        major, _ = torch.cuda.get_device_capability(idx)
        result = (BLOCK_M_DEFAULT, BLOCK_D_DEFAULT, BLOCK_H_DEFAULT) if major >= 8 else (BLOCK_M_SMALL, BLOCK_D_SMALL, BLOCK_H_SMALL)
    except Exception:
        result = (BLOCK_M_SMALL, BLOCK_D_SMALL, BLOCK_H_SMALL)
    _block_cache_moe[idx] = result
    return result


if HAS_TRITON:
    @triton.jit
    def _fused_moe_expert_fwd_kernel(
        x_ptr,
        out_ptr,
        W_gate_ptr,
        W_up_ptr,
        W_down_ptr,
        row_offsets_ptr,
        block_expert_ptr,
        block_row_ptr,
        m_sizes_ptr,
        D,
        H,
        stride_x_m,
        stride_w_gate_d,
        stride_w_gate_h,
        stride_w_up_d,
        stride_w_up_h,
        stride_w_down_h,
        stride_w_down_d,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """One program per (expert, M-block). Computes gate, up, silu_mul, down for one block of rows."""
        block_id = tl.program_id(0)
        expert = tl.load(block_expert_ptr + block_id)
        row_in_expert = tl.load(block_row_ptr + block_id)
        M_g = tl.load(m_sizes_ptr + expert)
        actual_M = tl.minimum(BLOCK_M, M_g - row_in_expert)
        if actual_M <= 0:
            return

        row_off = tl.load(row_offsets_ptr + expert)
        x_row_start = row_off + row_in_expert

        # W base for this expert: W_gate [D, H], W_up [D, H], W_down [H, D]
        w_gate_base = expert * D * H
        w_up_base = expert * D * H
        w_down_base = expert * H * D

        # Accumulate gate [actual_M, H] and up [actual_M, H] by tiling over D
        gate_offs_m = tl.arange(0, BLOCK_M)
        gate_offs_h = tl.arange(0, BLOCK_H)
        gate_mask_m = gate_offs_m < actual_M

        gate_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D

            for h_start in range(0, H, BLOCK_H):
                h_offs = h_start + tl.arange(0, BLOCK_H)
                h_mask = h_offs < H

                x_ptrs = x_ptr + (x_row_start + gate_offs_m[:, None]) * stride_x_m + (d_start + tl.arange(0, BLOCK_D))[None, :]
                x_mask = gate_mask_m[:, None] & d_mask[None, :]
                x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

                w_g_ptrs = W_gate_ptr + w_gate_base + d_offs[:, None] * stride_w_gate_d + h_offs[None, :] * stride_w_gate_h
                w_g_block = tl.load(w_g_ptrs, mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                gate_acc += tl.dot(x_block, w_g_block)

                w_u_ptrs = W_up_ptr + w_up_base + d_offs[:, None] * stride_w_up_d + h_offs[None, :] * stride_w_up_h
                w_u_block = tl.load(w_u_ptrs, mask=d_mask[:, None] & h_mask[None, :], other=0.0)
                up_acc += tl.dot(x_block, w_u_block)

        # silu(gate) * up -> h [actual_M, H]
        gate_silu = gate_acc * tl.sigmoid(gate_acc)
        h = gate_silu * up_acc

        # out [actual_M, D] = h @ W_down, tile over D and store each tile
        out_row_start = row_off + row_in_expert
        for d_start in range(0, D, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < D
            out_acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
            for h_start in range(0, H, BLOCK_H):
                h_offs = h_start + tl.arange(0, BLOCK_H)
                h_mask = h_offs < H
                w_d_ptrs = W_down_ptr + w_down_base + h_offs[:, None] * stride_w_down_h + d_offs[None, :] * stride_w_down_d
                w_d_block = tl.load(w_d_ptrs, mask=h_mask[:, None] & d_mask[None, :], other=0.0)
                out_acc += tl.dot(h, w_d_block)
            out_ptrs = out_ptr + (out_row_start + gate_offs_m[:, None]) * D + d_offs[None, :]
            tl.store(out_ptrs, out_acc.to(out_ptr.dtype.element_ty), mask=gate_mask_m[:, None] & d_mask[None, :])


class _FusedMoEExpertFunction(torch.autograd.Function):
    """
    Fused MoE expert: gate + up + silu_mul + down with recompute in backward.
    Uses either Triton fused kernel (when available) or 3x grouped_gemm + silu_mul.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, W_gate, W_up, W_down, expert_counts, use_triton):
        E = W_gate.size(0)
        D, H = W_gate.size(1), W_gate.size(2)
        device = x.device
        dtype = x.dtype
        x_in = x.contiguous().to(W_gate.dtype)

        if use_triton and HAS_TRITON and x.is_cuda:
            block_m, block_d, block_h = _get_block_sizes_moe(device)
            row_offsets = torch.zeros(E + 1, device=device, dtype=torch.int64)
            torch.cumsum(expert_counts.to(device), dim=0, out=row_offsets[1:])
            total_blocks = 0
            blocks_per_expert = []
            for g in range(E):
                n = (expert_counts[g].item() + block_m - 1) // block_m
                blocks_per_expert.append(n)
                total_blocks += n
            if total_blocks == 0:
                out = torch.empty_like(x_in)
            else:
                block_expert = []
                block_row = []
                for g in range(E):
                    for b in range(blocks_per_expert[g]):
                        block_expert.append(g)
                        block_row.append(b * block_m)
                block_expert_t = torch.tensor(block_expert, device=device, dtype=torch.int64)
                block_row_t = torch.tensor(block_row, device=device, dtype=torch.int64)
                m_sizes = expert_counts.to(device).contiguous()
                out = torch.empty_like(x_in)
                grid = (total_blocks,)
                _fused_moe_expert_fwd_kernel[grid](
                    x_in,
                    out,
                    W_gate,
                    W_up,
                    W_down,
                    row_offsets,
                    block_expert_t,
                    block_row_t,
                    m_sizes,
                    D,
                    H,
                    x_in.stride(0),
                    W_gate.stride(1),
                    W_gate.stride(2),
                    W_up.stride(1),
                    W_up.stride(2),
                    W_down.stride(1),
                    W_down.stride(2),
                    BLOCK_M=block_m,
                    BLOCK_D=block_d,
                    BLOCK_H=block_h,
                    num_stages=1,
                    num_warps=4,
                )
            ctx.use_triton = use_triton
            ctx.save_for_backward(x_in, W_gate, W_up, W_down)
            ctx.expert_counts = expert_counts
        else:
            gate_out = moe_grouped_gemm(x_in, W_gate, expert_counts)
            up_out = moe_grouped_gemm(x_in, W_up, expert_counts)
            h = F.silu(gate_out) * up_out
            out = moe_grouped_gemm(h, W_down, expert_counts)
            ctx.use_triton = False
            ctx.save_for_backward(x_in, W_gate, W_up, W_down)
            ctx.expert_counts = expert_counts

        return out.to(dtype)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        x_in, W_gate, W_up, W_down = ctx.saved_tensors
        expert_counts = ctx.expert_counts
        go = grad_out.contiguous().to(W_gate.dtype)

        # Recompute gate, up, h (no stored activations)
        gate_out = moe_grouped_gemm(x_in, W_gate, expert_counts)
        up_out = moe_grouped_gemm(x_in, W_up, expert_counts)
        h = F.silu(gate_out) * up_out

        # d_down: grad_out^T @ h -> need grouped gemm backward (h^T @ grad_out per expert)
        # Standard backward: d_L/d_W_down = h^T @ grad_out (per group)
        # grouped_gemm backward is not typically exposed; we use grouped gemm with transposed shapes.
        # out = h @ W_down  =>  d_W_down = h^T @ grad_out
        # grouped: h [sum(m), H], grad_out [sum(m), D] -> d_W_down [E, H, D]
        # So we need a "grouped gemm" that computes h^T @ go and scatters to experts.
        # Megatron/grouped_gemm often doesn't expose backward. So we do per-expert backward in a loop.
        E = W_gate.size(0)
        D, H = W_gate.size(1), W_gate.size(2)
        row_offsets = torch.zeros(E + 1, device=x_in.device, dtype=torch.int64)
        torch.cumsum(expert_counts.to(x_in.device), dim=0, out=row_offsets[1:])

        grad_W_gate = torch.zeros_like(W_gate)
        grad_W_up = torch.zeros_like(W_up)
        grad_W_down = torch.zeros_like(W_down)
        grad_x = torch.zeros_like(x_in)

        for g in range(E):
            start = row_offsets[g].item()
            end = row_offsets[g + 1].item()
            if start >= end:
                continue
            h_g = h[start:end]
            go_g = go[start:end]
            x_g = x_in[start:end]
            # d_L/d_W_down[g] = h_g^T @ go_g
            grad_W_down[g] = h_g.t() @ go_g
            # d_L/d_h = go_g @ W_down[g]^T
            dh_g = go_g @ W_down[g].t()
            # silu_mul backward: dh -> d_gate, d_up
            gate_g = gate_out[start:end]
            up_g = up_out[start:end]
            if _silu_mul_bwd_kernel is not None and dh_g.is_cuda:
                n = gate_g.numel()
                d_gate_g = torch.empty_like(gate_g)
                d_up_g = torch.empty_like(up_g)
                BLOCK = 1024
                _silu_mul_bwd_kernel[(n + BLOCK - 1) // BLOCK,](
                    d_gate_g.reshape(-1),
                    d_up_g.reshape(-1),
                    dh_g.contiguous().reshape(-1),
                    gate_g.reshape(-1),
                    up_g.reshape(-1),
                    n,
                    BLOCK_SIZE=BLOCK,
                )
            else:
                sigma = torch.sigmoid(gate_g)
                silu_gate = gate_g * sigma
                d_up_g = silu_gate * dh_g
                d_gate_g = (sigma * (1.0 + gate_g * (1.0 - sigma)) * up_g * dh_g)
            grad_W_gate[g] = x_g.t() @ d_gate_g
            grad_W_up[g] = x_g.t() @ d_up_g
            grad_x[start:end] = d_gate_g @ W_gate[g] + d_up_g @ W_up[g]

        return grad_x, grad_W_gate, grad_W_up, grad_W_down, None, None


def fused_moe_expert_forward(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    expert_counts: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Fused MoE expert block: gate(x) + up(x) + silu_mul + down.
    use_triton: use Triton fused kernel when available and on CUDA.
    """
    return _FusedMoEExpertFunction.apply(
        x, W_gate, W_up, W_down, expert_counts, use_triton
    )


def has_fused_moe_expert_triton() -> bool:
    """True if Triton fused MoE expert kernel is available."""
    return bool(HAS_TRITON)
