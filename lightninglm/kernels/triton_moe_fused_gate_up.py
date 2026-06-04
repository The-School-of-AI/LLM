"""
Fused MoE Gate+Up+SiLU expert kernel.

Replaces: 2 separate grouped GEMMs + liger_silu_mul (3 kernel launches)
With: 1 kernel that reads sorted_x ONCE and computes both projections + SiLU activation.

Called 1x per MoEFFN × 8 layers = 8 per forward pass (replaces 24 kernel launches).
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# Forward kernel: h = SiLU(x @ W_gate[e]) * (x @ W_up[e])
# Reads x once, computes dual GEMM + fused SiLU epilogue
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_H": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_H": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_H": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
    ],
    key=["K", "H"],
)
@triton.jit
def _fused_gate_up_silu_fwd_kernel(
    X_ptr,
    W_gate_ptr,
    W_up_ptr,
    Out_ptr,
    # For backward: save gate_out and up_out
    Gate_save_ptr,
    Up_save_ptr,
    Offsets_ptr,
    K: tl.constexpr,
    H: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_wg_e,
    stride_wg_k,
    stride_wg_h,
    stride_wu_e,
    stride_wu_k,
    stride_wu_h,
    stride_om,
    stride_oh,
    stride_gs_m,
    stride_gs_h,
    stride_us_m,
    stride_us_h,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)

    start = tl.load(Offsets_ptr + pid_e)
    end = tl.load(Offsets_ptr + pid_e + 1)
    M_e = end - start

    if pid_m * BLOCK_M >= M_e:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Base pointers
    x_base = X_ptr + start * stride_xm
    wg_base = W_gate_ptr + pid_e * stride_wg_e
    wu_base = W_up_ptr + pid_e * stride_wu_e

    # Dual accumulators
    acc_gate = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_a = (offs_m[:, None] < M_e) & (offs_k[None, :] < K)

        # Load X tile ONCE — reused for both gate and up GEMMs
        x_tile = tl.load(
            x_base + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_a,
            other=0.0,
        )

        mask_w = (offs_k[:, None] < K) & (offs_h[None, :] < H)

        # Load W_gate tile and accumulate
        wg_tile = tl.load(
            wg_base + offs_k[:, None] * stride_wg_k + offs_h[None, :] * stride_wg_h,
            mask=mask_w,
            other=0.0,
        )
        acc_gate += tl.dot(x_tile, wg_tile)

        # Load W_up tile and accumulate (x_tile reused!)
        wu_tile = tl.load(
            wu_base + offs_k[:, None] * stride_wu_k + offs_h[None, :] * stride_wu_h,
            mask=mask_w,
            other=0.0,
        )
        acc_up += tl.dot(x_tile, wu_tile)

    # Fused SiLU epilogue: h = SiLU(gate) * up = gate * sigmoid(gate) * up
    sig = tl.sigmoid(acc_gate)
    h = acc_gate * sig * acc_up

    # Store output
    mask_c = (offs_m[:, None] < M_e) & (offs_h[None, :] < H)
    out_base = Out_ptr + start * stride_om
    tl.store(
        out_base + offs_m[:, None] * stride_om + offs_h[None, :] * stride_oh,
        h.to(Out_ptr.dtype.element_ty),
        mask=mask_c,
    )

    # Save intermediates for backward (gate_out and up_out)
    gs_base = Gate_save_ptr + start * stride_gs_m
    us_base = Up_save_ptr + start * stride_us_m
    tl.store(
        gs_base + offs_m[:, None] * stride_gs_m + offs_h[None, :] * stride_gs_h,
        acc_gate.to(Gate_save_ptr.dtype.element_ty),
        mask=mask_c,
    )
    tl.store(
        us_base + offs_m[:, None] * stride_us_m + offs_h[None, :] * stride_us_h,
        acc_up.to(Up_save_ptr.dtype.element_ty),
        mask=mask_c,
    )


# ============================================================================
# Backward elementwise kernel: compute d_gate and d_up from dh
# ============================================================================


@triton.jit
def _fused_gate_up_silu_bwd_elem_kernel(
    Grad_ptr,
    Gate_ptr,
    Up_ptr,
    dGate_ptr,
    dUp_ptr,
    N_total,
    stride_g,
    stride_gate,
    stride_up,
    stride_dg,
    stride_du,
    BLOCK: tl.constexpr,
):
    """
    Given dh [M, H], gate_out [M, H], up_out [M, H]:
      d_up = dh * SiLU(gate_out)
      d_gate = dh * up_out * sigmoid(gate_out) * (1 + gate_out * (1 - sigmoid(gate_out)))
             = dh * up_out * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N_total

    dh = tl.load(Grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(Gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(Up_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu_gate = gate * sig

    d_up = dh * silu_gate
    # SiLU derivative: d/dx[x*sigmoid(x)] = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
    #                                      = sigmoid(x) * (1 + x*(1-sigmoid(x)))
    d_silu = dh * up
    d_gate = d_silu * sig * (1.0 + gate * (1.0 - sig))

    tl.store(dGate_ptr + offs, d_gate.to(dGate_ptr.dtype.element_ty), mask=mask)
    tl.store(dUp_ptr + offs, d_up.to(dUp_ptr.dtype.element_ty), mask=mask)


# ============================================================================
# Python wrappers
# ============================================================================


def _compute_offsets(expert_counts, device):
    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.to(device=device, dtype=torch.int64).contiguous()
    else:
        counts = torch.tensor(list(expert_counts), device=device, dtype=torch.int64)
    offsets = torch.zeros(counts.shape[0] + 1, device=device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, counts


class FusedMoEGateUpSiLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W_gate, W_up, expert_counts_tensor, offsets, max_M, E):
        M_total, K = x.shape
        H = W_gate.shape[2]

        out = torch.empty(M_total, H, device=x.device, dtype=x.dtype)
        gate_save = torch.empty(M_total, H, device=x.device, dtype=x.dtype)
        up_save = torch.empty(M_total, H, device=x.device, dtype=x.dtype)

        if M_total > 0:

            def grid(meta):
                mm = (max_M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
                hh = (H + meta["BLOCK_H"] - 1) // meta["BLOCK_H"]
                return (E, mm, hh)

            _fused_gate_up_silu_fwd_kernel[grid](
                x,
                W_gate,
                W_up,
                out,
                gate_save,
                up_save,
                offsets,
                K,
                H,
                x.stride(0),
                x.stride(1),
                W_gate.stride(0),
                W_gate.stride(1),
                W_gate.stride(2),
                W_up.stride(0),
                W_up.stride(1),
                W_up.stride(2),
                out.stride(0),
                out.stride(1),
                gate_save.stride(0),
                gate_save.stride(1),
                up_save.stride(0),
                up_save.stride(1),
            )

        ctx.save_for_backward(
            x, W_gate, W_up, gate_save, up_save, expert_counts_tensor, offsets
        )
        ctx.max_M = max_M
        ctx.E = E
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, W_gate, W_up, gate_save, up_save, counts, offsets = ctx.saved_tensors
        max_M = ctx.max_M
        E = ctx.E
        M_total, K = x.shape
        H = W_gate.shape[2]

        grad_output = grad_output.contiguous()

        # Step 1: Compute d_gate and d_up from grad_output, gate_save, up_save
        d_gate = torch.empty_like(grad_output)
        d_up = torch.empty_like(grad_output)

        N_total = M_total * H
        BLOCK = 1024
        grid_elem = ((N_total + BLOCK - 1) // BLOCK,)
        _fused_gate_up_silu_bwd_elem_kernel[grid_elem](
            grad_output.view(-1),
            gate_save.view(-1),
            up_save.view(-1),
            d_gate.view(-1),
            d_up.view(-1),
            N_total,
            1,
            1,
            1,
            1,
            1,
            BLOCK=BLOCK,
        )

        # Step 2: Use grouped GEMM for weight and input gradients
        # Import from our Priority 1 kernel
        from .triton_moe_grouped_gemm import (
            _grouped_gemm_dweight,
            _grouped_gemm_forward,
        )

        # dX = d_gate @ W_gate^T + d_up @ W_up^T (two grouped GEMMs)
        W_gate_t = W_gate.transpose(-2, -1).contiguous()
        W_up_t = W_up.transpose(-2, -1).contiguous()
        dx_gate = _grouped_gemm_forward(d_gate, W_gate_t, offsets, E, max_M)
        dx_up = _grouped_gemm_forward(d_up, W_up_t, offsets, E, max_M)
        dx = dx_gate + dx_up

        # dW_gate = x^T @ d_gate, dW_up = x^T @ d_up (per expert)
        dW_gate = _grouped_gemm_dweight(
            x, d_gate, offsets, E, K, H, max_M, W_gate.dtype
        )
        dW_up = _grouped_gemm_dweight(x, d_up, offsets, E, K, H, max_M, W_up.dtype)

        return dx, dW_gate, dW_up, None, None, None, None


def fused_moe_gate_up_silu(x, W_gate, W_up, expert_counts):
    """
    Fused gate+up+SiLU for MoE experts.

    Args:
        x: [M_total, K] — sorted tokens
        W_gate: [E, K, H] — expert gate weights
        W_up: [E, K, H] — expert up weights
        expert_counts: [E] — tokens per expert

    Returns: [M_total, H] — SiLU(x @ W_gate[e]) * (x @ W_up[e])
    """
    x = x.contiguous()
    W_gate = W_gate.contiguous()
    W_up = W_up.contiguous()

    E = W_gate.shape[0]
    offsets, counts = _compute_offsets(expert_counts, x.device)
    max_M = int(counts.max().item()) if counts.numel() > 0 else 0

    return FusedMoEGateUpSiLUFn.apply(x, W_gate, W_up, counts, offsets, max_M, E)


# ============================================================================
# Reference implementation
# ============================================================================


def pytorch_fused_gate_up_silu(x, W_gate, W_up, expert_counts):
    """Reference: two separate grouped GEMMs + SiLU*up."""
    import torch.nn.functional as F

    E = W_gate.shape[0]
    offsets, _ = _compute_offsets(expert_counts, x.device)
    H = W_gate.shape[2]
    chunks = []
    for e in range(E):
        s = offsets[e].item()
        t = offsets[e + 1].item()
        if s < t:
            gate_out = x[s:t] @ W_gate[e]
            up_out = x[s:t] @ W_up[e]
            chunks.append(F.silu(gate_out) * up_out)
    if chunks:
        return torch.cat(chunks, dim=0)
    return torch.empty(0, H, device=x.device, dtype=x.dtype)
