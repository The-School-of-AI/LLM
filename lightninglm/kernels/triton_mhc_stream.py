"""
Fused mHC Stream Collapse/Expand kernels.

Collapse: x_in = (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)
Expand+Residual: out = y.unsqueeze(2) * H_post.unsqueeze(-1) + H_res @ x_stream

Called 16x per forward pass (8 layers x 2 sublayers).
These are purely memory-bound kernels targeting 300 GB/s on L4.
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# Collapse kernel: weighted sum over stream dimension
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _mhc_collapse_kernel(
    X_stream_ptr,
    H_pre_ptr,
    X_out_ptr,
    B_T,
    D: tl.constexpr,
    N: tl.constexpr,  # n_streams = 4
    stride_xs_bt,
    stride_xs_n,
    stride_xs_d,
    stride_hp_bt,
    stride_hp_n,
    stride_xo_bt,
    stride_xo_d,
    BLOCK_D: tl.constexpr,
):
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    base_xs = X_stream_ptr + pid_bt * stride_xs_bt
    base_hp = H_pre_ptr + pid_bt * stride_hp_bt

    for s in tl.static_range(N):
        h = tl.load(base_hp + s * stride_hp_n).to(tl.float32)
        x = tl.load(
            base_xs + s * stride_xs_n + d_offset * stride_xs_d, mask=d_mask, other=0.0
        ).to(tl.float32)
        acc += h * x

    tl.store(
        X_out_ptr + pid_bt * stride_xo_bt + d_offset * stride_xo_d,
        acc.to(X_out_ptr.dtype.element_ty),
        mask=d_mask,
    )


# ============================================================================
# Expand + Residual kernel: y * H_post + H_res @ x_stream
# Process all N streams per program to read x_stream only once
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_D": 4096}, num_warps=8, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _mhc_expand_residual_kernel(
    Y_ptr,
    X_stream_ptr,
    H_post_ptr,
    H_res_ptr,
    Out_ptr,
    B_T,
    D: tl.constexpr,
    N: tl.constexpr,  # n_streams = 4
    stride_y_bt,
    stride_y_d,
    stride_xs_bt,
    stride_xs_n,
    stride_xs_d,
    stride_hp_bt,
    stride_hp_n,
    stride_hr_bt,
    stride_hr_s,
    stride_hr_j,
    stride_o_bt,
    stride_o_n,
    stride_o_d,
    BLOCK_D: tl.constexpr,
):
    """Process all N streams per (bt, d_tile), reading x_stream and y once."""
    pid_bt = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offset < D

    # Load y once (shared across all streams)
    y_val = tl.load(
        Y_ptr + pid_bt * stride_y_bt + d_offset * stride_y_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    # Load all N x_stream slices once (N vectors of BLOCK_D)
    base_xs = X_stream_ptr + pid_bt * stride_xs_bt
    # We'll load x_stream[j] on demand in the inner loop, but it's shared across output streams

    base_hr = H_res_ptr + pid_bt * stride_hr_bt
    base_hp = H_post_ptr + pid_bt * stride_hp_bt

    for s in tl.static_range(N):
        # Expand: y * H_post[s]
        h_post_s = tl.load(base_hp + s * stride_hp_n).to(tl.float32)
        expand = h_post_s * y_val

        # Residual: H_res[s, :] @ x_stream[:, d]
        res_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in tl.static_range(N):
            h_res_sj = tl.load(base_hr + s * stride_hr_s + j * stride_hr_j).to(
                tl.float32
            )
            x_sj = tl.load(
                base_xs + j * stride_xs_n + d_offset * stride_xs_d,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            res_acc += h_res_sj * x_sj

        out = res_acc + expand
        tl.store(
            Out_ptr + pid_bt * stride_o_bt + s * stride_o_n + d_offset * stride_o_d,
            out.to(Out_ptr.dtype.element_ty),
            mask=d_mask,
        )


# ============================================================================
# Autograd wrappers
# ============================================================================


class FusedMHCCollapseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_stream, H_pre):
        """
        x_stream: [B, T, N, D]  (N=4 streams, D=4096)
        H_pre:    [B, T, N]
        Returns:  [B, T, D]
        """
        B, T, N, D = x_stream.shape
        B_T = B * T

        x_out = torch.empty(B, T, D, device=x_stream.device, dtype=x_stream.dtype)

        xs = x_stream.reshape(B_T, N, D)
        hp = H_pre.reshape(B_T, N)
        xo = x_out.reshape(B_T, D)

        grid = lambda meta: (B_T, triton.cdiv(D, meta["BLOCK_D"]))
        _mhc_collapse_kernel[grid](
            xs,
            hp,
            xo,
            B_T,
            D,
            N,
            xs.stride(0),
            xs.stride(1),
            xs.stride(2),
            hp.stride(0),
            hp.stride(1),
            xo.stride(0),
            xo.stride(1),
        )

        ctx.save_for_backward(x_stream, H_pre)
        return x_out

    @staticmethod
    def backward(ctx, grad_output):
        x_stream, H_pre = ctx.saved_tensors
        # grad_output: [B, T, D]
        # dH_pre = (grad_output.unsqueeze(2) * x_stream).sum(-1)  → [B, T, N]
        # dX_stream = grad_output.unsqueeze(2) * H_pre.unsqueeze(-1)  → [B, T, N, D]
        dX_stream = grad_output.unsqueeze(2) * H_pre.unsqueeze(-1)
        dH_pre = (grad_output.unsqueeze(2) * x_stream).sum(-1)
        return dX_stream, dH_pre


class FusedMHCExpandResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, x_stream, H_post, H_res):
        """
        y:        [B, T, D]
        x_stream: [B, T, N, D]
        H_post:   [B, T, N]
        H_res:    [B, T, N, N]
        Returns:  [B, T, N, D]
        """
        B, T, N, D = x_stream.shape
        B_T = B * T

        out = torch.empty(B, T, N, D, device=y.device, dtype=y.dtype)

        y_flat = y.reshape(B_T, D)
        xs_flat = x_stream.reshape(B_T, N, D)
        hp_flat = H_post.reshape(B_T, N)
        hr_flat = H_res.reshape(B_T, N, N)
        o_flat = out.reshape(B_T, N, D)

        grid = lambda meta: (B_T, triton.cdiv(D, meta["BLOCK_D"]))
        _mhc_expand_residual_kernel[grid](
            y_flat,
            xs_flat,
            hp_flat,
            hr_flat,
            o_flat,
            B_T,
            D,
            N,
            y_flat.stride(0),
            y_flat.stride(1),
            xs_flat.stride(0),
            xs_flat.stride(1),
            xs_flat.stride(2),
            hp_flat.stride(0),
            hp_flat.stride(1),
            hr_flat.stride(0),
            hr_flat.stride(1),
            hr_flat.stride(2),
            o_flat.stride(0),
            o_flat.stride(1),
            o_flat.stride(2),
        )

        ctx.save_for_backward(y, x_stream, H_post, H_res)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y, x_stream, H_post, H_res = ctx.saved_tensors
        # grad_output: [B, T, N, D]
        # Forward: out = y.unsqueeze(2) * H_post.unsqueeze(-1) + H_res @ x_stream

        # dY = (grad_output * H_post.unsqueeze(-1)).sum(dim=2)
        dY = (grad_output * H_post.unsqueeze(-1)).sum(dim=2)

        # dH_post = (grad_output * y.unsqueeze(2)).sum(dim=-1)
        dH_post = (grad_output * y.unsqueeze(2)).sum(dim=-1)

        # dX_stream = H_res^T @ grad_output
        dX_stream = torch.matmul(H_res.transpose(-1, -2), grad_output)

        # dH_res = grad_output @ x_stream^T  (per batch×time, [N,D] @ [D,N] = [N,N])
        dH_res = torch.matmul(grad_output, x_stream.transpose(-1, -2))

        return dY, dX_stream, dH_post, dH_res


def fused_mhc_collapse(x_stream, H_pre):
    """Fused mHC stream collapse: weighted sum over stream dimension."""
    return FusedMHCCollapseFn.apply(x_stream, H_pre)


def fused_mhc_expand_residual(y, x_stream, H_post, H_res):
    """Fused mHC expand + residual routing."""
    return FusedMHCExpandResidualFn.apply(y, x_stream, H_post, H_res)


# ============================================================================
# PyTorch reference implementations
# ============================================================================


def pytorch_mhc_collapse(x_stream, H_pre):
    return (x_stream * H_pre.unsqueeze(-1)).sum(dim=2)


def pytorch_mhc_expand_residual(y, x_stream, H_post, H_res):
    y_stream = y.unsqueeze(2) * H_post.unsqueeze(-1)
    x_res = torch.matmul(H_res, x_stream)
    return x_res + y_stream
