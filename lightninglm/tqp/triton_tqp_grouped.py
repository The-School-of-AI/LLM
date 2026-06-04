"""
Triton grouped kernels for TQP MoE forward and backward.

Replaces the per-expert Python loop with fused grouped operations:
  - grouped_dequant_matmul: fused codebook lookup + norm scaling + matmul
  - grouped_tqp: batched rank-r matmul for all experts
  - grouped_rotate: batched rotation for all tokens

Provides FusedTQExpertForward autograd.Function for training.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# ============================================================================
# Triton Kernels
# ============================================================================


@triton.jit
def _grouped_dequant_matmul_kernel(
    X_ptr,
    INDICES_ptr,
    NORMS_ptr,
    CODEBOOK_ptr,
    OFFSETS_ptr,
    Y_ptr,
    D_IN: tl.constexpr,
    D_OUT: tl.constexpr,
    stride_x_m,
    stride_x_d,
    stride_idx_e,
    stride_idx_o,
    stride_idx_i,
    stride_norm_e,
    stride_norm_o,
    stride_y_m,
    stride_y_d,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_expert = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    expert_start = tl.load(OFFSETS_ptr + pid_expert)
    expert_end = tl.load(OFFSETS_ptr + pid_expert + 1)
    expert_M = expert_end - expert_start

    if expert_M <= 0:
        return
    if pid_m * BLOCK_M >= expert_M:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    norms = tl.load(
        NORMS_ptr + pid_expert * stride_norm_e + rn * stride_norm_o,
        mask=rn < D_OUT,
        other=0.0,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, D_IN, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr
            + (expert_start + rm)[:, None] * stride_x_m
            + rk[None, :] * stride_x_d,
            mask=(rm[:, None] < expert_M) & (rk[None, :] < D_IN),
            other=0.0,
        ).to(tl.float32)
        idx = tl.load(
            INDICES_ptr
            + pid_expert * stride_idx_e
            + rn[:, None] * stride_idx_o
            + rk[None, :] * stride_idx_i,
            mask=(rn[:, None] < D_OUT) & (rk[None, :] < D_IN),
            other=0,
        )
        w = tl.load(CODEBOOK_ptr + idx.to(tl.int32))
        acc += tl.dot(x, tl.trans(w))

    acc = acc * norms[None, :]

    tl.store(
        Y_ptr + (expert_start + rm)[:, None] * stride_y_m + rn[None, :] * stride_y_d,
        acc.to(tl.bfloat16),
        mask=(rm[:, None] < expert_M) & (rn[None, :] < D_OUT),
    )


# ============================================================================
# Python wrappers
# ============================================================================


def grouped_dequant_matmul(x_sorted, offsets, indices, norms, codebook):
    """
    Fused dequant + matmul for ALL experts in one launch.
    x_sorted: [total_M, d_in] bf16
    offsets: [E+1] int64
    indices: [E, d_out, d_in] int8
    norms: [E, d_out] fp32
    codebook: [num_levels] fp32
    returns: [total_M, d_out] bf16
    """
    total_M, D_IN = x_sorted.shape
    E, D_OUT = indices.shape[0], indices.shape[1]
    y = torch.empty(total_M, D_OUT, device=x_sorted.device, dtype=torch.bfloat16)

    if total_M == 0:
        return y

    BLOCK_M = 32
    BLOCK_N = min(64, triton.next_power_of_2(D_OUT))
    BLOCK_K = 64
    max_M = max(1, int((offsets[1:] - offsets[:-1]).max().item()))
    grid = (triton.cdiv(max_M, BLOCK_M), triton.cdiv(D_OUT, BLOCK_N), E)

    _grouped_dequant_matmul_kernel[grid](
        x_sorted,
        indices,
        norms,
        codebook,
        offsets,
        y,
        D_IN,
        D_OUT,
        x_sorted.stride(0),
        x_sorted.stride(1),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        norms.stride(0),
        norms.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return y


@triton.jit
def _grouped_dequant_matmul_bwd_kernel(
    G_ptr,
    INDICES_ptr,
    NORMS_ptr,
    CODEBOOK_ptr,
    OFFSETS_ptr,
    DX_ptr,
    D_OUT: tl.constexpr,
    D_IN: tl.constexpr,
    stride_g_m,
    stride_g_n,
    stride_idx_e,
    stride_idx_o,
    stride_idx_i,
    stride_norm_e,
    stride_norm_o,
    stride_dx_m,
    stride_dx_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Backward: grad_x[m, k] = sum_n grad_out[m, n] * W[n, k]
    where W[n, k] = codebook[indices[expert, n, k]] * norms[expert, n]
    This is grad_out @ W (no transpose — W is [d_out, d_in]).
    """
    pid_expert = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    expert_start = tl.load(OFFSETS_ptr + pid_expert)
    expert_end = tl.load(OFFSETS_ptr + pid_expert + 1)
    expert_M = expert_end - expert_start

    if expert_M <= 0:
        return
    if pid_m * BLOCK_M >= expert_M:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n in range(0, D_OUT, BLOCK_N):
        rn = n + tl.arange(0, BLOCK_N)

        # Load grad_output block: G[expert_start + rm, rn]
        g = tl.load(
            G_ptr
            + (expert_start + rm)[:, None] * stride_g_m
            + rn[None, :] * stride_g_n,
            mask=(rm[:, None] < expert_M) & (rn[None, :] < D_OUT),
            other=0.0,
        ).to(tl.float32)

        # Load norms for this block of output dims
        norms = tl.load(
            NORMS_ptr + pid_expert * stride_norm_e + rn * stride_norm_o,
            mask=rn < D_OUT,
            other=0.0,
        )

        # Scale g by norms: g_scaled[m, n] = g[m, n] * norms[n]
        g = g * norms[None, :]

        # Load and dequantize: W[rn, rk]
        idx = tl.load(
            INDICES_ptr
            + pid_expert * stride_idx_e
            + rn[:, None] * stride_idx_o
            + rk[None, :] * stride_idx_i,
            mask=(rn[:, None] < D_OUT) & (rk[None, :] < D_IN),
            other=0,
        )
        w = tl.load(CODEBOOK_ptr + idx.to(tl.int32))  # [BLOCK_N, BLOCK_K]

        # acc += g @ W  (g is [BLOCK_M, BLOCK_N], W is [BLOCK_N, BLOCK_K])
        acc += tl.dot(g, w)

    tl.store(
        DX_ptr + (expert_start + rm)[:, None] * stride_dx_m + rk[None, :] * stride_dx_k,
        acc.to(tl.bfloat16),
        mask=(rm[:, None] < expert_M) & (rk[None, :] < D_IN),
    )


def grouped_dequant_matmul_bwd(grad_output, offsets, indices, norms, codebook):
    """
    Backward: grad_x = grad_output @ W_dequant for ALL experts.
    grad_output: [total_M, d_out] bf16
    returns: [total_M, d_in] bf16
    """
    total_M, D_OUT = grad_output.shape
    E, _, D_IN = indices.shape[0], indices.shape[1], indices.shape[2]
    dx = torch.empty(total_M, D_IN, device=grad_output.device, dtype=torch.bfloat16)

    if total_M == 0:
        return dx

    BLOCK_M = 32
    BLOCK_K = 64
    BLOCK_N = 64
    max_M = max(1, int((offsets[1:] - offsets[:-1]).max().item()))
    grid = (triton.cdiv(max_M, BLOCK_M), triton.cdiv(D_IN, BLOCK_K), E)

    _grouped_dequant_matmul_bwd_kernel[grid](
        grad_output,
        indices,
        norms,
        codebook,
        offsets,
        dx,
        D_OUT,
        D_IN,
        grad_output.stride(0),
        grad_output.stride(1),
        indices.stride(0),
        indices.stride(1),
        indices.stride(2),
        norms.stride(0),
        norms.stride(1),
        dx.stride(0),
        dx.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return dx


def grouped_rotate(x_sorted, offsets, rotation_matrix):
    """
    Apply rotation to sorted tokens. All experts share the same rotation matrix.
    x_sorted: [total_M, d_in] bf16
    rotation_matrix: [d_in, d_in] bf16
    returns: [total_M, d_in] bf16

    Uses a single batched matmul (no per-expert loop needed since R is shared).
    """
    return x_sorted @ rotation_matrix.to(dtype=x_sorted.dtype).t()


def grouped_tqp(x_rotated, offsets, tqp_A, tqp_B):
    """
    Grouped TQP: y = (x @ B.T) @ A.T for all experts.
    Uses per-expert loop but with tiny rank-16 matmuls (fast).

    x_rotated: [total_M, d_in] bf16
    offsets: [E+1] int64
    tqp_A: [E, d_out, rank] bf16
    tqp_B: [E, rank, d_in] bf16
    returns: [total_M, d_out] bf16
    """
    E = tqp_A.shape[0]
    D_OUT = tqp_A.shape[1]
    total_M = x_rotated.shape[0]
    y = torch.zeros(total_M, D_OUT, device=x_rotated.device, dtype=x_rotated.dtype)

    if total_M == 0:
        return y

    for e in range(E):
        start = offsets[e].item()
        end = offsets[e + 1].item()
        if end > start:
            chunk = x_rotated[start:end]
            xB = F.linear(chunk, tqp_B[e])
            y[start:end] = F.linear(xB, tqp_A[e])

    return y


# ============================================================================
# Autograd Function for training
# ============================================================================


class FusedTQExpertForward(torch.autograd.Function):
    """
    Fused forward for one TQ expert weight set (gate, up, or down).

    Forward: y = dequant_matmul(rotate(x), indices, norms, cb) + tqp(rotate(x), A, B)
    Backward: computes gradients for x, tqp_A, tqp_B
    (indices, norms, codebook, rotation are frozen — no gradients)
    """

    @staticmethod
    def forward(
        ctx,
        x_sorted,
        offsets,
        rotation_matrix,
        weight_indices,
        row_norms,
        codebook,
        tqp_A,
        tqp_B,
    ):
        """
        x_sorted: [M, d_in] bf16 — NOT pre-rotated
        offsets: [E+1] int64
        rotation_matrix: [d_in, d_in] buffer
        weight_indices: [E, d_out, d_in] int8 buffer
        row_norms: [E, d_out] buffer
        codebook: [num_levels] buffer
        tqp_A: [E, d_out, rank] Parameter
        tqp_B: [E, rank, d_in] Parameter
        returns: [M, d_out] bf16
        """
        # Rotate
        x_rot = grouped_rotate(x_sorted, offsets, rotation_matrix)

        # Base: fused dequant + matmul (Triton)
        norms_f = row_norms.float()
        cb_f = codebook.float()
        y_base = grouped_dequant_matmul(x_rot, offsets, weight_indices, norms_f, cb_f)

        # Clamp to prevent bf16 overflow in SwiGLU
        y_base = y_base.clamp(-60000.0, 60000.0)

        # TQP
        y_tqp = grouped_tqp(x_rot, offsets, tqp_A, tqp_B)

        y = y_base + y_tqp

        # Save for backward
        ctx.save_for_backward(
            x_rot,
            offsets,
            tqp_A,
            tqp_B,
            weight_indices,
            row_norms,
            codebook,
            rotation_matrix,
        )
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (
            x_rot,
            offsets,
            tqp_A,
            tqp_B,
            weight_indices,
            row_norms,
            codebook,
            rotation_matrix,
        ) = ctx.saved_tensors

        E = tqp_A.shape[0]
        total_M = x_rot.shape[0]
        D_IN = x_rot.shape[1]

        grad_x_rot = None
        grad_tqp_A = None
        grad_tqp_B = None

        if ctx.needs_input_grad[0]:  # grad w.r.t. x_sorted
            # dy/dx_rot = W_base.T + (A @ B).T for each expert
            # Then dx/dx_sorted = dx_rot @ R (rotation backward)
            grad_x_rot = torch.zeros_like(x_rot)

        if ctx.needs_input_grad[6]:  # grad w.r.t. tqp_A
            grad_tqp_A = torch.zeros_like(tqp_A)

        if ctx.needs_input_grad[7]:  # grad w.r.t. tqp_B
            grad_tqp_B = torch.zeros_like(tqp_B)

        norms_f = row_norms.float()
        cb_f = codebook.float()

        # === Fused backward for base matmul (Triton grouped kernel) ===
        if grad_x_rot is not None:
            grad_x_rot = grouped_dequant_matmul_bwd(
                grad_output, offsets, weight_indices, norms_f, cb_f
            )

        # === TQP backward (per-expert loop — rank-16 matmuls are tiny) ===
        for e in range(E):
            start = offsets[e].item()
            end = offsets[e + 1].item()
            if end <= start:
                continue

            x_e = x_rot[start:end]
            g_e = grad_output[start:end]
            A_e = tqp_A[e]
            B_e = tqp_B[e]
            xB = F.linear(x_e, B_e)

            if grad_tqp_A is not None:
                grad_tqp_A[e] = g_e.t() @ xB

            if grad_tqp_B is not None:
                grad_xB = g_e @ A_e
                grad_tqp_B[e] = grad_xB.t() @ x_e

            if grad_x_rot is not None:
                grad_xB = g_e @ A_e
                grad_x_rot[start:end] += grad_xB @ B_e

        # Rotate gradient back
        grad_x_sorted = None
        if grad_x_rot is not None:
            grad_x_sorted = grad_x_rot @ rotation_matrix.to(dtype=grad_x_rot.dtype)

        # offsets, rotation_matrix, weight_indices, row_norms, codebook have no grad
        return grad_x_sorted, None, None, None, None, None, grad_tqp_A, grad_tqp_B


def fused_expert_forward(x_sorted, offsets, tq_weights):
    """
    Drop-in replacement for the per-expert Python loop.
    x_sorted: [M, d_in] bf16 (NOT pre-rotated)
    offsets: [E+1] int64
    tq_weights: TurboQuantPretrainingExpertWeights instance
    returns: [M, d_out] bf16
    """
    return FusedTQExpertForward.apply(
        x_sorted,
        offsets,
        tq_weights.rotation_matrix,
        tq_weights.weight_indices,
        tq_weights.row_norms,
        tq_weights.weight_codebook,
        tq_weights.tqp_A,
        tq_weights.tqp_B,
    )


# ============================================================================
# Test: correctness + speed + gradient check
# ============================================================================

if __name__ == "__main__":
    import time

    from lightninglm.tqp.turboquant_pretraining_linear import (
        TurboQuantPretrainingExpertWeights,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(42)

    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    for n_experts, d_in, d_out, label in [
        (20, 4096, 1024, "W_gate (small)"),
        (260, 4096, 1024, "W_gate (70B)"),
        (260, 1024, 4096, "W_down (70B)"),
    ]:
        print(f"\n{'='*60}")
        print(f"{label}: E={n_experts}, [{d_in}]->[{d_out}]")
        print(f"{'='*60}")

        W = torch.randn(n_experts, d_in, d_out, dtype=torch.bfloat16) * (
            1.0 / d_in**0.5
        )
        tq = TurboQuantPretrainingExpertWeights.from_weight_lazy(
            W, weight_bits=4, rank=16
        )
        tq._materialize_lazy_gpu(device)
        tq = tq.to(device)

        tokens_per_expert = max(1, 4096 // n_experts)
        total_M = tokens_per_expert * n_experts
        x = torch.randn(total_M, d_in, device=device, dtype=torch.bfloat16)

        offsets = torch.zeros(n_experts + 1, device=device, dtype=torch.int64)
        for e in range(n_experts):
            offsets[e + 1] = min(offsets[e] + tokens_per_expert, total_M)

        # --- 1. Correctness (forward) ---
        ref = torch.empty(total_M, d_out, device=device, dtype=torch.bfloat16)
        for e in range(n_experts):
            s, en = offsets[e].item(), offsets[e + 1].item()
            if en > s:
                ref[s:en] = tq.compute_expert_chunk(x[s:en], e).to(torch.bfloat16)

        fused = fused_expert_forward(x, offsets, tq)
        rel_err = (ref - fused).norm() / ref.norm()
        print(
            f"  Forward correctness: rel_err={rel_err.item():.6f} "
            f"{'PASS' if rel_err < 0.02 else 'FAIL'}"
        )

        # --- 2. Gradient check (tqp_A) ---
        tq.tqp_A.grad = None
        tq.tqp_B.grad = None
        x_grad = x.detach().requires_grad_(True)
        out = fused_expert_forward(x_grad, offsets, tq)
        loss = out.sum()
        loss.backward()
        a_grad = tq.tqp_A.grad.norm().item() if tq.tqp_A.grad is not None else -1
        b_grad = tq.tqp_B.grad.norm().item() if tq.tqp_B.grad is not None else -1
        x_grad_norm = x_grad.grad.norm().item() if x_grad.grad is not None else -1
        print(f"  Gradients: tqp_A={a_grad:.4f} tqp_B={b_grad:.4f} x={x_grad_norm:.4f}")
        tqp_ok = a_grad > 0
        print(f"  tqp_A grad: {'PASS' if tqp_ok else 'FAIL (zero!)'}")

        # --- 3. Speed benchmark ---
        n_iters = 30
        # Warmup
        for _ in range(5):
            for e in range(n_experts):
                s, en = offsets[e].item(), offsets[e + 1].item()
                if en > s:
                    _ = tq.compute_expert_chunk(x[s:en], e)
            _ = fused_expert_forward(x, offsets, tq)
        torch.cuda.synchronize()

        # Baseline (with grad)
        times_base = []
        for _ in range(n_iters):
            tq.tqp_A.grad = None
            tq.tqp_B.grad = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_out = []
            for e in range(n_experts):
                s, en = offsets[e].item(), offsets[e + 1].item()
                if en > s:
                    total_out.append(tq.compute_expert_chunk(x[s:en], e))
            out_cat = torch.cat(total_out)
            out_cat.sum().backward()
            torch.cuda.synchronize()
            times_base.append((time.perf_counter() - t0) * 1000)

        # Fused (with grad)
        times_fused = []
        for _ in range(n_iters):
            tq.tqp_A.grad = None
            tq.tqp_B.grad = None
            x_g = x.detach().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fused_expert_forward(x_g, offsets, tq)
            out.sum().backward()
            torch.cuda.synchronize()
            times_fused.append((time.perf_counter() - t0) * 1000)

        avg_base = sum(sorted(times_base)[3:-3]) / (n_iters - 6)
        avg_fused = sum(sorted(times_fused)[3:-3]) / (n_iters - 6)
        speedup = avg_base / avg_fused

        print(f"  Baseline (fwd+bwd): {avg_base:8.2f} ms")
        print(f"  Fused (fwd+bwd):    {avg_fused:8.2f} ms")
        print(f"  Speedup:            {speedup:8.1f}x")

    print("\nDONE")
