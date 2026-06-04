"""
Triton Grouped GEMM for MoE expert compute.

Drop-in replacement for the external grouped_gemm package.
Each expert e processes its contiguous block of tokens:
    output[offset_e : offset_e + M_e] = a[offset_e : offset_e + M_e] @ b[e]

Called 3x per MoEFFN (gate, up, down) x 8 layers = 24 grouped GEMMs per forward pass.
"""

import torch
import triton
import triton.language as tl

# ============================================================================
# Forward kernel: C[expert_block] = A[expert_block] @ B[expert]
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2
        ),
    ],
    key=["K", "N"],
)
@triton.jit
def _grouped_gemm_fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    Offsets_ptr,  # [E+1] cumulative offsets
    K: tl.constexpr,
    N: tl.constexpr,
    stride_ak,
    stride_an_unused,  # A is [M_total, K], row-major
    stride_be,
    stride_bk,
    stride_bn,  # B is [E, K, N]
    stride_cm,
    stride_cn,  # C is [M_total, N]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_e = tl.program_id(0)  # expert index
    pid_m = tl.program_id(1)  # M-tile index within this expert
    pid_n = tl.program_id(2)  # N-tile index

    # Look up this expert's token range
    start = tl.load(Offsets_ptr + pid_e)
    end = tl.load(Offsets_ptr + pid_e + 1)
    M_e = end - start

    # Early exit if this M-tile is beyond this expert's tokens
    if pid_m * BLOCK_M >= M_e:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Base pointers for this expert
    a_base = A_ptr + start * stride_ak
    b_base = B_ptr + pid_e * stride_be
    c_base = C_ptr + start * stride_cm

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = a_base + offs_m[:, None] * stride_ak + offs_k[None, :]
        mask_a = (offs_m[:, None] < M_e) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N]
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc += tl.dot(a_tile, b_tile)

    # Store result
    c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M_e) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_c)


# ============================================================================
# Weight gradient kernel: dB[e] = A_e^T @ dC_e for each expert
# ============================================================================


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 64}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_K": 128, "BLOCK_N": 64, "BLOCK_M": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_K": 64, "BLOCK_N": 128, "BLOCK_M": 32}, num_warps=4, num_stages=2
        ),
        triton.Config(
            {"BLOCK_K": 128, "BLOCK_N": 128, "BLOCK_M": 32}, num_warps=8, num_stages=2
        ),
    ],
    key=["K_out", "N_out"],
)
@triton.jit
def _grouped_gemm_dweight_kernel(
    # dB[e] = A_e^T @ dC_e
    # A_e: [M_e, K_in]  (transposed to [K_in, M_e])
    # dC_e: [M_e, N_out]
    # dB[e]: [K_in, N_out]
    A_ptr,
    dC_ptr,
    dB_ptr,
    Offsets_ptr,
    K_out: tl.constexpr,  # K_in dimension (output rows of dB)
    N_out: tl.constexpr,  # N dimension (output cols of dB)
    stride_am,
    stride_ak,  # A is [M_total, K_in]
    stride_dcm,
    stride_dcn,  # dC is [M_total, N_out]
    stride_dbe,
    stride_dbk,
    stride_dbn,  # dB is [E, K_in, N_out]
    BLOCK_K: tl.constexpr,  # tiles K_in dimension
    BLOCK_N: tl.constexpr,  # tiles N_out dimension
    BLOCK_M: tl.constexpr,  # tiles M_e (reduction) dimension
):
    pid_e = tl.program_id(0)
    pid_k = tl.program_id(1)  # K_in tile
    pid_n = tl.program_id(2)  # N_out tile

    start = tl.load(Offsets_ptr + pid_e)
    end = tl.load(Offsets_ptr + pid_e + 1)
    M_e = end - start

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_base = A_ptr + start * stride_am
    dc_base = dC_ptr + start * stride_dcm

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)

    for m_start in range(0, M_e, BLOCK_M):
        offs_m = m_start + tl.arange(0, BLOCK_M)

        # Load A^T tile: A[m, k] transposed → [BLOCK_K, BLOCK_M] read as A[m, k]
        a_ptrs = a_base + offs_m[None, :] * stride_am + offs_k[:, None] * stride_ak
        mask_a = (offs_m[None, :] < M_e) & (offs_k[:, None] < K_out)
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)  # [BLOCK_K, BLOCK_M]

        # Load dC tile [BLOCK_M, BLOCK_N]
        dc_ptrs = dc_base + offs_m[:, None] * stride_dcm + offs_n[None, :] * stride_dcn
        mask_dc = (offs_m[:, None] < M_e) & (offs_n[None, :] < N_out)
        dc_tile = tl.load(dc_ptrs, mask=mask_dc, other=0.0)  # [BLOCK_M, BLOCK_N]

        acc += tl.dot(
            a_tile, dc_tile
        )  # [BLOCK_K, BLOCK_M] @ [BLOCK_M, BLOCK_N] = [BLOCK_K, BLOCK_N]

    # Store dB[e, k, n]
    db_base = dB_ptr + pid_e * stride_dbe
    db_ptrs = db_base + offs_k[:, None] * stride_dbk + offs_n[None, :] * stride_dbn
    mask_db = (offs_k[:, None] < K_out) & (offs_n[None, :] < N_out)
    tl.store(db_ptrs, acc.to(dB_ptr.dtype.element_ty), mask=mask_db)


# ============================================================================
# Python wrappers
# ============================================================================


def _compute_offsets(expert_counts, device):
    """Compute [E+1] cumulative offsets from expert counts."""
    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.to(device=device, dtype=torch.int64).contiguous()
    else:
        counts = torch.tensor(list(expert_counts), device=device, dtype=torch.int64)
    offsets = torch.zeros(counts.shape[0] + 1, device=device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, counts


def _grouped_gemm_forward(a, b, offsets, E, max_M):
    """
    a: [M_total, K], b: [E, K, N], offsets: [E+1]
    Returns: [M_total, N]
    """
    M_total, K = a.shape
    N = b.shape[2]

    c = torch.empty(M_total, N, device=a.device, dtype=a.dtype)
    if M_total == 0:
        return c

    # Grid: (experts, max_M_tiles, N_tiles)
    BLOCK_M_est = 64  # will be overridden by autotune
    max_M_tiles = max((max_M + BLOCK_M_est - 1) // BLOCK_M_est, 1)
    # Use generous upper bound for grid since kernel early-exits
    max_M_tiles = max((max_M + 31) // 32, 1)  # smallest BLOCK_M in configs is 32

    def grid(meta):
        mm_tiles = (max_M + meta["BLOCK_M"] - 1) // meta["BLOCK_M"]
        nn_tiles = (N + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
        return (E, mm_tiles, nn_tiles)

    _grouped_gemm_fwd_kernel[grid](
        a,
        b,
        c,
        offsets,
        K,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
    )
    return c


def _grouped_gemm_dweight(a, grad_c, offsets, E, K_in, N_out, max_M, out_dtype):
    """
    Compute dB[e] = a_e^T @ grad_c_e for each expert.
    a: [M_total, K_in], grad_c: [M_total, N_out]
    Returns: dB [E, K_in, N_out]
    """
    dB = torch.zeros(E, K_in, N_out, device=a.device, dtype=out_dtype)
    M_total = a.shape[0]
    if M_total == 0:
        return dB

    def grid(meta):
        kk_tiles = (K_in + meta["BLOCK_K"] - 1) // meta["BLOCK_K"]
        nn_tiles = (N_out + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]
        return (E, kk_tiles, nn_tiles)

    _grouped_gemm_dweight_kernel[grid](
        a,
        grad_c,
        dB,
        offsets,
        K_in,
        N_out,
        a.stride(0),
        a.stride(1),
        grad_c.stride(0),
        grad_c.stride(1),
        dB.stride(0),
        dB.stride(1),
        dB.stride(2),
    )
    return dB


# ============================================================================
# Autograd Function
# ============================================================================


class TritonGroupedGEMMFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, expert_counts_tensor, offsets, max_M, E):
        c = _grouped_gemm_forward(a, b, offsets, E, max_M)
        ctx.save_for_backward(a, b, expert_counts_tensor, offsets)
        ctx.max_M = max_M
        ctx.E = E
        return c

    @staticmethod
    def backward(ctx, grad_output):
        a, b, expert_counts_tensor, offsets = ctx.saved_tensors
        max_M = ctx.max_M
        E = ctx.E

        grad_output = grad_output.contiguous()

        # dA = grad_output @ b^T  (grouped, same structure)
        # b is [E, K, N], b^T is [E, N, K]
        b_t = b.transpose(-2, -1).contiguous()
        grad_a = _grouped_gemm_forward(grad_output, b_t, offsets, E, max_M)

        # dB[e] = a_e^T @ grad_output_e
        K_in = a.shape[1]
        N_out = grad_output.shape[1]
        grad_b = _grouped_gemm_dweight(
            a, grad_output, offsets, E, K_in, N_out, max_M, b.dtype
        )

        return grad_a, grad_b, None, None, None, None


# ============================================================================
# Public API
# ============================================================================


def triton_grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    expert_counts,
) -> torch.Tensor:
    """
    Grouped GEMM: for each expert e with M_e tokens (defined by expert_counts),
    compute output[offset_e : offset_e+M_e] = a[offset_e : offset_e+M_e] @ b[e]

    Args:
        a: [M_total, K] — sorted tokens, contiguous
        b: [E, K, N] — expert weight matrices
        expert_counts: [E] tensor or list — number of tokens per expert

    Returns: [M_total, N]
    """
    assert (
        a.dim() == 2 and b.dim() == 3
    ), f"Expected a=[M,K] b=[E,K,N], got a={a.shape} b={b.shape}"
    a = a.contiguous()
    b = b.contiguous()

    E = b.shape[0]
    offsets, counts = _compute_offsets(expert_counts, a.device)
    max_M = int(counts.max().item()) if counts.numel() > 0 else 0

    return TritonGroupedGEMMFn.apply(a, b, counts, offsets, max_M, E)


# ============================================================================
# Reference implementation for correctness testing
# ============================================================================


def pytorch_grouped_gemm(a, b, expert_counts):
    """Reference: loop over experts, per-expert matmul."""
    E = b.shape[0]
    offsets, _ = _compute_offsets(expert_counts, a.device)
    N = b.shape[2]
    out = torch.empty(a.shape[0], N, device=a.device, dtype=a.dtype)
    for e in range(E):
        s = offsets[e].item()
        t = offsets[e + 1].item()
        if s < t:
            out[s:t] = a[s:t].to(torch.float32) @ b[e].to(torch.float32)
            out[s:t] = out[s:t].to(a.dtype)
    return out


def pytorch_grouped_gemm_simple(a, b, expert_counts):
    """Autograd-compatible reference using torch.cat for proper gradient flow."""
    E = b.shape[0]
    offsets, _ = _compute_offsets(expert_counts, a.device)
    N = b.shape[2]
    chunks = []
    for e in range(E):
        s = offsets[e].item()
        t = offsets[e + 1].item()
        if s < t:
            chunks.append(a[s:t] @ b[e])
    if chunks:
        return torch.cat(chunks, dim=0)
    return torch.empty(0, N, device=a.device, dtype=a.dtype)
