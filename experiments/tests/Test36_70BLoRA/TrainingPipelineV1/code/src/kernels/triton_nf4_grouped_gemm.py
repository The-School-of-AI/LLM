"""
K7: Fused NF4 Dequant + Grouped GEMM for QLoRA MoE Expert Compute.

Dequantizes 4-bit NF4 expert weights on-the-fly during the grouped GEMM,
never materializing the full bf16 weight tensor. This is the key memory
optimization: instead of dequantizing [E, K, N] to bf16 (6.4 GB/layer),
we dequantize tile-by-tile inside the Triton kernel.

For each expert e with M_e tokens:
    output[s:t] = x[s:t] @ dequant(W_nf4[e]).T + (x[s:t] @ A[e].T @ B[e].T) * scaling

Memory: O(BLOCK_K × BLOCK_N) dequantized tile in SRAM, not O(K × N) in HBM.

Integration with existing stack:
    - Replaces _grouped_gemm_forward in the MoE expert compute path
    - LoRA path remains unchanged (bf16 A/B matrices)
    - Backward: base weight gradient is NOT computed (frozen in QLoRA)
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    # NF4 lookup table as a compile-time constant
    # 16 values from the normal distribution quantiles
    NF4_TABLE = [
        -1.0000, -0.6962, -0.5251, -0.3949,
        -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,
         0.4407,  0.5626,  0.7230,  1.0000,
    ]

    # Shared NF4 lookup constants
    # 16 values from the normal distribution quantiles (used in kernel below)

    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        ],
        key=['K', 'N'],
    )
    @triton.jit
    def _nf4_grouped_gemm_fwd_kernel(
        # Inputs
        A_ptr,          # [M_total, K] activations (bf16)
        W_packed_ptr,   # [E, K*N//2] packed NF4 weights (uint8)
        Absmax_ptr,     # [E, num_blocks] per-block absmax scales (float32)
        C_ptr,          # [M_total, N] output (bf16)
        Offsets_ptr,    # [E+1] cumulative token offsets
        # Dimensions
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # NF4 quantization block size
        # Strides
        stride_ak, stride_an_unused,
        stride_wpe,     # stride for W_packed expert dim
        stride_ame,     # stride for absmax expert dim
        stride_cm, stride_cn,
        # Tile sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Fused NF4 dequant + grouped GEMM forward kernel.

        For each expert e, computes: C[s:t] = A[s:t] @ dequant(W_nf4[e]).T
        where dequantization happens tile-by-tile in SRAM using tl.dot.

        Weight layout: W_packed stores weights in row-major [K, N] order,
        packed as uint8 with 2 values per byte along the N dimension.
        For a [K, N] weight matrix, packed shape is [K, N//2].
        Element [k, n] maps to byte [k, n//2], nibble (n % 2).
        """
        pid_e = tl.program_id(0)   # expert index
        pid_m = tl.program_id(1)   # M-tile index
        pid_n = tl.program_id(2)   # N-tile index

        # Expert token range
        start = tl.load(Offsets_ptr + pid_e)
        end = tl.load(Offsets_ptr + pid_e + 1)
        M_e = end - start

        if pid_m * BLOCK_M >= M_e:
            return

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        # Base pointers for this expert
        a_base = A_ptr + start * stride_ak
        w_base = W_packed_ptr + pid_e * stride_wpe
        am_base = Absmax_ptr + pid_e * stride_ame
        c_base = C_ptr + start * stride_cm

        # Precompute N-dimension byte offsets and nibble selection
        half_N = N // 2
        byte_offs_n = offs_n // 2           # [BLOCK_N]
        is_high_nibble = (offs_n % 2) == 0  # [BLOCK_N]
        mask_n = offs_n < N
        mask_m = offs_m < M_e

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k  # [BLOCK_K]
            mask_k = k_offs < K

            # Load A tile: [BLOCK_M, BLOCK_K]
            a_ptrs = a_base + offs_m[:, None] * stride_ak + k_offs[None, :]
            a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            # Load packed W tile: [BLOCK_K, BLOCK_N//2] bytes
            # Each byte at [k, n//2] contains two 4-bit values for columns n and n+1
            w_byte_ptrs = w_base + k_offs[:, None] * half_N + byte_offs_n[None, :]
            packed_tile = tl.load(w_byte_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0).to(tl.uint8)

            # Extract 4-bit indices: select high or low nibble per column
            high = (packed_tile >> 4) & 0x0F
            low = packed_tile & 0x0F
            nf4_idx = tl.where(is_high_nibble[None, :], high, low).to(tl.int32)

            # NF4 lookup via cascading selects → [BLOCK_K, BLOCK_N] float values
            val = tl.where(nf4_idx == 0, -1.0,
                  tl.where(nf4_idx == 1, -0.6962,
                  tl.where(nf4_idx == 2, -0.5251,
                  tl.where(nf4_idx == 3, -0.3949,
                  tl.where(nf4_idx == 4, -0.2844,
                  tl.where(nf4_idx == 5, -0.1848,
                  tl.where(nf4_idx == 6, -0.0911,
                  tl.where(nf4_idx == 7,  0.0,
                  tl.where(nf4_idx == 8,  0.0796,
                  tl.where(nf4_idx == 9,  0.1609,
                  tl.where(nf4_idx == 10, 0.2461,
                  tl.where(nf4_idx == 11, 0.3379,
                  tl.where(nf4_idx == 12, 0.4407,
                  tl.where(nf4_idx == 13, 0.5626,
                  tl.where(nf4_idx == 14, 0.7230,
                           1.0)))))))))))))))

            # Load absmax scales for this tile: [BLOCK_K, BLOCK_N]
            # linear_idx = k * N + n, block_idx = linear_idx // BLOCK_SIZE
            linear_idx = k_offs[:, None] * N + offs_n[None, :]
            block_idx = linear_idx // BLOCK_SIZE
            absmax_vals = tl.load(am_base + block_idx,
                                  mask=mask_k[:, None] & mask_n[None, :], other=1.0)

            # Dequantize: val * absmax → [BLOCK_K, BLOCK_N] float32 weight tile
            w_tile = (val * absmax_vals).to(tl.float32)

            # Tiled matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
            acc += tl.dot(a_tile.to(tl.float32), w_tile)

        # Store result
        c_ptrs = c_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask_c = (offs_m[:, None] < M_e) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=mask_c)


# ============================================================================
# PyTorch wrapper functions
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


def _nf4_grouped_gemm_forward(
    a: torch.Tensor,
    w_packed: torch.Tensor,
    absmax: torch.Tensor,
    offsets: torch.Tensor,
    E: int,
    max_M: int,
    K: int,
    N: int,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Fused NF4 dequant + grouped GEMM forward.

    Args:
        a: [M_total, K] activations
        w_packed: [E, K*N//2] packed NF4 weights
        absmax: [E, num_blocks] per-block absmax scales
        offsets: [E+1] cumulative offsets
        E: number of experts
        max_M: max tokens for any expert
        K: input dimension
        N: output dimension
        block_size: NF4 quantization block size

    Returns: [M_total, N]
    """
    M_total = a.shape[0]
    c = torch.empty(M_total, N, device=a.device, dtype=a.dtype)
    if M_total == 0:
        return c

    def grid(meta):
        mm_tiles = (max_M + meta['BLOCK_M'] - 1) // meta['BLOCK_M']
        nn_tiles = (N + meta['BLOCK_N'] - 1) // meta['BLOCK_N']
        return (E, mm_tiles, nn_tiles)

    _nf4_grouped_gemm_fwd_kernel[grid](
        a, w_packed, absmax, c,
        offsets,
        K, N, block_size,
        a.stride(0), a.stride(1),
        w_packed.stride(0),
        absmax.stride(0),
        c.stride(0), c.stride(1),
    )
    return c


class NF4GroupedGEMMFn(torch.autograd.Function):
    """
    Fused NF4 dequant + grouped GEMM with LoRA.

    Forward: out = x @ dequant(W_nf4[e]).T + (x @ A[e].T @ B[e].T) * scaling
    Backward: NO gradient for W_nf4 (frozen base weights in QLoRA).
              Only computes gradients for x, lora_A, lora_B.
    """

    @staticmethod
    def forward(ctx, x, w_packed, absmax, lora_A, lora_B,
                expert_counts_tensor, offsets, max_M, E, K, N,
                block_size, scaling):
        from .triton_moe_grouped_gemm import _grouped_gemm_forward

        M_total = x.shape[0]

        # Base path: NF4 dequant + grouped GEMM
        base_out = _nf4_grouped_gemm_forward(
            x, w_packed, absmax, offsets, E, max_M, K, N, block_size
        )

        # LoRA path: standard bf16 grouped GEMM (reuse existing kernel)
        rank = lora_A.shape[1]
        A_t = lora_A.transpose(-2, -1).contiguous()  # [E, K, rank]
        lora_mid = _grouped_gemm_forward(x, A_t, offsets, E, max_M)  # [M_total, rank]

        B_t = lora_B.transpose(-2, -1).contiguous()  # [E, rank, N]
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)  # [M_total, N]

        result = base_out + lora_out * scaling

        # Save for backward — NO base weights saved (frozen, no grad needed)
        # We save w_packed/absmax for dx computation (need to dequant again in backward)
        ctx.save_for_backward(x, w_packed, absmax, lora_A, lora_B, lora_mid,
                              expert_counts_tensor, offsets)
        ctx.max_M = max_M
        ctx.E = E
        ctx.K = K
        ctx.N = N
        ctx.block_size = block_size
        ctx.scaling = scaling
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x, w_packed, absmax, lora_A, lora_B, lora_mid,
         counts, offsets) = ctx.saved_tensors
        max_M = ctx.max_M
        E = ctx.E
        K = ctx.K
        N = ctx.N
        block_size = ctx.block_size
        scaling = ctx.scaling

        from .triton_moe_grouped_gemm import _grouped_gemm_forward, _grouped_gemm_dweight

        grad_output = grad_output.contiguous()
        rank = lora_A.shape[1]

        # --- LoRA B gradient: dB[e] = (go * scaling)^T @ lora_mid ---
        go_scaled = (grad_output * scaling).contiguous()
        grad_lora_B = _grouped_gemm_dweight(
            go_scaled, lora_mid, offsets, E, N, rank, max_M, lora_B.dtype
        )

        # --- LoRA A gradient: dA[e] = ((go * scaling) @ B[e])^T @ x ---
        grad_lora_mid = _grouped_gemm_forward(
            go_scaled, lora_B, offsets, E, max_M
        )
        grad_lora_A = _grouped_gemm_dweight(
            grad_lora_mid, x, offsets, E, rank, K, max_M, lora_A.dtype
        )

        # --- Input gradient: dx = go @ dequant(W_nf4)^T + grad_lora_mid @ A ---
        # For dx_base, we need W^T. We dequant W and transpose.
        # Batched vectorized dequantization — all experts at once, no Python loop.
        from .nf4_quantize import NF4_LEVELS

        nf4_levels = NF4_LEVELS.to(x.device)
        numel = K * N

        # Unpack all experts at once: w_packed is [E, K*N//2] uint8
        # Use int16 intermediates instead of int64 to save memory
        high = ((w_packed >> 4) & 0x0F).to(torch.int16)  # [E, K*N//2]
        low = (w_packed & 0x0F).to(torch.int16)           # [E, K*N//2]
        # Interleave high/low to get [E, K*N] indices
        indices = torch.stack([high, low], dim=2).reshape(E, -1)  # [E, K*N]
        del high, low
        indices = indices[:, :numel]            # trim any padding

        # Lookup NF4 values for all experts at once
        values = nf4_levels[indices.long()].float()    # [E, K*N]
        del indices

        # Apply per-block absmax scaling
        pad = (block_size - numel % block_size) % block_size
        padded_N_total = numel + pad
        if pad > 0:
            values = F.pad(values, (0, pad))    # [E, padded_N_total]
        num_blocks = padded_N_total // block_size
        values = values.view(E, num_blocks, block_size)
        values = values * absmax[:, :num_blocks].unsqueeze(2)  # [E, num_blocks, block_size]
        values = values.reshape(E, -1)[:, :numel]  # [E, K*N]

        W_full_t = values.view(E, K, N).transpose(-2, -1).contiguous().to(x.dtype)  # [E, N, K]
        del values  # free intermediates

        grad_x = _grouped_gemm_forward(grad_output, W_full_t, offsets, E, max_M)

        # Add LoRA contribution to dx
        grad_x_lora = _grouped_gemm_forward(
            grad_lora_mid, lora_A, offsets, E, max_M
        )
        grad_x = grad_x + grad_x_lora

        # No gradient for w_packed, absmax (frozen base weights)
        return grad_x, None, None, grad_lora_A, grad_lora_B, None, None, None, None, None, None, None, None


def nf4_lora_grouped_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    absmax: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_counts,
    K: int,
    N: int,
    scaling: float,
    block_size: int = 64,
) -> torch.Tensor:
    """
    Fused NF4 dequant + LoRA grouped GEMM for MoE expert compute.

    For each expert e:
        out[s:t] = x @ dequant(W_nf4[e]).T + (x @ A[e].T @ B[e].T) * scaling

    Base weights are in NF4 (4-bit), LoRA weights in bf16.
    No gradient computed for base weights (QLoRA frozen).

    Args:
        x: [M_total, K] sorted tokens
        w_packed: [E, K*N//2] packed NF4 expert weights
        absmax: [E, num_blocks] per-block absmax scales
        lora_A: [E, rank, K] LoRA down-projection
        lora_B: [E, N, rank] LoRA up-projection
        expert_counts: [E] tokens per expert
        K: input dimension
        N: output dimension
        scaling: alpha / rank
        block_size: NF4 quantization block size

    Returns: [M_total, N]
    """
    E = lora_A.shape[0]
    offsets, counts = _compute_offsets(expert_counts, x.device)
    max_M = int(counts.max().item()) if counts.numel() > 0 else 0

    return NF4GroupedGEMMFn.apply(
        x, w_packed, absmax, lora_A, lora_B,
        counts, offsets, max_M, E, K, N, block_size, scaling,
    )


# ============================================================================
# PyTorch reference implementation for correctness testing
# ============================================================================

def pytorch_nf4_lora_grouped_gemm(
    x, w_packed, absmax, lora_A, lora_B, expert_counts,
    K, N, scaling, block_size=64,
):
    """Reference: loop over experts, dequant + base + LoRA matmul."""
    from .nf4_quantize import NF4_LEVELS, _dequantize_block_nf4

    E = lora_A.shape[0]
    offsets, _ = _compute_offsets(expert_counts, x.device)
    nf4_levels = NF4_LEVELS.to(x.device)

    out = torch.empty(x.shape[0], N, device=x.device, dtype=x.dtype)
    for e in range(E):
        s = offsets[e].item()
        t = offsets[e + 1].item()
        if s < t:
            xe = x[s:t].float()
            # Dequantize base weight
            numel = K * N
            W_e = _dequantize_block_nf4(
                w_packed[e], absmax[e], numel, block_size, nf4_levels, torch.float32
            ).view(K, N)
            base = xe @ W_e
            mid = xe @ lora_A[e].float().t()
            lora = mid @ lora_B[e].float().t()
            out[s:t] = (base + lora * scaling).to(x.dtype)
    return out
