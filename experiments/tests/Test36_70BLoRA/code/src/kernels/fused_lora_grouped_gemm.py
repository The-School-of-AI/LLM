"""
K3: Fused LoRA + Grouped GEMM for MoE expert compute.

Combines K1 (Grouped GEMM) + K2 (Fused LoRA Linear) for expert weights.

For each expert e with M_e tokens:
    output[offset_e : offset_e+M_e] = x @ W_base[e].T + (x @ A[e].T @ B[e].T) * scaling

LoRA weights are stacked along the expert dimension:
    lora_A: [E, rank, in_features]
    lora_B: [E, out_features, rank]

Memory optimization: saves lora_mid [M_total, rank] instead of base_out [M_total, out_features].

Reference: woct0rdho/transformers-qwen3-moe-fused LoraMoeFusedLinear
"""

import torch
import torch.nn.functional as F
from typing import Iterable


def _compute_offsets(expert_counts, device):
    """Compute [E+1] cumulative offsets from expert counts."""
    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.to(device=device, dtype=torch.int64).contiguous()
    else:
        counts = torch.tensor(list(expert_counts), device=device, dtype=torch.int64)
    offsets = torch.zeros(counts.shape[0] + 1, device=device, dtype=torch.int64)
    torch.cumsum(counts, dim=0, out=offsets[1:])
    return offsets, counts


class FusedLoRAGroupedGEMMFn(torch.autograd.Function):
    """
    Fused forward: for each expert e:
        out[s:t] = x[s:t] @ W_base[e].T + (x[s:t] @ A[e].T @ B[e].T) * scaling

    Memory-efficient backward: saves lora_mid [M_total, rank] not base_out.
    Uses the V1 grouped GEMM (3D grid) for the underlying matmuls.
    """

    @staticmethod
    def forward(ctx, x, W_base, lora_A, lora_B, expert_counts_tensor, offsets, max_M, E, scaling):
        """
        Args:
            x: [M_total, K] sorted tokens
            W_base: [E, K, N] expert base weights
            lora_A: [E, rank, K] LoRA down-projection per expert
            lora_B: [E, N, rank] LoRA up-projection per expert
            expert_counts_tensor: [E] tokens per expert
            offsets: [E+1] cumulative offsets
            max_M: int — max tokens for any expert
            E: int — number of experts
            scaling: float — alpha / rank
        """
        from .triton_moe_grouped_gemm import _grouped_gemm_forward

        M_total, K = x.shape
        N = W_base.shape[2]
        rank = lora_A.shape[1]

        # Base path: grouped GEMM
        base_out = _grouped_gemm_forward(x, W_base, offsets, E, max_M)

        # LoRA path: grouped GEMM for A, then grouped GEMM for B
        # x @ A[e].T: need A transposed to [E, K, rank]
        A_t = lora_A.transpose(-2, -1).contiguous()  # [E, K, rank]
        lora_mid = _grouped_gemm_forward(x, A_t, offsets, E, max_M)  # [M_total, rank]

        # lora_mid @ B[e].T: need B transposed to [E, rank, N]
        # lora_B is [E, N, rank], so B.T is [E, rank, N]
        B_t = lora_B.transpose(-2, -1).contiguous()  # [E, rank, N]
        lora_out = _grouped_gemm_forward(lora_mid, B_t, offsets, E, max_M)  # [M_total, N]

        result = base_out + lora_out * scaling

        # Save for backward — NOT base_out, NOT lora_out
        ctx.save_for_backward(x, W_base, lora_A, lora_B, lora_mid,
                              expert_counts_tensor, offsets)
        ctx.max_M = max_M
        ctx.E = E
        ctx.scaling = scaling
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x, W_base, lora_A, lora_B, lora_mid,
         counts, offsets) = ctx.saved_tensors
        max_M = ctx.max_M
        E = ctx.E
        scaling = ctx.scaling

        from .triton_moe_grouped_gemm import _grouped_gemm_forward, _grouped_gemm_dweight

        grad_output = grad_output.contiguous()
        M_total, K = x.shape
        N = W_base.shape[2]
        rank = lora_A.shape[1]

        # --- Gradient for LoRA B: dB[e] = (go * scaling)^T @ lora_mid ---
        go_scaled = (grad_output * scaling).contiguous()
        # dB[e] = go_scaled_e^T @ lora_mid_e: [N, M_e] @ [M_e, rank] = [N, rank]
        # This is _grouped_gemm_dweight with A=go_scaled (viewed as [M, N]) and dC=lora_mid
        # Actually: dB[e] = go_scaled_e^T @ lora_mid_e
        # Using dweight kernel: treats it as A^T @ C where A=[M, N], C=[M, rank], output=[N, rank]
        grad_lora_B = _grouped_gemm_dweight(
            go_scaled, lora_mid, offsets, E, N, rank, max_M, lora_B.dtype
        )  # [E, N, rank]

        # --- Gradient for LoRA A: dA[e] = ((go * scaling) @ B[e])^T @ x ---
        # grad_lora_mid[e] = go_scaled_e @ B[e]: [M_e, N] @ [N, rank] = [M_e, rank]
        # B is [E, N, rank]
        grad_lora_mid = _grouped_gemm_forward(
            go_scaled, lora_B, offsets, E, max_M
        )  # [M_total, rank]

        # dA[e] = grad_lora_mid_e^T @ x_e: [rank, M_e] @ [M_e, K] = [rank, K]
        grad_lora_A = _grouped_gemm_dweight(
            grad_lora_mid, x, offsets, E, rank, K, max_M, lora_A.dtype
        )  # [E, rank, K]

        # --- Gradient for x: dx = go @ W_base^T + grad_lora_mid @ A ---
        # dx_base[e] = go_e @ W_base[e]^T
        W_base_t = W_base.transpose(-2, -1).contiguous()  # [E, N, K]
        grad_x = _grouped_gemm_forward(grad_output, W_base_t, offsets, E, max_M)  # [M, K]

        # dx_lora[e] = grad_lora_mid_e @ A[e]: [M_e, rank] @ [rank, K] = [M_e, K]
        grad_x_lora = _grouped_gemm_forward(
            grad_lora_mid, lora_A, offsets, E, max_M
        )  # [M, K] (lora_A is [E, rank, K])
        grad_x = grad_x + grad_x_lora

        # --- Gradient for W_base: dW[e] = x_e^T @ go_e (needed if base weights are trainable) ---
        grad_W_base = _grouped_gemm_dweight(
            x, grad_output, offsets, E, K, N, max_M, W_base.dtype
        )  # [E, K, N]

        return grad_x, grad_W_base, grad_lora_A, grad_lora_B, None, None, None, None, None


def fused_lora_grouped_gemm(
    x: torch.Tensor,
    W_base: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    expert_counts,
    scaling: float,
) -> torch.Tensor:
    """
    Fused LoRA + Grouped GEMM for MoE expert compute.

    For each expert e:
        out[offset_e:offset_e+M_e] = x @ W_base[e].T + (x @ A[e].T @ B[e].T) * scaling

    Args:
        x: [M_total, K] — sorted tokens, contiguous
        W_base: [E, K, N] — expert base weight matrices (frozen during LoRA)
        lora_A: [E, rank, K] — LoRA down-projection per expert
        lora_B: [E, N, rank] — LoRA up-projection per expert
        expert_counts: [E] tensor or list — number of tokens per expert
        scaling: float — alpha / rank

    Returns: [M_total, N]
    """
    assert x.dim() == 2, f"Expected x=[M,K], got {x.shape}"
    assert W_base.dim() == 3, f"Expected W_base=[E,K,N], got {W_base.shape}"
    assert lora_A.dim() == 3, f"Expected lora_A=[E,rank,K], got {lora_A.shape}"
    assert lora_B.dim() == 3, f"Expected lora_B=[E,N,rank], got {lora_B.shape}"

    x = x.contiguous()
    W_base = W_base.contiguous()

    E = W_base.shape[0]
    offsets, counts = _compute_offsets(expert_counts, x.device)
    max_M = int(counts.max().item()) if counts.numel() > 0 else 0

    return FusedLoRAGroupedGEMMFn.apply(
        x, W_base, lora_A, lora_B, counts, offsets, max_M, E, scaling
    )


def fused_lora_gate_up_silu(
    x: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    lora_A_gate: torch.Tensor,
    lora_B_gate: torch.Tensor,
    lora_A_up: torch.Tensor,
    lora_B_up: torch.Tensor,
    expert_counts,
    scaling: float,
) -> torch.Tensor:
    """
    Fused LoRA + Gate+Up+SiLU for MoE experts.

    Computes: h = SiLU(x @ W_gate[e] + lora_gate) * (x @ W_up[e] + lora_up)

    This combines the fused gate+up+SiLU pattern with LoRA adaptation.
    Falls back to separate grouped GEMM + LoRA since the fused Triton kernel
    for gate+up doesn't support LoRA natively yet.

    Args:
        x: [M_total, K] sorted tokens
        W_gate, W_up: [E, K, H] expert gate/up weights
        lora_A_gate, lora_B_gate: [E, rank, K], [E, H, rank] gate LoRA
        lora_A_up, lora_B_up: [E, rank, K], [E, H, rank] up LoRA
        expert_counts: [E] tokens per expert
        scaling: float — alpha / rank

    Returns: [M_total, H]
    """
    # Gate path with LoRA
    gate_out = fused_lora_grouped_gemm(
        x, W_gate, lora_A_gate, lora_B_gate, expert_counts, scaling
    )

    # Up path with LoRA
    up_out = fused_lora_grouped_gemm(
        x, W_up, lora_A_up, lora_B_up, expert_counts, scaling
    )

    # Fused SiLU: gate * sigmoid(gate) * up
    h = torch.nn.functional.silu(gate_out) * up_out
    return h


# ============================================================================
# Reference implementation for correctness testing
# ============================================================================

def pytorch_fused_lora_grouped_gemm(x, W_base, lora_A, lora_B, expert_counts, scaling):
    """Reference: loop over experts, per-expert base + LoRA matmul."""
    E = W_base.shape[0]
    offsets, _ = _compute_offsets(expert_counts, x.device)
    N = W_base.shape[2]
    out = torch.empty(x.shape[0], N, device=x.device, dtype=x.dtype)
    for e in range(E):
        s = offsets[e].item()
        t = offsets[e + 1].item()
        if s < t:
            xe = x[s:t].float()
            base = xe @ W_base[e].float()  # [M_e, N]
            mid = xe @ lora_A[e].float().t()  # [M_e, rank]
            lora = mid @ lora_B[e].float().t()  # [M_e, N]
            out[s:t] = (base + lora * scaling).to(x.dtype)
    return out
