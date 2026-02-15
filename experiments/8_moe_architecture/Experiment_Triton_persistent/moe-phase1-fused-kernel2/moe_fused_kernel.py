"""
Phase 1: Fused SwiGLU Triton Kernel for MoEFFN
===============================================

Fuses gate + up projections + SwiGLU activation into a single Triton kernel.
Backward pass uses standard PyTorch ops (correct gradients guaranteed).

Forward:  h = silu(x @ W_gate) * (x @ W_up)   ← Triton kernel (fused, fast)
Backward: standard autograd                     ← PyTorch ops (correct)
Down:     out = h @ W_down                      ← PyTorch matmul (already fast)

Architecture compatibility:
- Works with null expert filtering (done before kernel)
- Works with shared expert (computed separately)
- Drop-in replacement for baseline expert loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple


# ============================================================================
# Triton Kernel: Fused Gate + Up + SwiGLU
# ============================================================================

@triton.jit
def fused_gate_up_kernel(
    x_ptr, W_gate_ptr, W_up_ptr, h_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_hm, stride_hn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: h = silu(x @ W_gate) * (x @ W_up)

    Saves 1 memory round-trip vs separate ops:
    - Baseline: write h_gate, write h_up, read both for activation
    - Fused:    compute both in registers, apply activation, write h once
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                     mask=x_mask, other=0.0)

        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_gate = tl.load(W_gate_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                         mask=w_mask, other=0.0)
        w_up = tl.load(W_up_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                       mask=w_mask, other=0.0)

        acc_gate += tl.dot(x, w_gate)
        acc_up += tl.dot(x, w_up)

    # SwiGLU fused in registers (no memory round-trip)
    h = (acc_gate * tl.sigmoid(acc_gate)) * acc_up

    h_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(h_ptr + offs_m[:, None] * stride_hm + offs_n[None, :] * stride_hn,
             h, mask=h_mask)


def _triton_fused_gate_up(x, W_gate, W_up):
    """Launch the Triton kernel. Returns h = silu(x @ W_gate) * (x @ W_up)."""
    M, K = x.shape
    _, N = W_gate.shape

    h = torch.empty((M, N), device=x.device, dtype=x.dtype)

    BLOCK_M, BLOCK_K, BLOCK_N = 64, 64, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_gate_up_kernel[grid](
        x, W_gate, W_up, h,
        M, K, N,
        x.stride(0), x.stride(1),
        W_gate.stride(0), W_gate.stride(1),
        h.stride(0), h.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
    )
    return h


# ============================================================================
# Autograd wrapper: Triton forward + PyTorch backward
# ============================================================================

class FusedGateUpFunction(torch.autograd.Function):
    """
    Custom autograd function that uses:
    - Forward:  Triton fused kernel  (fast, saves memory bandwidth)
    - Backward: PyTorch standard ops (correct gradients, already optimized)

    Activation recomputation pattern: we save (x, W_gate, W_up) and
    recompute h_gate / h_up in the backward pass instead of storing them.
    """

    @staticmethod
    def forward(ctx, x, W_gate, W_up):
        # Triton fused forward: h = silu(x @ W_gate) * (x @ W_up)
        h = _triton_fused_gate_up(x, W_gate, W_up)
        ctx.save_for_backward(x, W_gate, W_up)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        x, W_gate, W_up = ctx.saved_tensors

        # Recompute intermediates (activation-recomputation pattern)
        h_gate = x @ W_gate       # (M, N)
        h_up   = x @ W_up         # (M, N)

        sigmoid_gate = torch.sigmoid(h_gate)
        silu_gate    = h_gate * sigmoid_gate          # silu(h_gate)

        # d silu(z)/dz = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
        dsilu = sigmoid_gate * (1.0 + h_gate * (1.0 - sigmoid_gate))

        # Chain rule through SwiGLU:  h = silu(h_gate) * h_up
        grad_h_gate = grad_h * h_up * dsilu           # (M, N)
        grad_h_up   = grad_h * silu_gate              # (M, N)

        # Chain rule through matmuls
        grad_x      = grad_h_gate @ W_gate.T + grad_h_up @ W_up.T   # (M, K)
        grad_W_gate = x.T @ grad_h_gate                              # (K, N)
        grad_W_up   = x.T @ grad_h_up                                # (K, N)

        return grad_x, grad_W_gate, grad_W_up


# ============================================================================
# Public API
# ============================================================================

def fused_expert_forward(x, W_gate, W_up, W_down):
    """
    Compute one expert:  out = (silu(x @ W_gate) * (x @ W_up)) @ W_down

    Stage 1 — Triton fused kernel  (gate + up + SwiGLU)
    Stage 2 — PyTorch matmul       (down projection, already fast)
    """
    h   = FusedGateUpFunction.apply(x, W_gate, W_up)
    out = h @ W_down
    return out


# ============================================================================
# MoEFFN with Fused Kernels  (drop-in replacement for MoEFFN)
# ============================================================================

class MoEFFN_Fused(nn.Module):
    """
    Phase 1: MoEFFN with fused gate+up Triton kernel.

    Architecture unchanged:
    - Null expert support via data_sparsity
    - Shared expert (always active)
    - Same routing logic, same weights layout

    Optimization:
    - Fused gate+up kernel per expert (saves memory bandwidth)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        from moe_standalone_kaggle import MoEGate

        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        # Expert weights (same layout as baseline)
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up   = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared expert (unchanged)
        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up   = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None

    def _init_shared_weights(self):
        for m in [self.shared_gate, self.shared_up, self.shared_down]:
            m.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype

        # --- Shared expert (unchanged) ---
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # --- Routing (unchanged) ---
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()

        flat_x       = x.view(N, D)
        flat_idx     = topk_idx.view(N, K)
        flat_weight  = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        # --- Filter nulls (unchanged) ---
        real_mask = ~flat_is_null
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        real_token_indices  = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights        = flat_weight[real_mask]

        # --- Sort by expert (unchanged) ---
        sort_idx              = real_expert_indices.argsort()
        sorted_token_indices  = real_token_indices[sort_idx]
        sorted_weights        = real_weights[sort_idx]
        sorted_x              = flat_x[sorted_token_indices]
        sorted_expert_indices = real_expert_indices[sort_idx]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=E)

        # --- Process experts with fused kernel ---
        routed_out = torch.zeros_like(flat_x)

        if sorted_token_indices.numel() > 0:
            token_offset = 0
            for e in range(E):
                count = expert_counts[e].item()
                if count == 0:
                    continue

                expert_x  = sorted_x[token_offset:token_offset + count]
                expert_w  = sorted_weights[token_offset:token_offset + count]

                # ⚡ Fused kernel (Triton forward, PyTorch backward)
                expert_out = fused_expert_forward(
                    expert_x,
                    self.W_gate[e],
                    self.W_up[e],
                    self.W_down[e],
                )

                weighted_out = expert_out * expert_w.unsqueeze(1)
                idx = sorted_token_indices[token_offset:token_offset + count]
                routed_out.index_add_(0, idx, weighted_out)

                token_offset += count

        routed_out = routed_out.view(B, T, D)

        # --- Combine (unchanged) ---
        y = shared_out + routed_out
        return y, aux_loss


# ============================================================================
# MoEFFN with Batched GEMM  (the REAL optimization)
# ============================================================================

class MoEFFN_Batched(nn.Module):
    """
    MoEFFN with Padded Batched GEMM — eliminates the expert loop.

    Key insight: the baseline loops over E experts, launching 3×E separate
    matmul kernels.  Each is tiny (few tokens × d_model), wasting GPU SMs.

    This version:
    1. Pads all expert token groups to the same length
    2. Runs 3 torch.bmm calls (gate, up, down) — 3 kernel launches total
    3. Unpads and scatters results back

    Kernel launches: 3×E  →  3   (e.g. 762→3 for 254 experts)
    GPU utilization: dramatically better (all experts computed in parallel)

    Architecture: 100% identical to MoEFFN baseline
    Gradients:    100% correct (all standard PyTorch ops)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout

        from moe_standalone_kaggle import MoEGate

        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        # Expert weights — (E, D, H) layout is perfect for bmm!
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up   = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared expert (unchanged)
        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up   = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None

    def _init_shared_weights(self):
        for m in [self.shared_gate, self.shared_up, self.shared_down]:
            m.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts
        device, dtype = x.device, x.dtype

        # --- Shared expert (unchanged) ---
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # --- Routing (unchanged) ---
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()

        flat_x       = x.view(N, D)
        flat_idx     = topk_idx.view(N, K)
        flat_weight  = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        # --- Filter nulls (unchanged) ---
        real_mask = ~flat_is_null
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        real_token_indices  = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights        = flat_weight[real_mask]

        # --- Sort by expert (unchanged) ---
        sort_idx              = real_expert_indices.argsort()
        sorted_token_indices  = real_token_indices[sort_idx]
        sorted_weights        = real_weights[sort_idx]
        sorted_x              = flat_x[sorted_token_indices]
        sorted_expert_indices = real_expert_indices[sort_idx]

        expert_counts = torch.bincount(sorted_expert_indices, minlength=E)

        # ⚡ BATCHED GEMM: 3 bmm calls instead of 3×E matmuls ⚡
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)

        if sorted_token_indices.numel() > 0:
            max_count = expert_counts.max().item()

            # --- Pad expert inputs into (E, max_count, D) ---
            # Use cat+stack for clean autograd graph (no in-place ops)
            chunks = []
            offset = 0
            for e in range(E):
                c = expert_counts[e].item()
                if c > 0:
                    chunk = sorted_x[offset:offset + c]
                    if c < max_count:
                        pad = sorted_x.new_zeros(max_count - c, D)
                        chunk = torch.cat([chunk, pad], dim=0)
                    chunks.append(chunk)
                    offset += c
                else:
                    chunks.append(sorted_x.new_zeros(max_count, D))

            padded_x = torch.stack(chunks, dim=0)  # (E, max_count, D)

            # --- 3 batched matmuls (ALL experts at once!) ---
            h_gate = torch.bmm(padded_x, self.W_gate)   # (E, max_count, H)
            h_up   = torch.bmm(padded_x, self.W_up)     # (E, max_count, H)
            h      = F.silu(h_gate) * h_up               # SwiGLU activation
            if self.training and self.dropout > 0:
                h = F.dropout(h, p=self.dropout)
            out    = torch.bmm(h, self.W_down)           # (E, max_count, D)

            # --- Unpad and scatter back ---
            offset = 0
            for e in range(E):
                c = expert_counts[e].item()
                if c > 0:
                    expert_out  = out[e, :c]                      # (c, D)
                    weighted    = expert_out * sorted_weights[offset:offset + c].unsqueeze(1)
                    idx         = sorted_token_indices[offset:offset + c]
                    routed_out.index_add_(0, idx, weighted)
                    offset += c

        routed_out = routed_out.view(B, T, D)

        # --- Combine (unchanged) ---
        y = shared_out + routed_out
        return y, aux_loss


# ============================================================================
# Quick self-test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Phase 1: Fused SwiGLU Kernel - Quick Test")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - kernel requires GPU")
        exit(0)

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    M, K, N = 128, 576, 1536
    x      = torch.randn(M, K, device=device, requires_grad=True)
    W_gate = torch.randn(K, N, device=device, requires_grad=True)
    W_up   = torch.randn(K, N, device=device, requires_grad=True)
    W_down = torch.randn(N, K, device=device, requires_grad=True)

    # --- reference (separate ops) ---
    x_ref      = x.clone().detach().requires_grad_(True)
    Wg_ref     = W_gate.clone().detach().requires_grad_(True)
    Wu_ref     = W_up.clone().detach().requires_grad_(True)
    Wd_ref     = W_down.clone().detach().requires_grad_(True)

    h_gate_ref = x_ref @ Wg_ref
    h_up_ref   = x_ref @ Wu_ref
    h_ref      = F.silu(h_gate_ref) * h_up_ref
    out_ref    = h_ref @ Wd_ref
    out_ref.sum().backward()

    # --- fused ---
    out_fused = fused_expert_forward(x, W_gate, W_up, W_down)
    out_fused.sum().backward()

    # --- compare ---
    fwd_diff  = (out_ref - out_fused).abs().max().item()
    grad_x    = (x_ref.grad - x.grad).abs().max().item()
    grad_wg   = (Wg_ref.grad - W_gate.grad).abs().max().item()
    grad_wu   = (Wu_ref.grad - W_up.grad).abs().max().item()
    grad_wd   = (Wd_ref.grad - W_down.grad).abs().max().item()

    print(f"\nForward  max diff : {fwd_diff:.2e}")
    print(f"Grad x   max diff : {grad_x:.2e}")
    print(f"Grad Wg  max diff : {grad_wg:.2e}")
    print(f"Grad Wu  max diff : {grad_wu:.2e}")
    print(f"Grad Wd  max diff : {grad_wd:.2e}")

    ok = all(d < 1e-3 for d in [fwd_diff, grad_x, grad_wg, grad_wu, grad_wd])
    print(f"\n{'✅ All match!' if ok else '❌ Mismatch detected'}")
