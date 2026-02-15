"""
Standalone MoE Module for Kaggle - No External Dependencies
Extracted from model_gated_multitoken.py for easy Kaggle upload

Upload this single file to Kaggle and import:
    from moe_standalone_kaggle import MoEFFN, MoEGate, LlamaMLP

Compatible with both baseline and Triton-optimized versions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# MoE Implementation - Standalone (No Fourier Dependencies)
# ============================================================================

class MoEGate(nn.Module):
    """Router gate for MoE with null experts for data sparsity."""
    def __init__(self, d_model: int, num_experts: int, top_k: int, data_sparsity: float = 0.5):
        super().__init__()
        self.num_experts = num_experts  # N real experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity  # ρ (target data sparsity)

        # Calculate number of null expert copies: M = N · (1-ρ)/ρ
        # For ρ=0.5, N=8: M = 8 · 0.5/0.5 = 8 null copies
        self.num_null_copies = int(num_experts * (1 - data_sparsity) / data_sparsity)
        self.total_slots = num_experts + self.num_null_copies  # N + M

        # Gate for REAL experts only
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))

        # Single NULL expert logit (will be duplicated M times)
        # Initialize to 0 to start balanced
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        # Init gate to small values (matches Deepscreen exactly)
        self.gate.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        Returns:
            topk_idx: (B, T, K) - indices in range [0, N+M)
            topk_weight: (B, T, K) - renormalized weights (sum to 1)
            is_null: (B, T, K) - boolean mask indicating null expert selection
            aux_loss: scalar - routing loss
        """
        B, T, D = x.shape

        # 1. Compute logits for real experts: (B, T, N)
        real_logits = self.gate(x) + self.logit_bias

        # 2. Duplicate null logit M times: (B, T, M)
        null_logits = self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)

        # 3. Concatenate: (B, T, N+M)
        logits = torch.cat([real_logits, null_logits], dim=-1)

        # 4. Softmax routing (Paper Requirement)
        probs = F.softmax(logits, dim=-1)

        # 5. Select top-K from N+M slots
        topk_weight, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        # 6. Identify null expert selections (indices >= N)
        is_null = topk_idx >= self.num_experts

        # 7. Renormalize weights over ONLY real experts
        # Zero out null weights, then renormalize
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # 8. Compute Auxiliary Losses (Paper Eq 6 & 7)
        # Global Load Balancing Loss (Eq 6)
        # P_i: average routing probability for slot i
        P = probs.mean(dim=(0, 1))  # (N+M,)

        # f_i: fraction of tokens routed to slot i
        # Flatten topk_idx to count selections
        idx_flat = topk_idx.view(-1)
        counts = torch.bincount(idx_flat, minlength=self.total_slots).float()
        f = counts / (B * T)

        L_bal = self.total_slots * torch.sum(f * P)

        # Z-Loss (Eq 7)
        lse = torch.logsumexp(logits, dim=-1)
        L_z = (lse ** 2).mean()

        # Combine losses
        # Weights from paper: 2e-2 for Bal, 1e-3 for Z-Loss
        aux_loss = 2e-2 * L_bal + 1e-3 * L_z

        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN with null experts for data sparsity (batched tensor implementation).

    Key features:
    - Expert weights stored as batched 3D tensors (not separate nn.Linear modules)
    - Direct matrix multiplication: chunk_x @ self.W_gate[e]
    - Null experts: zero-compute slots that skip processing entirely
    - Identical gradient flow and numerical characteristics as Deepscreen
    """
    def __init__(self, d_model: int, d_hidden: int, num_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.0, data_sparsity: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts  # Only REAL experts
        self.top_k = top_k
        self.dropout = dropout

        # Gate with null experts
        self.gate = MoEGate(d_model, num_experts, top_k, data_sparsity=data_sparsity)

        # Expert weights for REAL experts only (no weights for null)
        # This is the key difference from ModuleList approach!
        self.W_gate = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_up = nn.Parameter(torch.randn(num_experts, d_model, d_hidden) * 0.02)
        self.W_down = nn.Parameter(torch.randn(num_experts, d_hidden, d_model) * 0.02)

        # Shared Expert (1 shared expert, always active)
        self.shared_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, d_hidden, bias=False)
        self.shared_down = nn.Linear(d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None  # For balancing

    def _init_shared_weights(self):
        """Initialize shared expert weights to std=0.02 (matches Deepscreen)."""
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        K = self.top_k
        E = self.num_experts  # Only real experts
        device, dtype = x.device, x.dtype

        # 1. Shared Expert Path (always active for all tokens)
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # 2. Routed Experts Path with NULL expert handling
        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()  # Cache for balancer

        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        # 3. Filter out null expert assignments
        # Create mask for real expert assignments
        real_mask = ~flat_is_null  # (N, K)

        # Flatten and filter
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        # Only keep real expert assignments
        real_token_indices = token_indices[real_mask]  # (num_real_assignments,)
        real_expert_indices = flat_idx[real_mask]  # (num_real_assignments,)
        real_weights = flat_weight[real_mask]  # (num_real_assignments,)

        # 4. Sort by expert for vectorized computation
        sort_idx = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]

        expert_counts = torch.bincount(real_expert_indices, minlength=E)
        offsets = expert_counts.cumsum(0)

        # 5. Process each REAL expert's chunk
        num_real_assignments = sorted_token_indices.size(0)
        sorted_out = torch.empty(num_real_assignments, D, device=device, dtype=dtype)

        start = 0
        for e in range(E):
            end = offsets[e].item()
            if end > start:
                chunk_x = sorted_x[start:end]
                # Expert SwiGLU with DIRECT MATMUL (matches Deepscreen exactly)
                h = F.silu(chunk_x @ self.W_gate[e]) * (chunk_x @ self.W_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.W_down[e]
            start = end

        # 6. Scatter back (only real expert outputs, null contributes 0)
        weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out)

        y = shared_out + routed_out.view(B, T, D)
        return y, aux_loss


class LlamaMLP(nn.Module):
    """MLP wrapper using MoEFFN with null experts for data sparsity."""
    def __init__(self, hidden_size, intermediate_size, num_experts, num_shared_experts, top_k,
                 data_sparsity=0.5):
        super().__init__()
        # Note: num_shared_experts is always 1 in Deepscreen, handled internally by MoEFFN
        self.moe = MoEFFN(
            d_model=hidden_size,
            d_hidden=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=0.0,
            data_sparsity=data_sparsity
        )

    def forward(self, x):
        return self.moe(x)


# ============================================================================
# Quick Test
# ============================================================================

if __name__ == "__main__":
    print("Testing standalone MoE module...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create MoE
    moe = MoEFFN(d_model=576, d_hidden=1536, num_experts=8, top_k=2).to(device)
    
    # Test forward pass
    x = torch.randn(2, 128, 576, device=device)
    out, aux_loss = moe(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Aux loss: {aux_loss.item():.4f}")
    print("✅ Standalone MoE module works!")
