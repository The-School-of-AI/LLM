"""
Mixture of Experts FFN with Null Experts
==========================================

Implementation matching Test_Code/model_1b.py lines 905-1057.

For the dense 1B model, num_experts=0 so MoEFFN degenerates to just
the shared expert (functionally equivalent to SwiGLUFFN(intermediate=2048)).
The structure supports future MoE scaling.

Components:
- MoEGate: Router with null experts for data sparsity
- MoEFFN: Shared expert (always active) + routed experts
- LightningMLP: Convenience wrapper

Note: Forward returns (output, aux_loss) tuple, unlike SwiGLUFFN which
returns just output. The MHCSublayerV2 handles both cases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    """
    Router gate for MoE with null experts.

    Null experts absorb tokens that don't match real experts,
    implementing data sparsity (rho parameter).
    """

    def __init__(
        self, d_model: int, num_experts: int, top_k: int, data_sparsity: float = 0.5
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.data_sparsity = data_sparsity

        self.num_null_copies = (
            int(num_experts * (1 - data_sparsity) / data_sparsity)
            if data_sparsity > 0
            else num_experts
        )
        self.total_slots = num_experts + self.num_null_copies

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.logit_bias = nn.Parameter(torch.zeros(num_experts))
        self.null_logit = nn.Parameter(torch.tensor(0.0))

        self.gate.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        real_logits = self.gate(x) + self.logit_bias
        null_logits = (
            self.null_logit.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_null_copies)
        )
        logits = torch.cat([real_logits, null_logits], dim=-1)

        probs = F.softmax(logits, dim=-1)
        topk_weight, topk_idx = torch.topk(probs, self.top_k, dim=-1)

        is_null = topk_idx >= self.num_experts
        real_weights = topk_weight * (~is_null).float()
        weight_sum = real_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        topk_weight = real_weights / weight_sum

        # Auxiliary losses
        P = probs.mean(dim=(0, 1))
        idx_flat = topk_idx.view(-1)
        counts = torch.bincount(idx_flat, minlength=self.total_slots).float()
        f = counts / (B * T)
        L_bal = self.total_slots * torch.sum(f * P)

        lse = torch.logsumexp(logits, dim=-1)
        L_z = (lse**2).mean()

        aux_loss = 2e-2 * L_bal + 1e-3 * L_z

        return topk_idx, topk_weight, is_null, aux_loss


class MoEFFN(nn.Module):
    """
    MoE FFN with null experts (batched tensor implementation).

    When num_experts=0, operates as a pure shared expert (dense FFN).
    Shared expert is always active regardless of routing.
    """

    def __init__(
        self,
        d_model: int,
        shared_d_hidden: int,
        expert_d_hidden: int = None,
        num_experts: int = 0,
        top_k: int = 0,
        dropout: float = 0.0,
        data_sparsity: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = shared_d_hidden  # Backward-compatible alias
        self.shared_d_hidden = shared_d_hidden
        self.expert_d_hidden = (
            shared_d_hidden if expert_d_hidden is None else int(expert_d_hidden)
        )
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout = dropout
        self.has_routed_experts = num_experts > 0 and top_k > 0

        # Routed experts (only if num_experts > 0)
        if self.has_routed_experts:
            self.gate = MoEGate(
                d_model, num_experts, top_k, data_sparsity=data_sparsity
            )
            self.W_gate = nn.Parameter(
                torch.randn(num_experts, d_model, self.expert_d_hidden) * 0.02
            )
            self.W_up = nn.Parameter(
                torch.randn(num_experts, d_model, self.expert_d_hidden) * 0.02
            )
            self.W_down = nn.Parameter(
                torch.randn(num_experts, self.expert_d_hidden, d_model) * 0.02
            )

        # Shared Expert (always active)
        self.shared_gate = nn.Linear(d_model, self.shared_d_hidden, bias=False)
        self.shared_up = nn.Linear(d_model, self.shared_d_hidden, bias=False)
        self.shared_down = nn.Linear(self.shared_d_hidden, d_model, bias=False)
        self._init_shared_weights()

        self.last_indices = None

    def _init_shared_weights(self):
        for module in [self.shared_gate, self.shared_up, self.shared_down]:
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Input tensor (B, T, d_model)

        Returns:
            output: (B, T, d_model)
            aux_loss: Scalar auxiliary loss
        """
        B, T, D = x.shape
        device, dtype = x.device, x.dtype

        # Shared expert (always active)
        shared_h = F.silu(self.shared_gate(x)) * self.shared_up(x)
        if self.training and self.dropout > 0:
            shared_h = F.dropout(shared_h, p=self.dropout)
        shared_out = self.shared_down(shared_h)

        # Dense model (no routed experts)
        if not self.has_routed_experts:
            aux_loss = x.new_zeros((), dtype=torch.float32)
            return shared_out, aux_loss

        # Routed experts
        N = B * T
        K = self.top_k
        E = self.num_experts

        topk_idx, topk_weight, is_null, aux_loss = self.gate(x)
        self.last_indices = topk_idx.detach().clone()

        flat_x = x.view(N, D)
        flat_idx = topk_idx.view(N, K)
        flat_weight = topk_weight.view(N, K)
        flat_is_null = is_null.view(N, K)

        real_mask = ~flat_is_null
        token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K)

        real_token_indices = token_indices[real_mask]
        real_expert_indices = flat_idx[real_mask]
        real_weights = flat_weight[real_mask]

        sort_idx = real_expert_indices.argsort()
        sorted_token_indices = real_token_indices[sort_idx]
        sorted_weights = real_weights[sort_idx]
        sorted_x = flat_x[sorted_token_indices]

        expert_counts = torch.bincount(real_expert_indices, minlength=E)
        offsets = expert_counts.cumsum(0)

        num_real_assignments = sorted_token_indices.size(0)
        sorted_out = torch.empty(num_real_assignments, D, device=device, dtype=dtype)

        start = 0
        for e in range(E):
            end = offsets[e].item()
            if end > start:
                chunk_x = sorted_x[start:end]
                h = F.silu(chunk_x @ self.W_gate[e]) * (chunk_x @ self.W_up[e])
                if self.training and self.dropout > 0:
                    h = F.dropout(h, p=self.dropout)
                sorted_out[start:end] = h @ self.W_down[e]
            start = end

        weighted_out = sorted_out * sorted_weights.unsqueeze(-1)
        routed_out = torch.zeros(N, D, device=device, dtype=dtype)
        routed_out.scatter_add_(
            0, sorted_token_indices.unsqueeze(-1).expand(-1, D), weighted_out
        )

        y = shared_out + routed_out.view(B, T, D)
        return y, aux_loss


class LightningMLP(nn.Module):
    """
    MLP wrapper using MoEFFN.

    For dense models (num_experts=0), this is functionally equivalent
    to SwiGLUFFN with intermediate_size.

    Forward returns (output, aux_loss) tuple.
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_experts=0,
        num_shared_experts=1,
        top_k=0,
        data_sparsity=0.5,
        expert_intermediate_size=None,
    ):
        super().__init__()
        self.moe = MoEFFN(
            d_model=hidden_size,
            shared_d_hidden=intermediate_size,
            expert_d_hidden=expert_intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            dropout=0.0,
            data_sparsity=data_sparsity,
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, hidden_size)
        Returns:
            (output, aux_loss) tuple
        """
        return self.moe(x)
