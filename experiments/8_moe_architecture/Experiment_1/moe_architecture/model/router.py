"""
GSA Gated Lightning Router
==========================

Multi-head router with sigmoid scoring inspired by Gated Sparse Attention paper.

Key Innovations (from arXiv:2601.15305v1):
1. Multi-head routing (like indexer heads H_I)
2. Sigmoid activations (bounded scores, no forced competition)
3. Query-dependent head weights
4. Adaptive top-k based on score variance
5. Bias-only load balancing (loss-free)

Formula (from paper Equation 7):
    I_{t,s} = Σⱼ σ(h_t W_j^Iw) · σ(q_{t,j}^I · k_s^I + b_j^I)

Applied to MoE:
    affinity_i = Σⱼ σ(x W_j^weight) · σ(q_j · expert_key_i + b_j)

This provides bounded scores in (0, H_I) that are easier to balance
than softmax-based routing which forces competition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import math

from model.config import MoEModelConfig, RouterConfig


class GSARouter(nn.Module):
    """
    GSA-style Gated Lightning Router for MoE.
    
    Architecture:
    1. Query projection: x → q (low-dim, per head)
    2. Head weight projection: x → w (query-dependent importance)
    3. Expert keys: learnable centroids for each expert
    4. Affinity: Σⱼ wⱼ · σ(qⱼ · expert_key + bⱼ)
    5. Bias adjustment for load balancing
    6. Top-k selection with gating from original scores
    
    Args:
        config: Model configuration with router parameters
    """
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        self.router_config = config.router
        
        # Total experts in routing pool (routed + null, NOT shared)
        self.num_routable = config.num_routed_experts + config.num_null_experts
        
        # Router dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = self.router_config.num_router_heads
        self.head_dim = self.router_config.router_dim
        
        # ============================================================
        # Router Projections (GSA-style multi-head)
        # ============================================================
        
        # Query projection: h_t → q_{t,j} for each head
        # Shape: [hidden_size, num_heads * head_dim]
        self.query_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False
        )
        
        # Head weight projection: h_t → w_{t,j} (query-dependent head importance)
        # Shape: [hidden_size, num_heads]
        self.head_weight_proj = nn.Linear(
            self.hidden_size,
            self.num_heads,
            bias=True
        )
        
        # Expert keys (learnable centroids)
        # Unlike attention where keys come from tokens,
        # in MoE the "keys" are learnable expert representations
        self.expert_keys = nn.Parameter(
            torch.empty(self.num_routable, self.head_dim)
        )
        
        # Per-head learnable bias (threshold)
        # b_j in σ(q · k + b)
        self.head_bias = nn.Parameter(torch.zeros(self.num_heads))
        
        # ============================================================
        # Load Balancing (Loss-Free via Bias)
        # ============================================================
        
        # Expert-wise bias for load balancing (NOT learned via gradients)
        # Updated algorithmically based on utilization
        self.register_buffer('expert_bias', torch.zeros(self.num_routable))
        
        # Initialize null expert bias slightly higher
        if config.num_null_experts > 0:
            null_start = config.num_routed_experts
            with torch.no_grad():
                self.expert_bias[null_start:] = self.router_config.null_bias_init
        
        # ============================================================
        # Adaptive Top-K (GSA-style)
        # ============================================================
        
        # EMA of score variance for adaptive k
        self.register_buffer('variance_ema', torch.tensor(1.0))
        
        # ============================================================
        # Tracking & Telemetry
        # ============================================================
        
        # Expert usage counts (reset after each bias update)
        self.register_buffer('expert_counts', torch.zeros(self.num_routable))
        self.register_buffer('total_tokens', torch.tensor(0, dtype=torch.long))
        
        # Null expert indices for telemetry
        self.null_indices = set(range(config.num_routed_experts, self.num_routable))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize router weights."""
        # Query projection (similar to attention init)
        nn.init.normal_(self.query_proj.weight, std=0.02)
        
        # Head weight projection
        nn.init.normal_(self.head_weight_proj.weight, std=0.02)
        nn.init.zeros_(self.head_weight_proj.bias)
        
        # Expert keys (small random init)
        nn.init.normal_(self.expert_keys, std=0.02)
        
        # Head bias (start at zero for σ(·) ≈ 0.5)
        nn.init.zeros_(self.head_bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Route tokens to experts using GSA-style gated indexer.
        
        Args:
            hidden_states: [batch, seq, hidden_size]
            return_all_scores: Return full affinity matrix for analysis
            
        Returns:
            expert_indices: [batch, seq, top_k] - Selected expert IDs
            gating_weights: [batch, seq, top_k] - Normalized weights
            aux_info: Dict with routing statistics
        """
        batch_size, seq_len, hidden = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # ============================================================
        # Step 1: Compute query-dependent head weights
        # w_j = σ(h_t W_j^Iw) ∈ (0, 1) for each head
        # ============================================================
        head_weights = torch.sigmoid(self.head_weight_proj(hidden_states))
        # Shape: [batch, seq, num_heads]
        
        # ============================================================
        # Step 2: Project to low-dim query space
        # q_{t,j} = h_t W_j^Iq
        # ============================================================
        queries = self.query_proj(hidden_states)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Shape: [batch, seq, num_heads, head_dim]
        
        # ============================================================
        # Step 3: Compute per-head affinity with expert keys
        # score_j = σ(q_j · expert_key + b_j)
        # ============================================================
        
        # Reshape for efficient batched computation
        queries_flat = queries.view(batch_size * seq_len, self.num_heads, self.head_dim)
        # [batch*seq, num_heads, head_dim]
        
        # Dot product with all expert keys
        # queries_flat: [batch*seq, num_heads, head_dim]
        # expert_keys: [num_experts, head_dim]
        dot_products = torch.einsum('bhd,ed->bhe', queries_flat, self.expert_keys)
        # Shape: [batch*seq, num_heads, num_experts]
        
        # Add per-head bias and apply sigmoid
        # This gives bounded scores in (0, 1) per head
        head_scores = torch.sigmoid(
            dot_products + self.head_bias.view(1, self.num_heads, 1)
        )
        # Shape: [batch*seq, num_heads, num_experts]
        
        # ============================================================
        # Step 4: Aggregate across heads with query-dependent weights
        # affinity_i = Σⱼ w_j · score_{j,i}
        # ============================================================
        head_weights_flat = head_weights.view(batch_size * seq_len, self.num_heads, 1)
        aggregated_scores = (head_weights_flat * head_scores).sum(dim=1)
        # Shape: [batch*seq, num_experts]
        
        # Reshape back
        aggregated_scores = aggregated_scores.view(batch_size, seq_len, self.num_routable)
        # Scores are bounded in (0, num_heads) - key GSA insight!
        
        # ============================================================
        # Step 5: Add expert bias for load balancing
        # adjusted = affinity + bias
        # ============================================================
        adjusted_scores = aggregated_scores + self.expert_bias
        
        # ============================================================
        # Step 6: Compute adaptive top-k (GSA adaptive sparsity)
        # k = clamp(k_base × Var / V̄, k_min, k_max)
        # ============================================================
        if self.router_config.use_adaptive_top_k and self.training:
            top_k = self._compute_adaptive_k(aggregated_scores)
        else:
            top_k = self.router_config.top_k
        
        # ============================================================
        # Step 7: Select top-k experts using adjusted scores
        # ============================================================
        top_k_adjusted, top_k_indices = torch.topk(
            adjusted_scores, k=top_k, dim=-1
        )
        # Shape: [batch, seq, top_k]
        
        # ============================================================
        # Step 8: Compute gating weights from ORIGINAL scores
        # Key insight: Bias affects SELECTION, not CONTRIBUTION
        # ============================================================
        original_scores = aggregated_scores.gather(-1, top_k_indices)
        
        # Normalize to get gating weights that sum to 1
        gating_weights = original_scores / (
            original_scores.sum(dim=-1, keepdim=True) + 1e-9
        )
        # Shape: [batch, seq, top_k]
        
        # ============================================================
        # Step 9: Update tracking for load balancing
        # ============================================================
        if self.training:
            self._update_tracking(top_k_indices)
        
        # ============================================================
        # Prepare auxiliary info
        # ============================================================
        aux_info = {
            'top_k_used': top_k,
            'score_mean': aggregated_scores.mean().item(),
            'score_std': aggregated_scores.std().item(),
            'score_max': aggregated_scores.max().item(),
            'score_min': aggregated_scores.min().item(),
            'bias_mean': self.expert_bias.mean().item(),
            'bias_range': (
                self.expert_bias.min().item(),
                self.expert_bias.max().item()
            ),
            'head_weight_mean': head_weights.mean().item(),
        }
        
        if return_all_scores:
            aux_info['all_scores'] = aggregated_scores
            aux_info['adjusted_scores'] = adjusted_scores
            aux_info['head_scores'] = head_scores.view(
                batch_size, seq_len, self.num_heads, self.num_routable
            )
        
        return top_k_indices, gating_weights, aux_info
    
    def _compute_adaptive_k(self, scores: torch.Tensor) -> int:
        """
        Compute adaptive top-k based on score variance.
        
        From GSA paper Equation 8:
            k_t = clamp(k_base × Var(I) / V̄, k_min, k_max)
        
        High variance → confident → can use fewer experts
        Low variance → uncertain → use more experts
        
        Note: For MoE we use inverse relationship - high confidence
        means we know which experts to use, so we can be sparse.
        """
        # Compute current variance
        current_var = scores.var().detach()
        
        # Update EMA
        with torch.no_grad():
            self.variance_ema.copy_(
                self.router_config.variance_ema_decay * self.variance_ema +
                (1 - self.router_config.variance_ema_decay) * current_var
            )
        
        # Compute ratio
        ratio = current_var / (self.variance_ema + 1e-9)
        
        # High variance ratio → confident → use base k
        # Low variance ratio → uncertain → use more experts
        # Inverse relationship for MoE
        adaptive_k = int(self.router_config.top_k / max(ratio.item(), 0.5))
        
        # Clamp to configured bounds
        adaptive_k = max(
            self.router_config.top_k_min,
            min(self.router_config.top_k_max, adaptive_k)
        )
        
        return adaptive_k
    
    def _update_tracking(self, indices: torch.Tensor):
        """Update expert usage counts for load balancing."""
        # Flatten indices
        flat_indices = indices.view(-1)
        
        # Count occurrences of each expert
        counts = torch.bincount(
            flat_indices,
            minlength=self.num_routable
        ).float()
        
        # Accumulate counts
        self.expert_counts.add_(counts)
        self.total_tokens.add_(flat_indices.numel())
    
    def update_expert_bias(self) -> Dict:
        """
        Update expert biases based on load (loss-free balancing).
        
        Call after each training step or N steps.
        
        Algorithm:
            target = total_tokens / num_experts
            if expert_count > target × 1.1: decrease bias
            if expert_count < target × 0.9: increase bias
        
        Returns:
            Dict with load balancing metrics
        """
        if self.total_tokens == 0:
            return {'updated': False}
        
        # Compute utilization
        utilization = self.expert_counts / self.total_tokens.float()
        target = 1.0 / self.num_routable
        
        # Compute adjustments
        adjustments = torch.zeros_like(self.expert_bias)
        
        with torch.no_grad():
            # Decrease bias for overloaded experts
            overloaded = utilization > target * 1.1
            adjustments[overloaded] = -self.router_config.bias_update_speed
            
            # Increase bias for underloaded experts
            underloaded = utilization < target * 0.9
            adjustments[underloaded] = self.router_config.bias_update_speed
            
            # Apply adjustments with clamping
            self.expert_bias.add_(adjustments)
            self.expert_bias.clamp_(
                self.router_config.bias_clamp_min,
                self.router_config.bias_clamp_max
            )
        
        # Compute metrics
        metrics = {
            'updated': True,
            'utilization': utilization.cpu().tolist(),
            'adjustments': adjustments.cpu().tolist(),
            'dead_experts': (utilization < 0.01).sum().item(),
            'overloaded_experts': (utilization > target * 2).sum().item(),
            'max_utilization': utilization.max().item(),
            'min_utilization': utilization.min().item(),
            'load_balance_cv': (utilization.std() / utilization.mean()).item(),
        }
        
        # Reset tracking
        self.expert_counts.zero_()
        self.total_tokens.zero_()
        
        return metrics
    
    def get_routing_entropy(self) -> float:
        """Compute normalized routing entropy for health check."""
        if self.total_tokens == 0:
            return 1.0
        
        probs = self.expert_counts / self.total_tokens.float()
        probs = probs + 1e-10  # Avoid log(0)
        
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(self.num_routable)
        
        return entropy / max_entropy
    
    def get_null_routing_stats(
        self,
        indices: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Compute null routing statistics for telemetry.
        
        Args:
            indices: [batch, seq, top_k] expert indices
            token_ids: [batch, seq] token IDs for junk/signal classification
            
        Returns:
            Dict with null routing rates
        """
        flat_indices = indices.view(-1)
        total = flat_indices.numel()
        
        # Count null expert selections
        null_count = sum(
            (flat_indices == idx).sum().item()
            for idx in self.null_indices
        )
        
        stats = {
            'overall_null_rate': null_count / total,
            'null_count': null_count,
            'total_selections': total,
        }
        
        # If token IDs provided, compute junk vs signal rates
        if token_ids is not None and hasattr(self.config, 'tokenizer'):
            junk_mask = torch.zeros_like(token_ids, dtype=torch.bool)
            for tid in self.config.tokenizer.junk_token_ids:
                junk_mask |= (token_ids == tid)
            
            # Expand mask to match indices
            junk_mask_expanded = junk_mask.unsqueeze(-1).expand_as(indices)
            signal_mask_expanded = ~junk_mask_expanded
            
            # Compute null routing for junk vs signal
            junk_indices = indices[junk_mask_expanded].view(-1)
            signal_indices = indices[signal_mask_expanded].view(-1)
            
            if junk_indices.numel() > 0:
                junk_null = sum(
                    (junk_indices == idx).sum().item()
                    for idx in self.null_indices
                )
                stats['junk_null_rate'] = junk_null / junk_indices.numel()
            
            if signal_indices.numel() > 0:
                signal_null = sum(
                    (signal_indices == idx).sum().item()
                    for idx in self.null_indices
                )
                stats['signal_null_rate'] = signal_null / signal_indices.numel()
        
        return stats


class RouterWarmup:
    """
    Router warmup procedure from GSA paper Section 6.1.
    
    Trains router in isolation to produce uniform distribution
    before full training begins.
    """
    
    def __init__(self, router: GSARouter, num_steps: int = 1000):
        self.router = router
        self.num_steps = num_steps
        self.step = 0
    
    def warmup_step(
        self,
        hidden_states: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single warmup step.
        
        Target: Uniform routing distribution.
        """
        optimizer.zero_grad()
        
        # Get routing scores
        _, _, aux = self.router(hidden_states, return_all_scores=True)
        scores = aux['all_scores']  # [batch, seq, num_experts]
        
        # Target: uniform distribution
        num_experts = scores.shape[-1]
        target = torch.ones_like(scores) / num_experts
        
        # KL divergence loss
        log_probs = F.log_softmax(scores, dim=-1)
        loss = F.kl_div(log_probs, target, reduction='batchmean')
        
        loss.backward()
        optimizer.step()
        
        self.step += 1
        
        return loss.item()
    
    @property
    def is_complete(self) -> bool:
        return self.step >= self.num_steps
