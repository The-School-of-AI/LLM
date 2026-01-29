"""
Loss-Free Load Balancer
=======================

Implementation of loss-free load balancing for MoE routers.

Based on DeepSeek-V3 approach:
- No auxiliary loss in training objective
- Expert biases adjusted algorithmically based on utilization
- Biases affect SELECTION, not CONTRIBUTION
- Automatic dead expert revival

Key Formula:
    bias_i(t+1) = clamp(bias_i(t) + γ × sign(target - count_i), bias_min, bias_max)

Where:
- γ = 0.001 (update speed from DeepSeek-V3)
- target = total_tokens / num_experts
- bias affects routing selection but not gating weights

Team 7 Integration:
- Telemetry collection for routing health
- Automatic alerts for dead/overloaded experts
- Plugin interface for custom balancing strategies

References:
- DeepSeek-V3: Loss-free load balancing
- arXiv:2406.13233: Expert routing analysis
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque


class BalancingStrategy(Enum):
    """Load balancing strategies."""
    BIAS_ONLY = "bias_only"           # Pure bias adjustment (recommended)
    AUX_LOSS = "aux_loss"             # Traditional auxiliary loss
    HYBRID = "hybrid"                  # Tiny aux loss + bias
    EXPERT_CHOICE = "expert_choice"    # Experts choose tokens


@dataclass
class LoadBalanceConfig:
    """Configuration for load balancer."""
    strategy: BalancingStrategy = BalancingStrategy.BIAS_ONLY
    
    # Bias adjustment parameters (from DeepSeek-V3)
    bias_update_speed: float = 0.001      # γ
    bias_min: float = -2.0
    bias_max: float = 2.0
    
    # Auxiliary loss (if used)
    aux_loss_weight: float = 0.0001       # Very small
    
    # Update frequency
    update_every_n_tokens: int = 1024     # Update bias every N tokens
    
    # Dead expert handling
    dead_threshold: float = 0.01          # <1% utilization = dead
    auto_revive: bool = True
    revive_bias_boost: float = 0.5        # Boost bias for dead experts
    
    # Overload handling  
    overload_threshold: float = 3.0       # >3× expected = overloaded
    
    # Null expert targets
    null_target_min: float = 0.05         # Minimum null utilization
    null_target_max: float = 0.20         # Maximum null utilization
    
    # History tracking
    history_window: int = 100             # Steps to track


@dataclass
class LoadBalanceMetrics:
    """Metrics from load balancing."""
    step: int
    total_tokens: int
    
    # Utilization stats
    utilization: List[float]
    utilization_mean: float
    utilization_std: float
    utilization_cv: float  # Coefficient of variation
    
    # Balance metrics
    gini_coefficient: float
    entropy: float
    normalized_entropy: float
    
    # Expert health
    dead_experts: List[int]
    overloaded_experts: List[int]
    
    # Null routing
    null_utilization: Optional[float] = None
    
    # Adjustments made
    adjustments: Optional[List[float]] = None
    
    def is_healthy(self, config: LoadBalanceConfig) -> bool:
        """Check if routing is healthy."""
        # Check for dead experts
        if len(self.dead_experts) > 0:
            return False
        
        # Check entropy
        if self.normalized_entropy < 0.7:
            return False
        
        # Check Gini coefficient
        if self.gini_coefficient > 0.5:
            return False
        
        return True


class LoadBalancer:
    """
    Loss-free load balancer for MoE routing.
    
    Features:
    - Bias-only adjustment (no auxiliary loss by default)
    - Automatic dead expert revival
    - Comprehensive telemetry
    - Plugin interface for custom strategies
    
    Usage:
        balancer = LoadBalancer(num_experts=8, config=config)
        
        # During training
        balancer.update(expert_counts)
        bias = balancer.get_bias()
        
        # Check health
        metrics = balancer.get_metrics()
        if not metrics.is_healthy(config):
            print("Warning: routing issues detected")
    """
    
    def __init__(
        self,
        num_experts: int,
        num_null_experts: int = 0,
        config: Optional[LoadBalanceConfig] = None,
        device: str = 'cuda'
    ):
        self.num_experts = num_experts
        self.num_null_experts = num_null_experts
        self.num_routed = num_experts - num_null_experts
        self.config = config or LoadBalanceConfig()
        self.device = device
        
        # Expert biases (learnable via algorithm, not gradient)
        self.expert_bias = torch.zeros(num_experts, device=device)
        
        # Initialize null expert bias slightly higher
        if num_null_experts > 0:
            null_start = num_experts - num_null_experts
            self.expert_bias[null_start:] = 0.1
        
        # Tracking
        self.expert_counts = torch.zeros(num_experts, device=device)
        self.total_tokens = 0
        self.step = 0
        
        # History for analysis
        self.utilization_history = deque(maxlen=self.config.history_window)
        self.entropy_history = deque(maxlen=self.config.history_window)
        self.metrics_history = deque(maxlen=self.config.history_window)
        
        # Plugins
        self._pre_update_hooks: List[Callable] = []
        self._post_update_hooks: List[Callable] = []
    
    def accumulate(self, expert_indices: torch.Tensor):
        """
        Accumulate expert usage counts from routing decisions.
        
        Args:
            expert_indices: [batch, seq, k] or [num_selections] expert IDs
        """
        flat_indices = expert_indices.view(-1)
        
        # Count occurrences
        counts = torch.bincount(
            flat_indices.long(),
            minlength=self.num_experts
        ).float()
        
        self.expert_counts += counts.to(self.device)
        self.total_tokens += flat_indices.numel()
    
    def update(self, force: bool = False) -> Optional[LoadBalanceMetrics]:
        """
        Update expert biases based on accumulated counts.
        
        Args:
            force: Force update even if not enough tokens
            
        Returns:
            LoadBalanceMetrics if update performed, None otherwise
        """
        # Check if enough tokens accumulated
        if not force and self.total_tokens < self.config.update_every_n_tokens:
            return None
        
        if self.total_tokens == 0:
            return None
        
        # Run pre-update hooks
        for hook in self._pre_update_hooks:
            hook(self)
        
        # Compute utilization
        utilization = self.expert_counts / self.total_tokens
        target = 1.0 / self.num_experts
        
        # Compute adjustments based on strategy
        if self.config.strategy in [BalancingStrategy.BIAS_ONLY, BalancingStrategy.HYBRID]:
            adjustments = self._compute_bias_adjustments(utilization, target)
            self._apply_adjustments(adjustments)
        else:
            adjustments = None
        
        # Handle dead experts
        dead_experts = self._find_dead_experts(utilization)
        if self.config.auto_revive and dead_experts:
            self._revive_experts(dead_experts)
        
        # Compute metrics
        metrics = self._compute_metrics(utilization, adjustments, dead_experts)
        
        # Store history
        self.utilization_history.append(utilization.cpu().numpy())
        self.entropy_history.append(metrics.normalized_entropy)
        self.metrics_history.append(metrics)
        
        # Reset counters
        self.expert_counts.zero_()
        self.total_tokens = 0
        self.step += 1
        
        # Run post-update hooks
        for hook in self._post_update_hooks:
            hook(self, metrics)
        
        return metrics
    
    def _compute_bias_adjustments(
        self,
        utilization: torch.Tensor,
        target: float
    ) -> torch.Tensor:
        """Compute bias adjustments based on utilization."""
        adjustments = torch.zeros_like(self.expert_bias)
        
        # Decrease bias for overloaded experts
        overloaded = utilization > target * (1 + self.config.overload_threshold / 10)
        adjustments[overloaded] = -self.config.bias_update_speed
        
        # Increase bias for underloaded experts
        underloaded = utilization < target * 0.9
        adjustments[underloaded] = self.config.bias_update_speed
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: torch.Tensor):
        """Apply bias adjustments with clamping."""
        self.expert_bias += adjustments
        self.expert_bias.clamp_(self.config.bias_min, self.config.bias_max)
    
    def _find_dead_experts(self, utilization: torch.Tensor) -> List[int]:
        """Find experts with utilization below threshold."""
        dead_mask = utilization < self.config.dead_threshold
        return dead_mask.nonzero().squeeze(-1).tolist()
    
    def _find_overloaded_experts(self, utilization: torch.Tensor) -> List[int]:
        """Find experts with utilization above threshold."""
        target = 1.0 / self.num_experts
        overload_mask = utilization > target * self.config.overload_threshold
        return overload_mask.nonzero().squeeze(-1).tolist()
    
    def _revive_experts(self, dead_experts: List[int]):
        """Boost bias for dead experts to encourage routing."""
        for expert_id in dead_experts:
            self.expert_bias[expert_id] += self.config.revive_bias_boost
        
        # Clamp after boosting
        self.expert_bias.clamp_(self.config.bias_min, self.config.bias_max)
    
    def _compute_metrics(
        self,
        utilization: torch.Tensor,
        adjustments: Optional[torch.Tensor],
        dead_experts: List[int]
    ) -> LoadBalanceMetrics:
        """Compute comprehensive load balance metrics."""
        # Basic stats
        util_list = utilization.cpu().tolist()
        util_mean = utilization.mean().item()
        util_std = utilization.std().item()
        util_cv = util_std / (util_mean + 1e-9)
        
        # Gini coefficient
        sorted_util = torch.sort(utilization)[0]
        n = len(sorted_util)
        cumsum = torch.cumsum(sorted_util, dim=0)
        gini = ((2 * torch.arange(1, n+1, device=utilization.device).float() - n - 1) * sorted_util).sum()
        gini = gini / (n * sorted_util.sum() + 1e-9)
        
        # Entropy
        probs = utilization + 1e-10
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy
        
        # Overloaded experts
        overloaded = self._find_overloaded_experts(utilization)
        
        # Null utilization (if applicable)
        null_util = None
        if self.num_null_experts > 0:
            null_start = self.num_experts - self.num_null_experts
            null_util = utilization[null_start:].sum().item()
        
        return LoadBalanceMetrics(
            step=self.step,
            total_tokens=self.total_tokens,
            utilization=util_list,
            utilization_mean=util_mean,
            utilization_std=util_std,
            utilization_cv=util_cv,
            gini_coefficient=gini.item(),
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            dead_experts=dead_experts,
            overloaded_experts=overloaded,
            null_utilization=null_util,
            adjustments=adjustments.cpu().tolist() if adjustments is not None else None
        )
    
    def get_bias(self) -> torch.Tensor:
        """Get current expert biases."""
        return self.expert_bias
    
    def set_bias(self, bias: torch.Tensor):
        """Set expert biases (e.g., from checkpoint)."""
        self.expert_bias.copy_(bias)
    
    def get_metrics(self) -> Optional[LoadBalanceMetrics]:
        """Get most recent metrics."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_auxiliary_loss(
        self,
        router_probs: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss (if using hybrid strategy).
        
        Based on Switch Transformer / GShard auxiliary loss.
        
        Args:
            router_probs: [batch*seq, num_experts] routing probabilities
            expert_indices: [batch*seq, k] selected experts
            
        Returns:
            aux_loss: Scalar auxiliary loss
        """
        if self.config.strategy not in [BalancingStrategy.AUX_LOSS, BalancingStrategy.HYBRID]:
            return torch.tensor(0.0, device=router_probs.device)
        
        num_tokens = router_probs.shape[0]
        
        # Expert load (fraction of tokens per expert)
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, expert_indices, 1.0)
        expert_load = expert_mask.mean(dim=0)
        
        # Router probability mass per expert
        prob_mass = router_probs.mean(dim=0)
        
        # Auxiliary loss: minimize load × probability
        # This encourages uniform distribution
        aux_loss = (expert_load * prob_mass * self.num_experts).sum()
        
        return aux_loss * self.config.aux_loss_weight
    
    def register_pre_update_hook(self, hook: Callable):
        """Register hook called before bias update."""
        self._pre_update_hooks.append(hook)
    
    def register_post_update_hook(self, hook: Callable):
        """Register hook called after bias update with metrics."""
        self._post_update_hooks.append(hook)
    
    def state_dict(self) -> Dict:
        """Get state for checkpointing."""
        return {
            'expert_bias': self.expert_bias.cpu(),
            'step': self.step,
            'config': {
                'strategy': self.config.strategy.value,
                'bias_update_speed': self.config.bias_update_speed,
                'bias_min': self.config.bias_min,
                'bias_max': self.config.bias_max,
            }
        }
    
    def load_state_dict(self, state: Dict):
        """Load state from checkpoint."""
        self.expert_bias.copy_(state['expert_bias'].to(self.device))
        self.step = state['step']


class NullRoutingMonitor:
    """
    Monitor null expert routing for junk token absorption.
    
    Targets (from spec):
    - Junk tokens: 60-80% should route to null
    - Signal tokens: <10% should route to null
    
    Provides telemetry for Team 7 integration.
    """
    
    def __init__(
        self,
        null_expert_ids: List[int],
        junk_token_ids: List[int],
        target_junk_rate: Tuple[float, float] = (0.6, 0.8),
        target_signal_rate: Tuple[float, float] = (0.0, 0.1)
    ):
        self.null_expert_ids = set(null_expert_ids)
        self.junk_token_ids = set(junk_token_ids)
        self.target_junk_rate = target_junk_rate
        self.target_signal_rate = target_signal_rate
        
        # Tracking
        self.junk_total = 0
        self.junk_to_null = 0
        self.signal_total = 0
        self.signal_to_null = 0
        
        # History
        self.junk_rate_history = deque(maxlen=100)
        self.signal_rate_history = deque(maxlen=100)
    
    def update(
        self,
        token_ids: torch.Tensor,
        expert_indices: torch.Tensor
    ):
        """
        Update null routing statistics.
        
        Args:
            token_ids: [batch, seq] token IDs
            expert_indices: [batch, seq, k] selected expert IDs
        """
        flat_tokens = token_ids.view(-1)
        flat_indices = expert_indices.view(-1, expert_indices.shape[-1])
        
        # Create masks
        junk_mask = torch.zeros_like(flat_tokens, dtype=torch.bool)
        for tid in self.junk_token_ids:
            junk_mask |= (flat_tokens == tid)
        
        signal_mask = ~junk_mask
        
        # Check if routed to null
        null_mask = torch.zeros_like(flat_indices[:, 0], dtype=torch.bool)
        for nid in self.null_expert_ids:
            null_mask |= (flat_indices == nid).any(dim=-1)
        
        # Update counts
        self.junk_total += junk_mask.sum().item()
        self.junk_to_null += (junk_mask & null_mask).sum().item()
        self.signal_total += signal_mask.sum().item()
        self.signal_to_null += (signal_mask & null_mask).sum().item()
    
    def get_rates(self) -> Dict[str, float]:
        """Get current null routing rates."""
        junk_rate = self.junk_to_null / (self.junk_total + 1e-9)
        signal_rate = self.signal_to_null / (self.signal_total + 1e-9)
        
        return {
            'junk_null_rate': junk_rate,
            'signal_null_rate': signal_rate,
            'junk_total': self.junk_total,
            'signal_total': self.signal_total,
        }
    
    def check_health(self) -> Dict[str, any]:
        """Check if null routing is within targets."""
        rates = self.get_rates()
        
        alerts = []
        
        # Check junk rate
        if rates['junk_null_rate'] < self.target_junk_rate[0]:
            alerts.append(f"Junk→null rate {rates['junk_null_rate']:.1%} below target {self.target_junk_rate[0]:.1%}")
        elif rates['junk_null_rate'] > self.target_junk_rate[1]:
            alerts.append(f"Junk→null rate {rates['junk_null_rate']:.1%} above target {self.target_junk_rate[1]:.1%}")
        
        # Check signal rate
        if rates['signal_null_rate'] > self.target_signal_rate[1]:
            alerts.append(f"Signal→null rate {rates['signal_null_rate']:.1%} above threshold {self.target_signal_rate[1]:.1%}")
        
        return {
            'is_healthy': len(alerts) == 0,
            'alerts': alerts,
            **rates
        }
    
    def reset(self):
        """Reset tracking counters."""
        # Store history before reset
        if self.junk_total > 0:
            self.junk_rate_history.append(self.junk_to_null / self.junk_total)
        if self.signal_total > 0:
            self.signal_rate_history.append(self.signal_to_null / self.signal_total)
        
        self.junk_total = 0
        self.junk_to_null = 0
        self.signal_total = 0
        self.signal_to_null = 0
