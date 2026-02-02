"""
MoE Block
=========

Complete Mixture of Experts block combining:
1. GSA Router - Token to expert routing
2. Shared Experts - Always active
3. Routed Experts - Selected by router
4. Null Experts - Zero-compute pathway
5. Telemetry - Health monitoring

Architecture Flow:
    Input
      ↓
    ┌─────────────────────────────────┐
    │         GSA Router              │
    │  (Multi-head sigmoid routing)   │
    └─────────────────────────────────┘
      ↓                     ↓
    ┌───────┐         ┌─────────────┐
    │Shared │         │   Top-K     │
    │Experts│         │  Selection  │
    │(always)│        └─────────────┘
    └───────┘               ↓
      ↓               ┌─────────────┐
      │               │   Routed    │
      │               │   Experts   │
      │               │  (+ Null)   │
      │               └─────────────┘
      ↓                     ↓
    ┌─────────────────────────────────┐
    │           Combine               │
    │   shared_out + routed_out       │
    └─────────────────────────────────┘
      ↓
    Output

Team 7 Integration:
- Null-routing telemetry
- Routing health gates
- Loss-free control plugin interface
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import math

from model.config import MoEModelConfig, RouterType
from model.router import GSARouter, NullExpertRouter
from model.expert import ExpertContainer, GatedExpert


class MoEBlock(nn.Module):
    """
    Complete Mixture of Experts block.
    
    Features:
    - GSA-style multi-head router with sigmoid scoring
    - Shared experts (always active for common patterns)
    - Routed experts (selected by router)
    - Null experts (zero-compute for junk tokens)
    - Loss-free load balancing via bias adjustment
    - Comprehensive telemetry for monitoring
    
    Args:
        config: Model configuration
        layer_idx: Layer index (for logging)
    """
    
    def __init__(self, config: MoEModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # ============================================================
        # Components
        # ============================================================
        
        # Router
        if config.router.router_type == RouterType.NULL_EXPERT:
            self.router = NullExpertRouter(config)
        else:
            self.router = GSARouter(config)
        
        # Experts container
        self.experts = ExpertContainer(config)
        
        # ============================================================
        # Telemetry (Team 7 Integration)
        # ============================================================
        
        self.telemetry = MoETelemetry(config)
        
        # Track whether we're in evaluation mode for telemetry
        self._collect_telemetry = True
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        return_router_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MoE block.
        
        Args:
            hidden_states: [batch, seq, hidden] input
            token_ids: [batch, seq] token IDs for telemetry
            return_router_info: Return detailed routing information
            
        Returns:
            output: [batch, seq, hidden]
            aux_info: Optional dict with routing stats and aux loss
        """
        batch_size, seq_len, hidden = hidden_states.shape
        
        # ============================================================
        # Shared Experts (always compute)
        # ============================================================
        shared_output = self.experts.forward_shared(hidden_states)
        
        # ============================================================
        # Routing
        # ============================================================
        expert_indices, gating_weights, router_aux = self.router(
            hidden_states,
            return_all_scores=return_router_info
        )
        
        # ============================================================
        # Routed Experts
        # ============================================================
        routed_output = self.experts.forward_routed(
            hidden_states,
            expert_indices,
            gating_weights
        )
        
        # ============================================================
        # Combine Outputs
        # ============================================================
        output = shared_output + routed_output
        
        # ============================================================
        # Telemetry
        # ============================================================
        aux_info = None
        if return_router_info or self._collect_telemetry:
            aux_info = self._collect_aux_info(
                expert_indices,
                gating_weights,
                router_aux,
                token_ids,
                return_router_info
            )
        
        return output, aux_info
    
    def _collect_aux_info(
        self,
        expert_indices: torch.Tensor,
        gating_weights: torch.Tensor,
        router_aux: Dict,
        token_ids: Optional[torch.Tensor],
        return_router_info: bool = False
    ) -> Dict:
        """Collect auxiliary information for telemetry and logging."""
        aux_info = {
            'layer_idx': self.layer_idx,
            **router_aux
        }

        if return_router_info:
            aux_info['expert_indices'] = expert_indices
            aux_info['gating_weights'] = gating_weights
            if router_aux and 'aux_loss' in router_aux:
                aux_info['aux_loss'] = router_aux['aux_loss']
        
        # Add null routing stats
        null_stats = self.router.get_null_routing_stats(
            expert_indices,
            token_ids
        )
        aux_info.update(null_stats)
        
        # Add telemetry
        if self._collect_telemetry:
            health = self.telemetry.check_health(
                expert_indices,
                gating_weights,
                self.router.expert_counts,
                self.router.total_tokens
            )
            aux_info['health'] = health
        
        return aux_info
    
    def post_training_step(self) -> Dict:
        """
        Call after each training step.
        
        Updates expert biases and returns metrics.
        """
        return self.router.update_expert_bias()
    
    @torch.no_grad()
    def init_from_dense(self, dense_ffn: nn.Module, noise_std: float = 1e-4):
        """
        Initialize experts from a dense FFN.
        
        Used for "explosion" expansion (1B Dense → 3B MoE).
        """
        self.experts.init_from_dense(dense_ffn, noise_std)
    
    @torch.no_grad()
    def expand_from_moe(
        self,
        source_moe: 'MoEBlock',
        children_per_parent: int = 8,
        noise_std: float = 1e-3
    ):
        """
        Initialize from a smaller MoE via expert expansion.
        
        Used for expert expansion (8B MoE-8 → 70B MoE-64).
        """
        self.experts.expand_experts(
            source_moe.experts,
            children_per_parent,
            noise_std
        )
        
        # Optionally warm-start router from parent
        # (hierarchical key initialization)
        self._init_router_from_parent(source_moe.router, children_per_parent)
    
    def _init_router_from_parent(
        self,
        parent_router: GSARouter,
        children_per_parent: int
    ):
        """Initialize router keys hierarchically from parent."""
        num_parents = parent_router.expert_keys.shape[0]
        
        with torch.no_grad():
            for parent_idx in range(num_parents):
                parent_key = parent_router.expert_keys[parent_idx]
                
                for child_idx in range(children_per_parent):
                    global_idx = parent_idx * children_per_parent + child_idx
                    
                    if global_idx < self.config.num_routed_experts:
                        # Child key = parent key + small offset
                        self.router.expert_keys[global_idx].copy_(parent_key)
                        self.router.expert_keys[global_idx].add_(
                            torch.randn_like(parent_key) * 0.1
                        )


class MoETelemetry:
    """
    Telemetry for MoE health monitoring.
    
    Team 7 Integration:
    - Null-routing telemetry
    - Routing health gates
    - Loss-free control plugin interface
    
    Monitors:
    - Expert utilization balance
    - Dead expert detection
    - Null routing rates (junk vs signal)
    - Router entropy
    - Load balance coefficient
    """
    
    def __init__(self, config: MoEModelConfig):
        self.config = config
        self.telemetry_config = config.telemetry
        
        # History for trend analysis
        self.utilization_history = []
        self.entropy_history = []
        self.null_rate_history = []
    
    def check_health(
        self,
        expert_indices: torch.Tensor,
        gating_weights: torch.Tensor,
        expert_counts: torch.Tensor,
        total_tokens: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive health check for routing.
        
        Returns dict with health status and metrics.
        """
        health = {
            'is_healthy': True,
            'alerts': [],
            'metrics': {}
        }
        
        if total_tokens == 0:
            health['metrics']['status'] = 'no_data'
            return health
        
        # Compute utilization
        utilization = expert_counts / total_tokens.float()
        num_experts = len(utilization)
        expected = 1.0 / num_experts
        
        # ============================================================
        # Check 1: Dead Experts
        # ============================================================
        dead_mask = utilization < self.telemetry_config.dead_expert_threshold
        dead_count = dead_mask.sum().item()
        
        if dead_count > 0:
            health['is_healthy'] = False
            health['alerts'].append({
                'type': 'dead_experts',
                'message': f'{dead_count} experts below {self.telemetry_config.dead_expert_threshold:.1%} utilization',
                'expert_ids': dead_mask.nonzero().squeeze(-1).tolist()
            })
        
        # ============================================================
        # Check 2: Overloaded Experts
        # ============================================================
        overload_threshold = expected * self.telemetry_config.overload_expert_threshold
        overloaded_mask = utilization > overload_threshold
        overloaded_count = overloaded_mask.sum().item()
        
        if overloaded_count > 0:
            health['alerts'].append({
                'type': 'overloaded_experts',
                'message': f'{overloaded_count} experts above {self.telemetry_config.overload_expert_threshold:.1f}× expected',
                'expert_ids': overloaded_mask.nonzero().squeeze(-1).tolist()
            })
        
        # ============================================================
        # Check 3: Router Entropy
        # ============================================================
        probs = utilization + 1e-10
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(num_experts)
        normalized_entropy = entropy / max_entropy
        
        if normalized_entropy < self.telemetry_config.min_router_entropy:
            health['is_healthy'] = False
            health['alerts'].append({
                'type': 'low_entropy',
                'message': f'Router entropy {normalized_entropy:.2f} below threshold {self.telemetry_config.min_router_entropy}'
            })
        
        # ============================================================
        # Check 4: Load Balance (Gini Coefficient)
        # ============================================================
        sorted_util = torch.sort(utilization)[0]
        n = len(sorted_util)
        cumsum = torch.cumsum(sorted_util, dim=0)
        gini = (2 * torch.arange(1, n+1, device=utilization.device) - n - 1).float()
        gini = (gini * sorted_util).sum() / (n * sorted_util.sum())
        gini = gini.item()
        
        if gini > self.telemetry_config.max_gini_coefficient:
            health['alerts'].append({
                'type': 'load_imbalance',
                'message': f'Gini coefficient {gini:.2f} above threshold {self.telemetry_config.max_gini_coefficient}'
            })
        
        # ============================================================
        # Metrics
        # ============================================================
        health['metrics'] = {
            'dead_experts': dead_count,
            'overloaded_experts': overloaded_count,
            'normalized_entropy': normalized_entropy,
            'gini_coefficient': gini,
            'max_utilization': utilization.max().item(),
            'min_utilization': utilization.min().item(),
            'utilization_std': utilization.std().item(),
            'utilization_cv': (utilization.std() / utilization.mean()).item(),
        }
        
        # Update history
        self.utilization_history.append(utilization.cpu().numpy())
        self.entropy_history.append(normalized_entropy)
        
        # Keep only recent history
        max_history = 1000
        if len(self.utilization_history) > max_history:
            self.utilization_history = self.utilization_history[-max_history:]
            self.entropy_history = self.entropy_history[-max_history:]
        
        return health
    
    def check_null_routing(
        self,
        junk_null_rate: float,
        signal_null_rate: float
    ) -> Dict[str, Any]:
        """
        Check null routing rates against targets.
        
        Targets (from spec):
        - Junk tokens: 60-80% should route to null
        - Signal tokens: <10% should route to null
        """
        alerts = []
        
        # Check junk null rate
        if junk_null_rate < self.telemetry_config.junk_null_rate_alert_low:
            alerts.append({
                'type': 'junk_null_low',
                'message': f'Junk→null rate {junk_null_rate:.1%} below target {self.telemetry_config.junk_null_rate_alert_low:.1%}'
            })
        elif junk_null_rate > self.telemetry_config.junk_null_rate_alert_high:
            alerts.append({
                'type': 'junk_null_high',
                'message': f'Junk→null rate {junk_null_rate:.1%} above target {self.telemetry_config.junk_null_rate_alert_high:.1%}'
            })
        
        # Check signal null rate
        if signal_null_rate > self.telemetry_config.signal_null_rate_alert:
            alerts.append({
                'type': 'signal_null_high',
                'message': f'Signal→null rate {signal_null_rate:.1%} above threshold {self.telemetry_config.signal_null_rate_alert:.1%}'
            })
        
        self.null_rate_history.append({
            'junk': junk_null_rate,
            'signal': signal_null_rate
        })
        
        return {
            'is_healthy': len(alerts) == 0,
            'alerts': alerts,
            'junk_null_rate': junk_null_rate,
            'signal_null_rate': signal_null_rate
        }
    
    def get_summary(self) -> str:
        """Get human-readable telemetry summary."""
        if not self.utilization_history:
            return "No telemetry data collected yet."
        
        latest_util = self.utilization_history[-1]
        latest_entropy = self.entropy_history[-1]
        
        lines = [
            "MoE Telemetry Summary",
            "=" * 40,
            f"Utilization range: [{latest_util.min():.1%}, {latest_util.max():.1%}]",
            f"Utilization std: {latest_util.std():.3f}",
            f"Normalized entropy: {latest_entropy:.3f}",
        ]
        
        if self.null_rate_history:
            latest_null = self.null_rate_history[-1]
            lines.extend([
                f"Junk→null rate: {latest_null['junk']:.1%}",
                f"Signal→null rate: {latest_null['signal']:.1%}",
            ])
        
        return "\n".join(lines)


class DenseFFN(nn.Module):
    """
    Dense FFN (SwiGLU) for non-MoE layers.
    
    Used when moe_layer_frequency > 1 (some layers are dense).
    Also used as the source for expert initialization.
    """
    
    def __init__(self, config: MoEModelConfig):
        super().__init__()
        self.config = config
        
        # SwiGLU projections
        self.w1 = nn.Linear(config.hidden_size, config.expert.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.expert.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.expert.intermediate_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU forward pass."""
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
