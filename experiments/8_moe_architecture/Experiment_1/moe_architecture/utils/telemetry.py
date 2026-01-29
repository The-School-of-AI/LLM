"""
Team 7 Telemetry Interface
==========================

Comprehensive telemetry system for MoE routing health monitoring.

Provides:
1. Null-routing telemetry (junk vs signal token routing)
2. Routing health gates (entropy, balance, dead experts)
3. Loss-free control plugin interface
4. Alerts and automatic correction

Integration Points:
- Pre-routing hooks: Modify inputs before routing
- Post-routing hooks: Monitor and adjust after routing
- Health checks: Validate routing decisions
- Alerts: Notify when thresholds exceeded

Usage:
    telemetry = MoETelemetrySystem(config)
    
    # Register with MoE block
    moe_block.register_telemetry(telemetry)
    
    # After forward pass
    telemetry.log_routing(expert_indices, gating_weights, token_ids)
    
    # Check health periodically
    health = telemetry.check_health()
    if not health['is_healthy']:
        print(health['alerts'])
"""

import torch
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import json
import math
import logging


# Configure logging
logger = logging.getLogger('moe_telemetry')


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    # Logging
    enabled: bool = True
    log_level: str = 'INFO'
    log_interval: int = 100
    detailed_logging: bool = False
    
    # Health thresholds
    dead_expert_threshold: float = 0.01
    overload_threshold: float = 3.0
    min_entropy: float = 0.70
    max_gini: float = 0.50
    
    # Null routing targets
    junk_null_target: Tuple[float, float] = (0.60, 0.80)
    signal_null_target_max: float = 0.10
    
    # Alerts
    alert_on_dead_expert: bool = True
    alert_on_collapse: bool = True
    alert_on_null_violation: bool = True
    
    # Auto-correction
    auto_correct: bool = True
    correction_strength: float = 0.1
    
    # Export
    export_metrics: bool = False
    export_path: Optional[str] = None


@dataclass
class RoutingEvent:
    """Single routing event for logging."""
    timestamp: str
    step: int
    layer_idx: int
    batch_size: int
    seq_len: int
    
    # Routing stats
    expert_utilization: List[float]
    entropy: float
    gini: float
    
    # Null routing
    null_rate: Optional[float] = None
    junk_null_rate: Optional[float] = None
    signal_null_rate: Optional[float] = None
    
    # Alerts
    alerts: List[str] = field(default_factory=list)


class MoETelemetrySystem:
    """
    Comprehensive telemetry system for MoE routing.
    
    Provides monitoring, alerting, and auto-correction for:
    - Expert load balance
    - Routing entropy
    - Dead expert detection
    - Null routing effectiveness
    """
    
    def __init__(
        self,
        num_experts: int,
        num_null_experts: int = 0,
        null_expert_ids: Optional[List[int]] = None,
        junk_token_ids: Optional[List[int]] = None,
        config: Optional[TelemetryConfig] = None
    ):
        self.num_experts = num_experts
        self.num_null_experts = num_null_experts
        self.config = config or TelemetryConfig()
        
        # Null expert tracking
        if null_expert_ids is None:
            self.null_expert_ids = list(range(
                num_experts - num_null_experts, 
                num_experts
            ))
        else:
            self.null_expert_ids = null_expert_ids
        
        self.junk_token_ids = set(junk_token_ids or [0])  # Default: padding token
        
        # Tracking state
        self.step = 0
        self.expert_counts = torch.zeros(num_experts)
        self.total_selections = 0
        
        # Null routing tracking
        self.junk_total = 0
        self.junk_to_null = 0
        self.signal_total = 0
        self.signal_to_null = 0
        
        # History
        self.utilization_history = deque(maxlen=1000)
        self.entropy_history = deque(maxlen=1000)
        self.events: List[RoutingEvent] = []
        
        # Hooks
        self._pre_routing_hooks: List[Callable] = []
        self._post_routing_hooks: List[Callable] = []
        self._health_check_hooks: List[Callable] = []
        
        # Alert handlers
        self._alert_handlers: List[Callable] = []
        
        # Set up logging
        if self.config.enabled:
            logger.setLevel(getattr(logging, self.config.log_level))
    
    # =========================================================================
    # Logging Methods
    # =========================================================================
    
    def log_routing(
        self,
        expert_indices: torch.Tensor,
        gating_weights: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        layer_idx: int = 0
    ):
        """
        Log routing decision for telemetry.
        
        Args:
            expert_indices: [batch, seq, k] selected expert IDs
            gating_weights: [batch, seq, k] gating weights
            token_ids: [batch, seq] token IDs for null routing analysis
            layer_idx: Layer index for multi-layer tracking
        """
        if not self.config.enabled:
            return
        
        batch_size = expert_indices.shape[0]
        seq_len = expert_indices.shape[1]
        
        # Update expert counts
        flat_indices = expert_indices.view(-1)
        counts = torch.bincount(flat_indices.long(), minlength=self.num_experts).float()
        self.expert_counts += counts.cpu()
        self.total_selections += flat_indices.numel()
        
        # Update null routing stats
        if token_ids is not None:
            self._update_null_routing(token_ids, expert_indices)
        
        # Increment step
        self.step += 1
        
        # Log at interval
        if self.step % self.config.log_interval == 0:
            self._log_metrics(batch_size, seq_len, layer_idx)
    
    def _update_null_routing(
        self,
        token_ids: torch.Tensor,
        expert_indices: torch.Tensor
    ):
        """Update null routing statistics."""
        flat_tokens = token_ids.view(-1)
        flat_indices = expert_indices.view(-1, expert_indices.shape[-1])
        
        # Create junk mask
        junk_mask = torch.zeros_like(flat_tokens, dtype=torch.bool)
        for tid in self.junk_token_ids:
            junk_mask |= (flat_tokens == tid)
        
        signal_mask = ~junk_mask
        
        # Check null routing
        null_mask = torch.zeros(flat_indices.shape[0], dtype=torch.bool, device=flat_indices.device)
        for nid in self.null_expert_ids:
            null_mask |= (flat_indices == nid).any(dim=-1)
        
        null_mask = null_mask.cpu()
        junk_mask = junk_mask.cpu()
        signal_mask = signal_mask.cpu()
        
        # Update counts
        self.junk_total += junk_mask.sum().item()
        self.junk_to_null += (junk_mask & null_mask).sum().item()
        self.signal_total += signal_mask.sum().item()
        self.signal_to_null += (signal_mask & null_mask).sum().item()
    
    def _log_metrics(self, batch_size: int, seq_len: int, layer_idx: int):
        """Compute and log metrics at interval."""
        if self.total_selections == 0:
            return
        
        # Compute utilization
        utilization = self.expert_counts / self.total_selections
        util_list = utilization.tolist()
        
        # Compute entropy
        probs = utilization + 1e-10
        entropy = -(probs * probs.log()).sum().item()
        max_entropy = math.log(self.num_experts)
        normalized_entropy = entropy / max_entropy
        
        # Compute Gini coefficient
        sorted_util = torch.sort(utilization)[0]
        n = len(sorted_util)
        gini = ((2 * torch.arange(1, n+1).float() - n - 1) * sorted_util).sum()
        gini = (gini / (n * sorted_util.sum() + 1e-9)).item()
        
        # Null routing rates
        null_rate = None
        junk_null_rate = None
        signal_null_rate = None
        
        if self.num_null_experts > 0:
            null_rate = utilization[self.null_expert_ids].sum().item()
        
        if self.junk_total > 0:
            junk_null_rate = self.junk_to_null / self.junk_total
        
        if self.signal_total > 0:
            signal_null_rate = self.signal_to_null / self.signal_total
        
        # Check for alerts
        alerts = self._check_alerts(utilization, normalized_entropy, gini, junk_null_rate, signal_null_rate)
        
        # Create event
        event = RoutingEvent(
            timestamp=datetime.now().isoformat(),
            step=self.step,
            layer_idx=layer_idx,
            batch_size=batch_size,
            seq_len=seq_len,
            expert_utilization=util_list,
            entropy=normalized_entropy,
            gini=gini,
            null_rate=null_rate,
            junk_null_rate=junk_null_rate,
            signal_null_rate=signal_null_rate,
            alerts=alerts
        )
        
        # Store
        self.events.append(event)
        self.utilization_history.append(util_list)
        self.entropy_history.append(normalized_entropy)
        
        # Log
        if self.config.detailed_logging:
            logger.info(f"Step {self.step}: entropy={normalized_entropy:.3f}, gini={gini:.3f}, "
                       f"null_rate={null_rate:.3f if null_rate else 'N/A'}")
        
        # Handle alerts
        if alerts:
            for alert in alerts:
                logger.warning(f"Alert at step {self.step}: {alert}")
            for handler in self._alert_handlers:
                handler(alerts, event)
        
        # Reset counters
        self.expert_counts.zero_()
        self.total_selections = 0
        self.junk_total = 0
        self.junk_to_null = 0
        self.signal_total = 0
        self.signal_to_null = 0
    
    def _check_alerts(
        self,
        utilization: torch.Tensor,
        entropy: float,
        gini: float,
        junk_null_rate: Optional[float],
        signal_null_rate: Optional[float]
    ) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Dead experts
        if self.config.alert_on_dead_expert:
            dead = (utilization < self.config.dead_expert_threshold).nonzero().squeeze(-1).tolist()
            if dead:
                alerts.append(f"Dead experts detected: {dead}")
        
        # Low entropy (collapse)
        if self.config.alert_on_collapse:
            if entropy < self.config.min_entropy:
                alerts.append(f"Low routing entropy: {entropy:.3f} < {self.config.min_entropy}")
        
        # High Gini (imbalance)
        if gini > self.config.max_gini:
            alerts.append(f"High Gini coefficient: {gini:.3f} > {self.config.max_gini}")
        
        # Null routing violations
        if self.config.alert_on_null_violation:
            if junk_null_rate is not None:
                if junk_null_rate < self.config.junk_null_target[0]:
                    alerts.append(f"Junk→null rate too low: {junk_null_rate:.1%} < {self.config.junk_null_target[0]:.1%}")
                elif junk_null_rate > self.config.junk_null_target[1]:
                    alerts.append(f"Junk→null rate too high: {junk_null_rate:.1%} > {self.config.junk_null_target[1]:.1%}")
            
            if signal_null_rate is not None:
                if signal_null_rate > self.config.signal_null_target_max:
                    alerts.append(f"Signal→null rate too high: {signal_null_rate:.1%} > {self.config.signal_null_target_max:.1%}")
        
        return alerts
    
    # =========================================================================
    # Health Check Methods
    # =========================================================================
    
    def check_health(self) -> Dict[str, Any]:
        """
        Comprehensive health check.
        
        Returns dict with:
        - is_healthy: bool
        - alerts: List[str]
        - metrics: Dict of current metrics
        - recommendations: List[str]
        """
        health = {
            'is_healthy': True,
            'alerts': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Get recent metrics
        if self.events:
            recent = self.events[-1]
            health['metrics'] = {
                'entropy': recent.entropy,
                'gini': recent.gini,
                'null_rate': recent.null_rate,
                'junk_null_rate': recent.junk_null_rate,
                'signal_null_rate': recent.signal_null_rate,
            }
            health['alerts'] = recent.alerts
            health['is_healthy'] = len(recent.alerts) == 0
        
        # Add recommendations
        if not health['is_healthy']:
            if any('Dead experts' in a for a in health['alerts']):
                health['recommendations'].append(
                    "Increase dead expert bias boost or enable auto-revival"
                )
            
            if any('entropy' in a.lower() for a in health['alerts']):
                health['recommendations'].append(
                    "Check for router collapse - may need to increase noise or adjust initialization"
                )
            
            if any('junk' in a.lower() for a in health['alerts']):
                health['recommendations'].append(
                    "Adjust null expert bias to better capture junk tokens"
                )
        
        # Run custom health check hooks
        for hook in self._health_check_hooks:
            hook_result = hook(health)
            if hook_result:
                health['alerts'].extend(hook_result.get('alerts', []))
                health['recommendations'].extend(hook_result.get('recommendations', []))
        
        return health
    
    # =========================================================================
    # Plugin Interface (Team 7 Control Hooks)
    # =========================================================================
    
    def register_pre_routing_hook(self, hook: Callable):
        """
        Register hook called before routing.
        
        Hook signature: hook(hidden_states: Tensor) -> Tensor
        Can modify hidden states before routing.
        """
        self._pre_routing_hooks.append(hook)
    
    def register_post_routing_hook(self, hook: Callable):
        """
        Register hook called after routing.
        
        Hook signature: hook(indices: Tensor, weights: Tensor, aux_info: Dict) -> None
        For monitoring and logging.
        """
        self._post_routing_hooks.append(hook)
    
    def register_health_check_hook(self, hook: Callable):
        """
        Register custom health check hook.
        
        Hook signature: hook(health: Dict) -> Dict with 'alerts' and 'recommendations'
        """
        self._health_check_hooks.append(hook)
    
    def register_alert_handler(self, handler: Callable):
        """
        Register alert handler.
        
        Handler signature: handler(alerts: List[str], event: RoutingEvent) -> None
        """
        self._alert_handlers.append(handler)
    
    def apply_pre_routing_hooks(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply all pre-routing hooks."""
        for hook in self._pre_routing_hooks:
            hidden_states = hook(hidden_states)
        return hidden_states
    
    def apply_post_routing_hooks(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        aux_info: Dict
    ):
        """Apply all post-routing hooks."""
        for hook in self._post_routing_hooks:
            hook(indices, weights, aux_info)
    
    # =========================================================================
    # Export Methods
    # =========================================================================
    
    def export_metrics(self, path: Optional[str] = None) -> str:
        """Export metrics to JSON file."""
        path = path or self.config.export_path or f"moe_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'config': {
                'num_experts': self.num_experts,
                'num_null_experts': self.num_null_experts,
                'null_expert_ids': self.null_expert_ids,
            },
            'events': [
                {
                    'timestamp': e.timestamp,
                    'step': e.step,
                    'layer_idx': e.layer_idx,
                    'entropy': e.entropy,
                    'gini': e.gini,
                    'null_rate': e.null_rate,
                    'junk_null_rate': e.junk_null_rate,
                    'signal_null_rate': e.signal_null_rate,
                    'alerts': e.alerts,
                }
                for e in self.events
            ],
            'summary': self.get_summary()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {path}")
        return path
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.events:
            return {'status': 'no_data'}
        
        entropies = [e.entropy for e in self.events]
        ginis = [e.gini for e in self.events]
        
        return {
            'total_steps': self.step,
            'total_events': len(self.events),
            'entropy_mean': sum(entropies) / len(entropies),
            'entropy_min': min(entropies),
            'entropy_max': max(entropies),
            'gini_mean': sum(ginis) / len(ginis),
            'gini_max': max(ginis),
            'total_alerts': sum(len(e.alerts) for e in self.events),
        }
    
    def reset(self):
        """Reset all tracking state."""
        self.step = 0
        self.expert_counts.zero_()
        self.total_selections = 0
        self.junk_total = 0
        self.junk_to_null = 0
        self.signal_total = 0
        self.signal_to_null = 0
        self.events.clear()
        self.utilization_history.clear()
        self.entropy_history.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_telemetry(
    config: Any,  # MoEModelConfig
) -> MoETelemetrySystem:
    """Create telemetry system from model config."""
    num_null = getattr(config.expert, 'num_null_experts', 0) if hasattr(config, 'expert') else 0
    num_routed = getattr(config.expert, 'num_routed_experts', 8) if hasattr(config, 'expert') else 8
    
    null_ids = list(range(num_routed, num_routed + num_null))
    junk_ids = [0]  # Default: padding token
    
    telem_config = TelemetryConfig(
        enabled=True,
        log_interval=100,
    )
    
    return MoETelemetrySystem(
        num_experts=num_routed + num_null,
        num_null_experts=num_null,
        null_expert_ids=null_ids,
        junk_token_ids=junk_ids,
        config=telem_config
    )
