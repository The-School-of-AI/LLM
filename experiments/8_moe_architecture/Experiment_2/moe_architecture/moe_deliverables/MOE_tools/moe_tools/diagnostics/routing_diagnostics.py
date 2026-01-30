#!/usr/bin/env python3
"""
MoE Routing Diagnostics
=======================
Comprehensive routing analysis for Team 7's diagnostics needs:
- Null expert utilization tracking
- Token-to-expert mapping analysis
- Curriculum bucket (B0-B5) routing patterns
- Router head specialization detection
- Health gates and alerts

This module provides the telemetry stream and metrics for:
1. Deciding when to grow experts
2. Deciding when MoE blocks are LoRA-ready (stability milestones)
3. Validating null expert compute savings

Usage:
    from diagnostics.routing_diagnostics import RoutingDiagnostics
    
    diagnostics = RoutingDiagnostics(config)
    diagnostics.log_batch(expert_indices, expert_weights, token_ids, ...)
    report = diagnostics.get_dashboard_metrics()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from enum import Enum
import json
import time


class CurriculumBucket(Enum):
    """Curriculum difficulty buckets."""
    B0_TRIVIAL = 0       # Simple patterns, whitespace, punctuation
    B1_BASIC = 1         # Common words, basic syntax
    B2_INTERMEDIATE = 2  # Standard content
    B3_ADVANCED = 3      # Complex reasoning, technical
    B4_EXPERT = 4        # Domain expertise required
    B5_FRONTIER = 5      # Novel/challenging content


class TokenType(Enum):
    """Token classification for routing analysis."""
    JUNK = "junk"           # PAD, special tokens, whitespace
    BOILERPLATE = "boilerplate"  # Template fragments, repeated patterns
    SIGNAL = "signal"       # Meaningful content tokens


@dataclass
class RoutingConfig:
    """Configuration for routing diagnostics."""
    
    # Expert configuration
    num_routed_experts: int = 64
    num_shared_experts: int = 4
    num_null_experts: int = 2
    top_k: int = 4
    num_layers: int = 80
    
    # Null expert indices (last N in routing pool)
    null_expert_start_idx: int = 64  # null experts are [64, 65]
    
    # Token classification
    junk_token_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])  # PAD, BOS, EOS, UNK
    boilerplate_patterns: List[str] = field(default_factory=lambda: [
        "\\n\\n", "\\t", "  ", "---", "===", "```", "<!--", "-->"
    ])
    
    # Health gate thresholds
    min_null_junk_rate: float = 0.60      # Junk tokens → null (min 60%)
    max_null_junk_rate: float = 0.80      # Junk tokens → null (max 80%)
    max_null_signal_rate: float = 0.10    # Signal tokens → null (max 10%)
    min_routing_entropy: float = 0.70     # Normalized entropy
    max_gini_coefficient: float = 0.50    # Load balance
    dead_expert_threshold: float = 0.01   # <1% = dead
    overload_threshold: float = 3.0       # >3x expected = overloaded
    
    # Stability detection
    stability_window: int = 1000          # Steps for stability check
    stability_variance_threshold: float = 0.05  # Max variance for stable


@dataclass
class LayerRoutingStats:
    """Per-layer routing statistics."""
    
    layer_idx: int = 0
    
    # Expert utilization counts
    expert_counts: Dict[int, int] = field(default_factory=dict)
    
    # Null routing
    null_total: int = 0
    null_from_junk: int = 0
    null_from_boilerplate: int = 0
    null_from_signal: int = 0
    
    # Token type counts
    junk_total: int = 0
    boilerplate_total: int = 0
    signal_total: int = 0
    
    # Router head activations per expert
    head_expert_affinity: Dict[int, Dict[int, float]] = field(default_factory=dict)
    
    # Entropy and balance
    routing_entropy: float = 0.0
    gini_coefficient: float = 0.0


@dataclass
class RoutingSnapshot:
    """Point-in-time routing snapshot for dashboard."""
    
    timestamp: float = 0.0
    step: int = 0
    
    # Global metrics
    null_junk_rate: float = 0.0
    null_boilerplate_rate: float = 0.0
    null_signal_rate: float = 0.0
    avg_routing_entropy: float = 0.0
    avg_gini_coefficient: float = 0.0
    
    # Expert health
    dead_experts: List[int] = field(default_factory=list)
    overloaded_experts: List[int] = field(default_factory=list)
    
    # Compute savings
    null_compute_savings_pct: float = 0.0
    
    # Health gate status
    health_gates: Dict[str, bool] = field(default_factory=dict)
    
    # Stability
    is_stable: bool = False
    stability_score: float = 0.0


class RoutingDiagnostics:
    """
    MoE Routing Diagnostics System.
    
    Provides:
    1. Real-time routing telemetry
    2. Null expert analysis (Team 7 core requirement)
    3. Token family → expert mapping
    4. Curriculum bucket tracking (B0-B5)
    5. Health gates and alerts
    6. Stability milestones for LoRA-readiness
    7. Expert growth triggers
    """
    
    def __init__(self, config: RoutingConfig):
        self.config = config
        
        # Per-layer stats (current window)
        self.layer_stats: Dict[int, LayerRoutingStats] = {}
        
        # Historical snapshots
        self.history: List[RoutingSnapshot] = []
        self.max_history = 1000
        
        # Token ID → expert affinity mapping
        self.token_expert_affinity: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Curriculum bucket → expert mapping
        self.bucket_expert_affinity: Dict[CurriculumBucket, Dict[int, int]] = {
            b: defaultdict(int) for b in CurriculumBucket
        }
        
        # Router head specialization tracking
        self.head_specialization: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Stability tracking
        self.stability_buffer: List[float] = []
        self.current_step = 0
        
        # Alerts
        self.active_alerts: List[Dict] = []
        
        # Initialize layer stats
        for layer_idx in range(config.num_layers):
            self.layer_stats[layer_idx] = LayerRoutingStats(layer_idx=layer_idx)
    
    def _classify_token(self, token_id: int, token_text: Optional[str] = None) -> TokenType:
        """Classify token as junk, boilerplate, or signal."""
        if token_id in self.config.junk_token_ids:
            return TokenType.JUNK
        
        if token_text:
            for pattern in self.config.boilerplate_patterns:
                if pattern in token_text:
                    return TokenType.BOILERPLATE
        
        return TokenType.SIGNAL
    
    def _classify_curriculum_bucket(
        self, 
        token_id: int, 
        position: int, 
        context_difficulty: Optional[float] = None
    ) -> CurriculumBucket:
        """
        Classify token into curriculum bucket.
        
        This is a simplified heuristic - in production, this would
        use the actual curriculum classification from the data pipeline.
        """
        if token_id in self.config.junk_token_ids:
            return CurriculumBucket.B0_TRIVIAL
        
        if context_difficulty is not None:
            if context_difficulty < 0.2:
                return CurriculumBucket.B1_BASIC
            elif context_difficulty < 0.4:
                return CurriculumBucket.B2_INTERMEDIATE
            elif context_difficulty < 0.6:
                return CurriculumBucket.B3_ADVANCED
            elif context_difficulty < 0.8:
                return CurriculumBucket.B4_EXPERT
            else:
                return CurriculumBucket.B5_FRONTIER
        
        return CurriculumBucket.B2_INTERMEDIATE
    
    def _is_null_expert(self, expert_idx: int) -> bool:
        """Check if expert index is a null expert."""
        return expert_idx >= self.config.null_expert_start_idx
    
    def _compute_entropy(self, counts: Dict[int, int]) -> float:
        """Compute normalized routing entropy."""
        total = sum(counts.values())
        if total == 0:
            return 0.0
        
        import math
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize by max entropy
        max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_gini(self, counts: Dict[int, int]) -> float:
        """Compute Gini coefficient for load balance."""
        values = sorted(counts.values())
        n = len(values)
        if n == 0:
            return 0.0
        
        total = sum(values)
        if total == 0:
            return 0.0
        
        cumsum = 0
        weighted_sum = 0
        for i, v in enumerate(values):
            cumsum += v
            weighted_sum += (i + 1) * v
        
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n
        return gini
    
    def log_batch(
        self,
        layer_idx: int,
        expert_indices: List[List[int]],  # [batch * seq, top_k]
        expert_weights: List[List[float]],  # [batch * seq, top_k]
        token_ids: List[int],  # [batch * seq]
        token_texts: Optional[List[str]] = None,
        context_difficulties: Optional[List[float]] = None,
        router_head_scores: Optional[Dict[int, List[float]]] = None,  # head -> [expert scores]
    ):
        """
        Log routing decisions for a batch.
        
        This is the main entry point for telemetry collection.
        Call this for each layer during forward pass.
        """
        stats = self.layer_stats[layer_idx]
        
        for i, (indices, weights, token_id) in enumerate(zip(
            expert_indices, expert_weights, token_ids
        )):
            # Classify token
            token_text = token_texts[i] if token_texts else None
            token_type = self._classify_token(token_id, token_text)
            
            # Curriculum bucket
            context_diff = context_difficulties[i] if context_difficulties else None
            bucket = self._classify_curriculum_bucket(token_id, i, context_diff)
            
            # Update token type counts
            if token_type == TokenType.JUNK:
                stats.junk_total += 1
            elif token_type == TokenType.BOILERPLATE:
                stats.boilerplate_total += 1
            else:
                stats.signal_total += 1
            
            # Track expert routing
            for expert_idx, weight in zip(indices, weights):
                # Expert utilization
                stats.expert_counts[expert_idx] = stats.expert_counts.get(expert_idx, 0) + 1
                
                # Token → expert affinity
                self.token_expert_affinity[token_id][expert_idx] += 1
                
                # Bucket → expert affinity
                self.bucket_expert_affinity[bucket][expert_idx] += 1
                
                # Null routing tracking
                if self._is_null_expert(expert_idx):
                    stats.null_total += 1
                    if token_type == TokenType.JUNK:
                        stats.null_from_junk += 1
                    elif token_type == TokenType.BOILERPLATE:
                        stats.null_from_boilerplate += 1
                    else:
                        stats.null_from_signal += 1
        
        # Track router head specialization
        if router_head_scores:
            for head_idx, scores in router_head_scores.items():
                for expert_idx, score in enumerate(scores):
                    self.head_specialization[head_idx][expert_idx] += score
        
        # Update entropy and Gini
        stats.routing_entropy = self._compute_entropy(stats.expert_counts)
        stats.gini_coefficient = self._compute_gini(stats.expert_counts)
    
    def step(self):
        """
        Called at end of training step.
        Computes aggregate metrics and updates history.
        """
        self.current_step += 1
        snapshot = self._compute_snapshot()
        
        # Add to history
        self.history.append(snapshot)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Update stability tracking
        self.stability_buffer.append(snapshot.null_junk_rate)
        if len(self.stability_buffer) > self.config.stability_window:
            self.stability_buffer.pop(0)
        
        # Check health gates
        self._check_health_gates(snapshot)
        
        # Reset per-step stats
        self._reset_layer_stats()
        
        return snapshot
    
    def _compute_snapshot(self) -> RoutingSnapshot:
        """Compute current routing snapshot."""
        snapshot = RoutingSnapshot(
            timestamp=time.time(),
            step=self.current_step,
        )
        
        # Aggregate across layers
        total_junk = sum(s.junk_total for s in self.layer_stats.values())
        total_boilerplate = sum(s.boilerplate_total for s in self.layer_stats.values())
        total_signal = sum(s.signal_total for s in self.layer_stats.values())
        
        null_from_junk = sum(s.null_from_junk for s in self.layer_stats.values())
        null_from_boilerplate = sum(s.null_from_boilerplate for s in self.layer_stats.values())
        null_from_signal = sum(s.null_from_signal for s in self.layer_stats.values())
        null_total = sum(s.null_total for s in self.layer_stats.values())
        
        # Null routing rates
        snapshot.null_junk_rate = null_from_junk / max(total_junk, 1)
        snapshot.null_boilerplate_rate = null_from_boilerplate / max(total_boilerplate, 1)
        snapshot.null_signal_rate = null_from_signal / max(total_signal, 1)
        
        # Average entropy and Gini
        entropies = [s.routing_entropy for s in self.layer_stats.values()]
        ginis = [s.gini_coefficient for s in self.layer_stats.values()]
        snapshot.avg_routing_entropy = sum(entropies) / len(entropies) if entropies else 0
        snapshot.avg_gini_coefficient = sum(ginis) / len(ginis) if ginis else 0
        
        # Expert health
        total_tokens = total_junk + total_boilerplate + total_signal
        expected_per_expert = total_tokens * self.config.top_k / self.config.num_routed_experts
        
        all_expert_counts = defaultdict(int)
        for stats in self.layer_stats.values():
            for exp_idx, count in stats.expert_counts.items():
                all_expert_counts[exp_idx] += count
        
        for exp_idx in range(self.config.num_routed_experts):
            count = all_expert_counts.get(exp_idx, 0)
            utilization = count / max(expected_per_expert * self.config.num_layers, 1)
            
            if utilization < self.config.dead_expert_threshold:
                snapshot.dead_experts.append(exp_idx)
            elif utilization > self.config.overload_threshold:
                snapshot.overloaded_experts.append(exp_idx)
        
        # Compute savings from null routing
        total_routed = total_tokens * self.config.top_k * self.config.num_layers
        snapshot.null_compute_savings_pct = null_total / max(total_routed, 1) * 100
        
        # Stability check
        if len(self.stability_buffer) >= self.config.stability_window:
            import statistics
            variance = statistics.variance(self.stability_buffer)
            snapshot.stability_score = 1.0 - min(variance / self.config.stability_variance_threshold, 1.0)
            snapshot.is_stable = variance < self.config.stability_variance_threshold
        
        return snapshot
    
    def _check_health_gates(self, snapshot: RoutingSnapshot):
        """Check health gates and raise alerts."""
        gates = {}
        
        # Null routing gates
        gates['null_junk_min'] = snapshot.null_junk_rate >= self.config.min_null_junk_rate
        gates['null_junk_max'] = snapshot.null_junk_rate <= self.config.max_null_junk_rate
        gates['null_signal_max'] = snapshot.null_signal_rate <= self.config.max_null_signal_rate
        
        # Routing quality gates
        gates['entropy_min'] = snapshot.avg_routing_entropy >= self.config.min_routing_entropy
        gates['gini_max'] = snapshot.avg_gini_coefficient <= self.config.max_gini_coefficient
        
        # Expert health gates
        gates['no_dead_experts'] = len(snapshot.dead_experts) == 0
        gates['no_overloaded_experts'] = len(snapshot.overloaded_experts) == 0
        
        snapshot.health_gates = gates
        
        # Generate alerts for failures
        self.active_alerts = []
        
        if not gates['null_junk_min']:
            self.active_alerts.append({
                'severity': 'WARNING',
                'gate': 'null_junk_min',
                'message': f"Null routing for junk tokens too low: {snapshot.null_junk_rate:.1%} < {self.config.min_null_junk_rate:.1%}",
                'recommendation': "Increase null expert bias or check token classification"
            })
        
        if not gates['null_signal_max']:
            self.active_alerts.append({
                'severity': 'WARNING',
                'gate': 'null_signal_max',
                'message': f"Null routing for signal tokens too high: {snapshot.null_signal_rate:.1%} > {self.config.max_null_signal_rate:.1%}",
                'recommendation': "Reduce null expert bias or check router learning"
            })
        
        if not gates['entropy_min']:
            self.active_alerts.append({
                'severity': 'CRITICAL',
                'gate': 'entropy_min',
                'message': f"Router entropy too low (possible collapse): {snapshot.avg_routing_entropy:.2f} < {self.config.min_routing_entropy:.2f}",
                'recommendation': "Check for expert collapse, consider router reset"
            })
        
        if not gates['no_dead_experts']:
            self.active_alerts.append({
                'severity': 'WARNING',
                'gate': 'no_dead_experts',
                'message': f"Dead experts detected: {snapshot.dead_experts}",
                'recommendation': "Boost bias for dead experts or check initialization"
            })
    
    def _reset_layer_stats(self):
        """Reset per-step layer statistics."""
        for layer_idx in self.layer_stats:
            self.layer_stats[layer_idx] = LayerRoutingStats(layer_idx=layer_idx)
    
    def get_dashboard_metrics(self) -> Dict:
        """
        Get metrics formatted for Team 7 dashboard.
        
        Returns:
            Dictionary of dashboard-ready metrics
        """
        if not self.history:
            return {'status': 'no_data'}
        
        latest = self.history[-1]
        
        # Trend data (last 100 steps)
        trend_window = min(100, len(self.history))
        recent = self.history[-trend_window:]
        
        return {
            'timestamp': latest.timestamp,
            'step': latest.step,
            
            # Core null expert metrics (Team 7 focus)
            'null_expert': {
                'junk_to_null_rate': f"{latest.null_junk_rate:.1%}",
                'boilerplate_to_null_rate': f"{latest.null_boilerplate_rate:.1%}",
                'signal_to_null_rate': f"{latest.null_signal_rate:.1%}",
                'compute_savings_pct': f"{latest.null_compute_savings_pct:.1f}%",
                'target_junk_rate': f"{self.config.min_null_junk_rate:.0%}-{self.config.max_null_junk_rate:.0%}",
            },
            
            # Routing health
            'routing_health': {
                'entropy': f"{latest.avg_routing_entropy:.3f}",
                'gini_coefficient': f"{latest.avg_gini_coefficient:.3f}",
                'dead_experts': latest.dead_experts,
                'overloaded_experts': latest.overloaded_experts,
            },
            
            # Stability (for LoRA-readiness)
            'stability': {
                'is_stable': latest.is_stable,
                'stability_score': f"{latest.stability_score:.2f}",
                'lora_ready': latest.is_stable and all(latest.health_gates.values()),
            },
            
            # Health gates
            'health_gates': latest.health_gates,
            'all_gates_pass': all(latest.health_gates.values()),
            
            # Alerts
            'alerts': self.active_alerts,
            
            # Trends
            'trends': {
                'null_junk_rate': [s.null_junk_rate for s in recent],
                'entropy': [s.avg_routing_entropy for s in recent],
                'steps': [s.step for s in recent],
            },
            
            # Expert growth trigger
            'growth_trigger': self._check_growth_trigger(),
        }
    
    def _check_growth_trigger(self) -> Dict:
        """
        Check if conditions are met for expert growth.
        
        Returns recommendation for expert expansion.
        """
        if len(self.history) < 100:
            return {
                'recommend_growth': False,
                'reason': 'Insufficient history for growth decision',
                'confidence': 0.0
            }
        
        latest = self.history[-1]
        
        # Growth conditions:
        # 1. Stable routing (no collapse)
        # 2. High expert utilization (no dead experts)
        # 3. Balanced load
        # 4. Sustained performance
        
        conditions = {
            'stable': latest.is_stable,
            'no_dead': len(latest.dead_experts) == 0,
            'balanced': latest.avg_gini_coefficient < self.config.max_gini_coefficient,
            'healthy_null': self.config.min_null_junk_rate <= latest.null_junk_rate <= self.config.max_null_junk_rate,
        }
        
        passed = sum(conditions.values())
        confidence = passed / len(conditions)
        
        return {
            'recommend_growth': passed == len(conditions),
            'conditions': conditions,
            'confidence': confidence,
            'reason': 'All conditions met' if passed == len(conditions) else f"{len(conditions) - passed} conditions not met"
        }
    
    def get_token_expert_map(self, top_n: int = 10) -> Dict:
        """
        Get top token → expert mappings.
        
        Useful for understanding which tokens prefer which experts.
        """
        result = {}
        
        for token_id, expert_counts in list(self.token_expert_affinity.items())[:100]:
            sorted_experts = sorted(
                expert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            result[token_id] = {
                'top_experts': [e[0] for e in sorted_experts],
                'counts': [e[1] for e in sorted_experts],
            }
        
        return result
    
    def get_bucket_expert_map(self) -> Dict:
        """
        Get curriculum bucket → expert mapping.
        
        Shows how routing specializes by content difficulty.
        """
        result = {}
        
        for bucket in CurriculumBucket:
            expert_counts = self.bucket_expert_affinity[bucket]
            total = sum(expert_counts.values())
            
            if total == 0:
                continue
            
            sorted_experts = sorted(
                expert_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            result[bucket.name] = {
                'total_tokens': total,
                'top_experts': [e[0] for e in sorted_experts],
                'percentages': [e[1] / total * 100 for e in sorted_experts],
            }
        
        return result
    
    def export_telemetry(self, filepath: str):
        """Export telemetry data to JSON."""
        data = {
            'config': {
                'num_routed_experts': self.config.num_routed_experts,
                'num_shared_experts': self.config.num_shared_experts,
                'num_null_experts': self.config.num_null_experts,
                'top_k': self.config.top_k,
            },
            'history': [
                {
                    'step': s.step,
                    'timestamp': s.timestamp,
                    'null_junk_rate': s.null_junk_rate,
                    'null_signal_rate': s.null_signal_rate,
                    'entropy': s.avg_routing_entropy,
                    'gini': s.avg_gini_coefficient,
                    'compute_savings': s.null_compute_savings_pct,
                    'is_stable': s.is_stable,
                    'health_gates': s.health_gates,
                }
                for s in self.history
            ],
            'token_expert_map': self.get_token_expert_map(),
            'bucket_expert_map': self.get_bucket_expert_map(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Factory function
def create_diagnostics(model_size: str) -> RoutingDiagnostics:
    """Create diagnostics for standard model sizes."""
    configs = {
        '3b_moe': RoutingConfig(
            num_routed_experts=8,
            num_shared_experts=2,
            num_null_experts=1,
            top_k=2,
            num_layers=24,
            null_expert_start_idx=8,
        ),
        '8b_moe': RoutingConfig(
            num_routed_experts=8,
            num_shared_experts=2,
            num_null_experts=1,
            top_k=2,
            num_layers=40,
            null_expert_start_idx=8,
        ),
        '70b_moe': RoutingConfig(
            num_routed_experts=64,
            num_shared_experts=4,
            num_null_experts=2,
            top_k=4,
            num_layers=40,
            null_expert_start_idx=64,
        ),
    }
    
    config = configs.get(model_size, configs['70b_moe'])
    return RoutingDiagnostics(config)


if __name__ == "__main__":
    # Demo usage
    import random
    
    diagnostics = create_diagnostics('3b_moe')
    
    # Simulate training loop
    for step in range(100):
        for layer_idx in range(24):
            # Simulate batch of 32 tokens
            batch_size = 32
            
            expert_indices = [
                random.sample(range(9), 2)  # 8 routed + 1 null
                for _ in range(batch_size)
            ]
            expert_weights = [
                [random.random() for _ in range(2)]
                for _ in range(batch_size)
            ]
            token_ids = [
                random.choice([0, 1, 2, 3] + list(range(100, 1000)))
                for _ in range(batch_size)
            ]
            
            diagnostics.log_batch(
                layer_idx=layer_idx,
                expert_indices=expert_indices,
                expert_weights=expert_weights,
                token_ids=token_ids,
            )
        
        # End of step
        snapshot = diagnostics.step()
        
        if step % 20 == 0:
            print(f"Step {step}: null_junk={snapshot.null_junk_rate:.1%}, entropy={snapshot.avg_routing_entropy:.2f}")
    
    # Get dashboard metrics
    metrics = diagnostics.get_dashboard_metrics()
    print("\n📊 Dashboard Metrics:")
    print(json.dumps(metrics, indent=2, default=str))
