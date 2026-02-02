#!/usr/bin/env python3
"""
Weights & Biases Integration for MoE Routing Dashboard
======================================================
Production-ready W&B integration for Team 7's null-expert monitoring.

Features:
- Real-time metric streaming
- Custom dashboards with charts
- Team collaboration
- Alerting via Slack/Email
- Historical comparison
- Cloud-native (no infrastructure)

Setup:
    pip install wandb
    wandb login

Usage:
    from moe_dashboard.wandb_dashboard import WandBDashboard
    
    dashboard = WandBDashboard(project="moe-training", run_name="70b-exp1")
    
    # In training loop
    dashboard.log_routing_metrics(diagnostics.get_dashboard_metrics())
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Run: pip install wandb")


@dataclass
class WandBConfig:
    """W&B configuration."""
    project: str = "moe-routing-monitor"
    entity: Optional[str] = None  # Team/org name
    run_name: Optional[str] = None
    tags: List[str] = None
    
    # Alert thresholds
    alert_on_entropy_collapse: bool = True
    alert_on_dead_experts: bool = True
    alert_on_signal_leakage: bool = True
    
    # Logging frequency
    log_every_n_steps: int = 1
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["moe", "routing", "team7"]


class WandBDashboard:
    """
    Weights & Biases Dashboard for MoE Routing Monitoring.
    
    Provides:
    1. Real-time metric streaming to W&B cloud
    2. Custom dashboard panels
    3. Automatic alerting
    4. Team sharing and collaboration
    5. Historical run comparison
    """
    
    def __init__(self, config: WandBConfig = None, diagnostics=None):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Run: pip install wandb")
        
        self.config = config or WandBConfig()
        self.diagnostics = diagnostics
        self.run = None
        self.step = 0
        self._alert_cooldown = {}
        
    def init_run(self, model_config: Dict = None):
        """Initialize W&B run with model configuration."""
        self.run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            name=self.config.run_name,
            tags=self.config.tags,
            config=model_config or {},
            reinit=True,
        )
        
        # Define custom charts
        self._setup_custom_charts()
        
        print(f"📊 W&B Dashboard initialized: {self.run.url}")
        return self.run
    
    def _setup_custom_charts(self):
        """Setup custom W&B charts and panels."""
        # Define custom x-axis for all metrics
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")
    
    def log_routing_metrics(self, metrics: Dict, step: int = None):
        """
        Log routing metrics to W&B.
        
        Args:
            metrics: Dictionary from diagnostics.get_dashboard_metrics()
            step: Training step (auto-incremented if not provided)
        """
        if self.run is None:
            self.init_run()
        
        if step is None:
            step = self.step
            self.step += 1
        
        # Parse metrics
        null_expert = metrics.get('null_expert', {})
        routing_health = metrics.get('routing_health', {})
        stability = metrics.get('stability', {})
        
        # Helper to parse percentage strings
        def parse_pct(s):
            if isinstance(s, str):
                return float(s.replace('%', ''))
            return float(s) if s else 0.0
        
        # Core metrics
        log_data = {
            "step": step,
            
            # Null Expert Metrics (Team 7 Focus)
            "null/junk_to_null_rate": parse_pct(null_expert.get('junk_to_null_rate', 0)),
            "null/boilerplate_to_null_rate": parse_pct(null_expert.get('boilerplate_to_null_rate', 0)),
            "null/signal_to_null_rate": parse_pct(null_expert.get('signal_to_null_rate', 0)),
            "null/compute_savings_pct": parse_pct(null_expert.get('compute_savings_pct', 0)),
            
            # Routing Health
            "health/entropy": float(routing_health.get('entropy', 0)),
            "health/gini_coefficient": float(routing_health.get('gini_coefficient', 0)),
            "health/dead_expert_count": len(routing_health.get('dead_experts', [])),
            "health/overloaded_expert_count": len(routing_health.get('overloaded_experts', [])),
            
            # Stability
            "stability/score": float(stability.get('stability_score', 0)),
            "stability/is_stable": 1 if stability.get('is_stable') else 0,
            "stability/lora_ready": 1 if stability.get('lora_ready') else 0,
            
            # Health Gates
            "gates/all_pass": 1 if metrics.get('all_gates_pass') else 0,
        }
        
        # Log health gates individually
        for gate_name, gate_value in metrics.get('health_gates', {}).items():
            log_data[f"gates/{gate_name}"] = 1 if gate_value else 0
        
        # Log to W&B
        wandb.log(log_data)
        
        # Check for alerts
        self._check_alerts(log_data, step)
        
        return log_data
    
    def log_expert_utilization(self, expert_counts: Dict[int, int], step: int = None):
        """Log per-expert utilization as a bar chart."""
        if self.run is None:
            return
        
        # Create bar chart data
        data = [[exp_id, count] for exp_id, count in sorted(expert_counts.items())]
        table = wandb.Table(data=data, columns=["Expert ID", "Token Count"])
        
        wandb.log({
            "expert_utilization": wandb.plot.bar(
                table, "Expert ID", "Token Count",
                title="Expert Utilization Distribution"
            ),
            "step": step or self.step,
        })
    
    def log_curriculum_heatmap(self, bucket_expert_map: Dict, step: int = None):
        """Log curriculum bucket to expert routing heatmap."""
        if self.run is None:
            return
        
        # Create heatmap data
        buckets = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5']
        num_experts = 64  # Adjust as needed
        
        data = []
        for bucket in buckets:
            bucket_data = bucket_expert_map.get(f"{bucket}_TRIVIAL", {}) or \
                         bucket_expert_map.get(bucket, {})
            for exp_id in range(min(num_experts, 16)):  # Show first 16
                pct = bucket_data.get('percentages', [0] * 16)[exp_id] if exp_id < len(bucket_data.get('percentages', [])) else 0
                data.append([bucket, f"E{exp_id}", pct])
        
        table = wandb.Table(data=data, columns=["Bucket", "Expert", "Percentage"])
        
        wandb.log({
            "curriculum_routing": wandb.plot.scatter(
                table, "Expert", "Bucket", 
                title="Curriculum Bucket → Expert Routing"
            ),
            "step": step or self.step,
        })
    
    def _check_alerts(self, metrics: Dict, step: int):
        """Check metrics and trigger W&B alerts."""
        alerts = []
        
        # Entropy collapse alert
        if self.config.alert_on_entropy_collapse:
            entropy = metrics.get('health/entropy', 1.0)
            if entropy < 0.5:
                alerts.append({
                    'title': '🚨 Router Entropy Collapse',
                    'text': f'Entropy dropped to {entropy:.3f} at step {step}. Check for expert collapse!',
                    'level': 'ERROR',
                })
        
        # Dead experts alert
        if self.config.alert_on_dead_experts:
            dead_count = metrics.get('health/dead_expert_count', 0)
            if dead_count > 0:
                alerts.append({
                    'title': '⚠️ Dead Experts Detected',
                    'text': f'{dead_count} experts have <1% utilization at step {step}',
                    'level': 'WARN',
                })
        
        # Signal leakage alert
        if self.config.alert_on_signal_leakage:
            signal_null = metrics.get('null/signal_to_null_rate', 0)
            if signal_null > 15:
                alerts.append({
                    'title': '⚠️ High Signal Leakage to Null',
                    'text': f'{signal_null:.1f}% signal tokens routed to null at step {step}',
                    'level': 'WARN',
                })
        
        # Send alerts (with cooldown to avoid spam)
        for alert in alerts:
            alert_key = alert['title']
            last_alert = self._alert_cooldown.get(alert_key, 0)
            if step - last_alert > 100:  # Alert cooldown: 100 steps
                wandb.alert(
                    title=alert['title'],
                    text=alert['text'],
                    level=getattr(wandb.AlertLevel, alert['level']),
                )
                self._alert_cooldown[alert_key] = step
    
    def log_training_metrics(self, loss: float, lr: float, throughput: float, step: int = None):
        """Log standard training metrics."""
        wandb.log({
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/throughput_tokens_per_sec": throughput,
            "step": step or self.step,
        })
    
    def create_summary_table(self, metrics_history: List[Dict]):
        """Create a summary table for the run."""
        if not metrics_history:
            return
        
        # Compute averages
        avg_metrics = {}
        for key in metrics_history[0].keys():
            if isinstance(metrics_history[0][key], (int, float)):
                values = [m[key] for m in metrics_history if key in m]
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        wandb.summary.update(avg_metrics)
    
    def finish(self):
        """Finish W&B run."""
        if self.run:
            wandb.finish()
            print("📊 W&B run completed")


def create_wandb_dashboard_config() -> Dict:
    """
    Generate W&B dashboard configuration that can be imported.
    
    This creates a shareable dashboard template.
    """
    return {
        "name": "Team 7 - MoE Routing Monitor",
        "description": "Real-time monitoring of null expert routing and MoE health",
        "panels": [
            {
                "title": "Null Expert Routing Rates",
                "type": "line",
                "metrics": [
                    "null/junk_to_null_rate",
                    "null/signal_to_null_rate",
                    "null/boilerplate_to_null_rate"
                ],
                "layout": {"x": 0, "y": 0, "w": 12, "h": 8}
            },
            {
                "title": "Routing Health",
                "type": "line",
                "metrics": [
                    "health/entropy",
                    "health/gini_coefficient"
                ],
                "layout": {"x": 0, "y": 8, "w": 6, "h": 6}
            },
            {
                "title": "Stability & Milestones",
                "type": "line",
                "metrics": [
                    "stability/score",
                    "stability/lora_ready"
                ],
                "layout": {"x": 6, "y": 8, "w": 6, "h": 6}
            },
            {
                "title": "Compute Savings",
                "type": "line",
                "metrics": ["null/compute_savings_pct"],
                "layout": {"x": 0, "y": 14, "w": 6, "h": 6}
            },
            {
                "title": "Expert Health",
                "type": "line",
                "metrics": [
                    "health/dead_expert_count",
                    "health/overloaded_expert_count"
                ],
                "layout": {"x": 6, "y": 14, "w": 6, "h": 6}
            },
        ],
        "alerts": [
            {
                "name": "Entropy Collapse",
                "condition": "health/entropy < 0.5",
                "severity": "critical"
            },
            {
                "name": "Signal Leakage High",
                "condition": "null/signal_to_null_rate > 15",
                "severity": "warning"
            },
            {
                "name": "Dead Experts",
                "condition": "health/dead_expert_count > 0",
                "severity": "warning"
            }
        ]
    }


# Example usage
if __name__ == "__main__":
    # Demo without actual W&B connection
    print("W&B Dashboard Configuration:")
    print(json.dumps(create_wandb_dashboard_config(), indent=2))
    
    print("\n" + "="*60)
    print("USAGE EXAMPLE")
    print("="*60)
    print("""
    from moe_dashboard.wandb_dashboard import WandBDashboard, WandBConfig
    
    # Initialize
    config = WandBConfig(
        project="moe-training",
        run_name="70b-moe-exp1",
        entity="your-team",
    )
    dashboard = WandBDashboard(config)
    dashboard.init_run(model_config={'model': '70b_moe', 'experts': 64})
    
    # In training loop
    for step in range(num_steps):
        # ... training code ...
        
        # Log routing metrics
        metrics = diagnostics.get_dashboard_metrics()
        dashboard.log_routing_metrics(metrics, step)
        
        # Log training metrics
        dashboard.log_training_metrics(
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            throughput=tokens_per_sec,
            step=step
        )
    
    dashboard.finish()
    """)
