#!/usr/bin/env python3
"""
Team 7 Dashboard Configuration
==============================
Dashboard configuration and data streaming for MoE routing diagnostics.

This module provides:
1. Dashboard panel definitions for Grafana/custom UI
2. Metric aggregation and formatting
3. Real-time telemetry streaming
4. Alert definitions and thresholds
5. Historical trend analysis

Team 7 Objectives Supported:
- Null expert fire rate on low-information tokens
- Token ID → expert family mapping
- Curriculum bucket (B0-B5) routing tracking
- Expert growth and LoRA-readiness milestones
- Compute savings validation

Usage:
    from dashboards.team7_dashboard import Team7Dashboard
    
    dashboard = Team7Dashboard(diagnostics)
    metrics = dashboard.get_live_metrics()
    panels = dashboard.get_panel_config()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import time


class PanelType(Enum):
    """Dashboard panel types."""
    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    HEATMAP = "heatmap"
    TABLE = "table"
    ALERT_LIST = "alert_list"
    STATUS = "status"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a dashboard metric."""
    
    name: str
    display_name: str
    description: str
    unit: str = ""
    
    # Thresholds for coloring
    green_min: Optional[float] = None
    green_max: Optional[float] = None
    yellow_min: Optional[float] = None
    yellow_max: Optional[float] = None
    red_min: Optional[float] = None
    red_max: Optional[float] = None
    
    # Aggregation
    aggregation: str = "last"  # last, avg, max, min, sum


@dataclass
class PanelConfig:
    """Configuration for a dashboard panel."""
    
    id: str
    title: str
    panel_type: PanelType
    metrics: List[str]
    
    # Layout (grid-based)
    row: int = 0
    col: int = 0
    width: int = 4
    height: int = 3
    
    # Options
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert rule definition."""
    
    id: str
    name: str
    description: str
    severity: AlertSeverity
    
    # Condition
    metric: str
    condition: str  # "above", "below", "outside_range"
    threshold: float
    threshold_max: Optional[float] = None
    
    # Timing
    for_duration: int = 60  # seconds
    
    # Actions
    recommendation: str = ""


class Team7Dashboard:
    """
    Team 7 Dashboard for MoE Routing Diagnostics.
    
    Provides dashboard configuration and live metrics for:
    1. Null Expert Analysis
       - Junk token → null routing rate
       - Signal token → null leakage
       - Compute savings achieved
    
    2. Token → Expert Mapping
       - Token family affinities
       - Router head specialization
       - Curriculum bucket distribution
    
    3. Health & Stability
       - Routing entropy
       - Load balance (Gini)
       - Dead/overloaded experts
       - LoRA-readiness status
    
    4. Growth Triggers
       - Expert expansion recommendations
       - Capacity utilization
    """
    
    def __init__(self, diagnostics=None):
        """
        Initialize dashboard.
        
        Args:
            diagnostics: RoutingDiagnostics instance (optional)
        """
        self.diagnostics = diagnostics
        self._metric_history = {}
        self._setup_metrics()
        self._setup_panels()
        self._setup_alerts()
    
    def _setup_metrics(self):
        """Define all dashboard metrics."""
        self.metrics = {
            # ==================== Null Expert Metrics ====================
            'null_junk_rate': MetricDefinition(
                name='null_junk_rate',
                display_name='Junk → Null Rate',
                description='Percentage of junk tokens routed to null expert',
                unit='%',
                green_min=60.0, green_max=80.0,
                yellow_min=50.0, yellow_max=60.0,
                red_min=0.0, red_max=50.0,
            ),
            'null_boilerplate_rate': MetricDefinition(
                name='null_boilerplate_rate',
                display_name='Boilerplate → Null Rate',
                description='Percentage of boilerplate tokens routed to null',
                unit='%',
                green_min=40.0, green_max=70.0,
            ),
            'null_signal_rate': MetricDefinition(
                name='null_signal_rate',
                display_name='Signal → Null Rate',
                description='Percentage of signal tokens leaked to null (should be low)',
                unit='%',
                green_min=0.0, green_max=10.0,
                yellow_min=10.0, yellow_max=15.0,
                red_min=15.0, red_max=100.0,
            ),
            'compute_savings': MetricDefinition(
                name='compute_savings',
                display_name='Compute Savings',
                description='FLOPs saved by null expert routing',
                unit='%',
                green_min=10.0, green_max=100.0,
            ),
            
            # ==================== Routing Health Metrics ====================
            'routing_entropy': MetricDefinition(
                name='routing_entropy',
                display_name='Routing Entropy',
                description='Normalized entropy of expert selection (1.0 = uniform)',
                unit='',
                green_min=0.70, green_max=1.0,
                yellow_min=0.50, yellow_max=0.70,
                red_min=0.0, red_max=0.50,
            ),
            'gini_coefficient': MetricDefinition(
                name='gini_coefficient',
                display_name='Load Balance (Gini)',
                description='Gini coefficient of expert utilization (0 = perfect balance)',
                unit='',
                green_min=0.0, green_max=0.30,
                yellow_min=0.30, yellow_max=0.50,
                red_min=0.50, red_max=1.0,
            ),
            'dead_expert_count': MetricDefinition(
                name='dead_expert_count',
                display_name='Dead Experts',
                description='Number of experts with <1% utilization',
                unit='',
                green_min=0.0, green_max=0.0,
                yellow_min=1.0, yellow_max=2.0,
                red_min=3.0, red_max=100.0,
            ),
            'overloaded_expert_count': MetricDefinition(
                name='overloaded_expert_count',
                display_name='Overloaded Experts',
                description='Number of experts with >3x expected utilization',
                unit='',
                green_min=0.0, green_max=0.0,
                yellow_min=1.0, yellow_max=2.0,
                red_min=3.0, red_max=100.0,
            ),
            
            # ==================== Stability Metrics ====================
            'stability_score': MetricDefinition(
                name='stability_score',
                display_name='Stability Score',
                description='Routing stability over time (1.0 = fully stable)',
                unit='',
                green_min=0.80, green_max=1.0,
                yellow_min=0.60, yellow_max=0.80,
                red_min=0.0, red_max=0.60,
            ),
            'lora_ready': MetricDefinition(
                name='lora_ready',
                display_name='LoRA Ready',
                description='Whether MoE block is stable enough for LoRA',
                unit='',
            ),
            'growth_ready': MetricDefinition(
                name='growth_ready',
                display_name='Growth Ready',
                description='Whether conditions are met for expert expansion',
                unit='',
            ),
            
            # ==================== Throughput Metrics ====================
            'tokens_per_second': MetricDefinition(
                name='tokens_per_second',
                display_name='Throughput',
                description='Training tokens processed per second',
                unit='tok/s',
            ),
            'expert_latency_ms': MetricDefinition(
                name='expert_latency_ms',
                display_name='Expert Latency',
                description='Average expert computation time',
                unit='ms',
            ),
        }
    
    def _setup_panels(self):
        """Define dashboard panel layout."""
        self.panels = [
            # ==================== Row 1: Key Metrics ====================
            PanelConfig(
                id='null_junk_gauge',
                title='Junk → Null Rate',
                panel_type=PanelType.GAUGE,
                metrics=['null_junk_rate'],
                row=0, col=0, width=3, height=3,
                options={'min': 0, 'max': 100, 'target': 70}
            ),
            PanelConfig(
                id='null_signal_gauge',
                title='Signal Leakage to Null',
                panel_type=PanelType.GAUGE,
                metrics=['null_signal_rate'],
                row=0, col=3, width=3, height=3,
                options={'min': 0, 'max': 30, 'target': 5, 'inverse': True}
            ),
            PanelConfig(
                id='entropy_gauge',
                title='Routing Entropy',
                panel_type=PanelType.GAUGE,
                metrics=['routing_entropy'],
                row=0, col=6, width=3, height=3,
                options={'min': 0, 'max': 1, 'target': 0.85}
            ),
            PanelConfig(
                id='status_panel',
                title='System Status',
                panel_type=PanelType.STATUS,
                metrics=['lora_ready', 'growth_ready'],
                row=0, col=9, width=3, height=3,
            ),
            
            # ==================== Row 2: Trends ====================
            PanelConfig(
                id='null_rates_trend',
                title='Null Routing Rates Over Time',
                panel_type=PanelType.LINE_CHART,
                metrics=['null_junk_rate', 'null_boilerplate_rate', 'null_signal_rate'],
                row=1, col=0, width=6, height=4,
                options={'legend': True, 'y_min': 0, 'y_max': 100}
            ),
            PanelConfig(
                id='health_trend',
                title='Routing Health Over Time',
                panel_type=PanelType.LINE_CHART,
                metrics=['routing_entropy', 'gini_coefficient'],
                row=1, col=6, width=6, height=4,
                options={'legend': True, 'y_min': 0, 'y_max': 1}
            ),
            
            # ==================== Row 3: Expert Analysis ====================
            PanelConfig(
                id='expert_utilization',
                title='Expert Utilization Distribution',
                panel_type=PanelType.BAR_CHART,
                metrics=['expert_utilization'],
                row=2, col=0, width=8, height=4,
                options={'x_label': 'Expert ID', 'y_label': 'Utilization %'}
            ),
            PanelConfig(
                id='expert_health',
                title='Expert Health Status',
                panel_type=PanelType.TABLE,
                metrics=['dead_expert_count', 'overloaded_expert_count'],
                row=2, col=8, width=4, height=4,
            ),
            
            # ==================== Row 4: Curriculum Buckets ====================
            PanelConfig(
                id='bucket_routing',
                title='Routing by Curriculum Bucket (B0-B5)',
                panel_type=PanelType.HEATMAP,
                metrics=['bucket_expert_distribution'],
                row=3, col=0, width=8, height=4,
                options={
                    'x_label': 'Expert ID',
                    'y_label': 'Bucket',
                    'y_categories': ['B0 Trivial', 'B1 Basic', 'B2 Intermediate', 
                                     'B3 Advanced', 'B4 Expert', 'B5 Frontier']
                }
            ),
            PanelConfig(
                id='compute_savings',
                title='Compute Savings',
                panel_type=PanelType.GAUGE,
                metrics=['compute_savings'],
                row=3, col=8, width=4, height=4,
                options={'min': 0, 'max': 30, 'suffix': '% FLOPs saved'}
            ),
            
            # ==================== Row 5: Alerts ====================
            PanelConfig(
                id='alerts',
                title='Active Alerts',
                panel_type=PanelType.ALERT_LIST,
                metrics=['alerts'],
                row=4, col=0, width=12, height=3,
            ),
        ]
    
    def _setup_alerts(self):
        """Define alert rules."""
        self.alert_rules = [
            AlertRule(
                id='null_junk_low',
                name='Low Null Routing for Junk',
                description='Junk tokens not routing to null as expected',
                severity=AlertSeverity.WARNING,
                metric='null_junk_rate',
                condition='below',
                threshold=50.0,
                for_duration=300,
                recommendation='Increase null expert bias or review token classification'
            ),
            AlertRule(
                id='null_signal_high',
                name='High Signal Leakage to Null',
                description='Too many signal tokens routing to null expert',
                severity=AlertSeverity.WARNING,
                metric='null_signal_rate',
                condition='above',
                threshold=15.0,
                for_duration=300,
                recommendation='Decrease null expert bias or check router learning'
            ),
            AlertRule(
                id='entropy_collapse',
                name='Router Entropy Collapse',
                description='Routing entropy critically low - possible expert collapse',
                severity=AlertSeverity.CRITICAL,
                metric='routing_entropy',
                condition='below',
                threshold=0.50,
                for_duration=60,
                recommendation='URGENT: Check for expert collapse, consider router reset'
            ),
            AlertRule(
                id='load_imbalance',
                name='Expert Load Imbalance',
                description='Expert utilization is highly uneven',
                severity=AlertSeverity.WARNING,
                metric='gini_coefficient',
                condition='above',
                threshold=0.50,
                for_duration=300,
                recommendation='Review load balancing bias adjustment'
            ),
            AlertRule(
                id='dead_experts',
                name='Dead Experts Detected',
                description='One or more experts have near-zero utilization',
                severity=AlertSeverity.WARNING,
                metric='dead_expert_count',
                condition='above',
                threshold=0,
                for_duration=600,
                recommendation='Boost bias for dead experts or reinitialize'
            ),
        ]
    
    def get_live_metrics(self) -> Dict:
        """
        Get current metrics for dashboard display.
        
        Returns:
            Dictionary of metric values formatted for display
        """
        if self.diagnostics:
            dashboard_metrics = self.diagnostics.get_dashboard_metrics()
        else:
            # Return placeholder data
            dashboard_metrics = self._get_placeholder_metrics()
        
        return self._format_for_display(dashboard_metrics)
    
    def _get_placeholder_metrics(self) -> Dict:
        """Get placeholder metrics when no diagnostics available."""
        return {
            'null_expert': {
                'junk_to_null_rate': '68.5%',
                'boilerplate_to_null_rate': '52.3%',
                'signal_to_null_rate': '6.2%',
                'compute_savings_pct': '14.2%',
            },
            'routing_health': {
                'entropy': '0.87',
                'gini_coefficient': '0.12',
                'dead_experts': [],
                'overloaded_experts': [],
            },
            'stability': {
                'is_stable': True,
                'stability_score': '0.92',
                'lora_ready': True,
            },
            'health_gates': {
                'null_junk_min': True,
                'null_junk_max': True,
                'null_signal_max': True,
                'entropy_min': True,
                'gini_max': True,
                'no_dead_experts': True,
                'no_overloaded_experts': True,
            },
            'alerts': [],
            'growth_trigger': {
                'recommend_growth': False,
                'confidence': 0.75,
            }
        }
    
    def _format_for_display(self, raw_metrics: Dict) -> Dict:
        """Format raw metrics for dashboard display."""
        null_exp = raw_metrics.get('null_expert', {})
        health = raw_metrics.get('routing_health', {})
        stability = raw_metrics.get('stability', {})
        
        # Parse percentage strings
        def parse_pct(s):
            if isinstance(s, str):
                return float(s.replace('%', ''))
            return float(s) if s else 0.0
        
        return {
            'metrics': {
                'null_junk_rate': {
                    'value': parse_pct(null_exp.get('junk_to_null_rate', '0')),
                    'display': null_exp.get('junk_to_null_rate', 'N/A'),
                    'status': self._get_status('null_junk_rate', parse_pct(null_exp.get('junk_to_null_rate', '0'))),
                },
                'null_signal_rate': {
                    'value': parse_pct(null_exp.get('signal_to_null_rate', '0')),
                    'display': null_exp.get('signal_to_null_rate', 'N/A'),
                    'status': self._get_status('null_signal_rate', parse_pct(null_exp.get('signal_to_null_rate', '0'))),
                },
                'routing_entropy': {
                    'value': float(health.get('entropy', 0)),
                    'display': health.get('entropy', 'N/A'),
                    'status': self._get_status('routing_entropy', float(health.get('entropy', 0))),
                },
                'gini_coefficient': {
                    'value': float(health.get('gini_coefficient', 0)),
                    'display': health.get('gini_coefficient', 'N/A'),
                    'status': self._get_status('gini_coefficient', float(health.get('gini_coefficient', 1))),
                },
                'compute_savings': {
                    'value': parse_pct(null_exp.get('compute_savings_pct', '0')),
                    'display': null_exp.get('compute_savings_pct', 'N/A'),
                },
                'stability_score': {
                    'value': float(stability.get('stability_score', 0)),
                    'display': stability.get('stability_score', 'N/A'),
                },
                'lora_ready': {
                    'value': stability.get('lora_ready', False),
                    'display': '✓ Ready' if stability.get('lora_ready') else '✗ Not Ready',
                },
                'growth_ready': {
                    'value': raw_metrics.get('growth_trigger', {}).get('recommend_growth', False),
                    'display': '✓ Ready' if raw_metrics.get('growth_trigger', {}).get('recommend_growth') else '✗ Not Ready',
                },
            },
            'health_gates': raw_metrics.get('health_gates', {}),
            'all_healthy': raw_metrics.get('all_gates_pass', False),
            'alerts': raw_metrics.get('alerts', []),
            'expert_health': {
                'dead': health.get('dead_experts', []),
                'overloaded': health.get('overloaded_experts', []),
            },
        }
    
    def _get_status(self, metric_name: str, value: float) -> str:
        """Get status color for a metric value."""
        metric = self.metrics.get(metric_name)
        if not metric:
            return 'unknown'
        
        if metric.green_min is not None and metric.green_max is not None:
            if metric.green_min <= value <= metric.green_max:
                return 'green'
        
        if metric.yellow_min is not None and metric.yellow_max is not None:
            if metric.yellow_min <= value <= metric.yellow_max:
                return 'yellow'
        
        if metric.red_min is not None and metric.red_max is not None:
            if metric.red_min <= value <= metric.red_max:
                return 'red'
        
        return 'unknown'
    
    def get_panel_config(self) -> List[Dict]:
        """
        Get panel configurations for dashboard rendering.
        
        Returns:
            List of panel configurations as dictionaries
        """
        return [
            {
                'id': p.id,
                'title': p.title,
                'type': p.panel_type.value,
                'metrics': p.metrics,
                'layout': {
                    'row': p.row,
                    'col': p.col,
                    'width': p.width,
                    'height': p.height,
                },
                'options': p.options,
            }
            for p in self.panels
        ]
    
    def get_alert_rules(self) -> List[Dict]:
        """Get alert rule definitions."""
        return [
            {
                'id': a.id,
                'name': a.name,
                'description': a.description,
                'severity': a.severity.value,
                'metric': a.metric,
                'condition': a.condition,
                'threshold': a.threshold,
                'for_duration': a.for_duration,
                'recommendation': a.recommendation,
            }
            for a in self.alert_rules
        ]
    
    def export_dashboard_config(self, filepath: str):
        """Export complete dashboard configuration to JSON."""
        config = {
            'title': 'Team 7 - MoE Routing Diagnostics',
            'version': '1.0.0',
            'refresh_interval': 5,  # seconds
            
            'metrics': {
                name: {
                    'display_name': m.display_name,
                    'description': m.description,
                    'unit': m.unit,
                    'thresholds': {
                        'green': [m.green_min, m.green_max],
                        'yellow': [m.yellow_min, m.yellow_max],
                        'red': [m.red_min, m.red_max],
                    }
                }
                for name, m in self.metrics.items()
            },
            
            'panels': self.get_panel_config(),
            'alerts': self.get_alert_rules(),
            
            'objectives': {
                'null_expert_fire_rate': {
                    'target': '60-80% for junk tokens',
                    'description': 'Null experts should heavily fire on low-information tokens',
                },
                'token_expert_mapping': {
                    'description': 'Track which token ID families route to which experts',
                },
                'curriculum_tracking': {
                    'buckets': ['B0', 'B1', 'B2', 'B3', 'B4', 'B5'],
                    'description': 'Monitor routing specialization by curriculum difficulty',
                },
                'stability_milestones': {
                    'description': 'Determine LoRA-readiness and expert growth triggers',
                },
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Dashboard config exported to: {filepath}")
    
    def print_summary(self):
        """Print dashboard summary."""
        metrics = self.get_live_metrics()
        
        print("=" * 60)
        print("TEAM 7 - MoE ROUTING DASHBOARD")
        print("=" * 60)
        
        print("\n🎯 Null Expert Metrics:")
        print(f"  Junk → Null: {metrics['metrics']['null_junk_rate']['display']} [{metrics['metrics']['null_junk_rate']['status']}]")
        print(f"  Signal → Null: {metrics['metrics']['null_signal_rate']['display']} [{metrics['metrics']['null_signal_rate']['status']}]")
        print(f"  Compute Savings: {metrics['metrics']['compute_savings']['display']}")
        
        print("\n📊 Routing Health:")
        print(f"  Entropy: {metrics['metrics']['routing_entropy']['display']} [{metrics['metrics']['routing_entropy']['status']}]")
        print(f"  Gini (Balance): {metrics['metrics']['gini_coefficient']['display']} [{metrics['metrics']['gini_coefficient']['status']}]")
        
        print("\n✅ Status:")
        print(f"  LoRA Ready: {metrics['metrics']['lora_ready']['display']}")
        print(f"  Growth Ready: {metrics['metrics']['growth_ready']['display']}")
        print(f"  All Gates Pass: {'✓' if metrics['all_healthy'] else '✗'}")
        
        if metrics['expert_health']['dead']:
            print(f"\n⚠️ Dead Experts: {metrics['expert_health']['dead']}")
        if metrics['expert_health']['overloaded']:
            print(f"⚠️ Overloaded Experts: {metrics['expert_health']['overloaded']}")
        
        if metrics['alerts']:
            print("\n🚨 Active Alerts:")
            for alert in metrics['alerts']:
                print(f"  [{alert['severity']}] {alert['message']}")
        
        print("=" * 60)


if __name__ == "__main__":
    # Demo
    dashboard = Team7Dashboard()
    dashboard.print_summary()
    dashboard.export_dashboard_config('/tmp/team7_dashboard_config.json')
