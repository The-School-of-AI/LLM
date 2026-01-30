#!/usr/bin/env python3
"""
🎯 Team 7 MoE Routing Dashboard
================================
Production-ready real-time monitoring dashboard for MoE architecture.

Features:
- Real-time metric streaming
- Interactive Plotly charts
- Health gate monitoring with alerts
- Expert utilization heatmaps
- Curriculum bucket analysis (B0-B5)
- LoRA-readiness and growth triggers
- Export capabilities

Quick Start:
    pip install streamlit plotly pandas numpy redis
    streamlit run app.py --server.port 8501

Deploy to Cloud:
    - Streamlit Cloud: https://streamlit.io/cloud (free)
    - Docker: See Dockerfile
    - Any VM: streamlit run app.py --server.address 0.0.0.0
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import time
from dataclasses import dataclass
import math
import copy

try:
    import redis as redis_lib
except ModuleNotFoundError:
    redis_lib = None

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Team 7 - MoE Routing Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-org/moe-dashboard',
        'Report a bug': 'https://github.com/your-org/moe-dashboard/issues',
        'About': '# Team 7 MoE Routing Dashboard\nReal-time monitoring for null expert routing.'
    }
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    /* Main container */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Status badges */
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-critical {
        background-color: #f8d7da;
        color: #721c24;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    
    /* Headers */
    .dashboard-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    /* Alert boxes */
    .alert-critical {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .alert-warning {
        background-color: #fff8e1;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    
    /* Gauge styling */
    .gauge-container {
        text-align: center;
        padding: 10px;
    }
    
    /* Table styling */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Sources
# =============================================================================

@dataclass
class ThresholdConfig:
    """Configurable thresholds for health gates."""
    null_junk_min: float = 60.0
    null_junk_max: float = 80.0
    null_signal_max: float = 10.0
    entropy_min: float = 0.70
    gini_max: float = 0.50
    dead_expert_threshold: float = 1.0
    overload_threshold: float = 300.0


class MetricsSimulator:
    """
    Simulates realistic MoE routing metrics.
    
    In production, replace with:
    - Redis subscription
    - REST API polling
    - WebSocket connection
    - File-based metrics
    """
    
    def __init__(self):
        self.step = 0
        self.history = []
        self.thresholds = ThresholdConfig()
        
        # Simulate training progression
        self.training_progress = 0.0
        
    def get_metrics(self) -> Dict:
        """Generate realistic metrics."""
        self.step += 1
        self.training_progress = min(1.0, self.training_progress + 0.001)
        
        # Simulate improvement over training
        improvement = self.training_progress * 0.3
        
        # Base values with realistic patterns
        base_junk_null = 65 + improvement * 15 + math.sin(self.step / 30) * 3
        base_signal_null = 8 - improvement * 3 + math.sin(self.step / 50) * 1
        base_entropy = 0.75 + improvement * 0.15 + math.sin(self.step / 100) * 0.03
        base_gini = 0.25 - improvement * 0.15 + math.sin(self.step / 80) * 0.02
        
        # Add some noise
        noise = lambda: np.random.normal(0, 1)
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'step': self.step,
            
            # Null Expert Metrics
            'null_expert': {
                'junk_to_null_rate': max(0, min(100, base_junk_null + noise())),
                'boilerplate_to_null_rate': max(0, min(100, 50 + improvement * 10 + noise() * 2)),
                'signal_to_null_rate': max(0, min(100, base_signal_null + noise() * 0.5)),
                'compute_savings_pct': max(0, 12 + improvement * 5 + noise() * 0.5),
            },
            
            # Routing Health
            'routing_health': {
                'entropy': max(0, min(1, base_entropy + noise() * 0.01)),
                'gini_coefficient': max(0, min(1, base_gini + noise() * 0.01)),
                'dead_experts': self._get_dead_experts(),
                'overloaded_experts': self._get_overloaded_experts(),
            },
            
            # Stability
            'stability': {
                'is_stable': np.random.random() > 0.1,
                'stability_score': max(0, min(1, 0.7 + improvement * 0.25 + noise() * 0.02)),
                'lora_ready': self.training_progress > 0.5 and np.random.random() > 0.2,
            },
            
            # Training metrics
            'training': {
                'loss': max(0.5, 4.0 - self.training_progress * 3 + noise() * 0.1),
                'throughput': max(0, 45000 + noise() * 2000),
                'learning_rate': 1e-4 * (1 - self.training_progress * 0.5),
            },
            
            # Growth trigger
            'growth_trigger': {
                'recommend_growth': self.training_progress > 0.7 and np.random.random() > 0.6,
                'confidence': min(1.0, 0.5 + improvement * 0.5 + noise() * 0.05),
            },
        }
        
        # Calculate health gates
        metrics['health_gates'] = self._check_health_gates(metrics)
        metrics['all_gates_pass'] = all(metrics['health_gates'].values())
        
        # Store in history
        self.history.append(metrics)
        if len(self.history) > 500:
            self.history.pop(0)
        
        return metrics
    
    def _get_dead_experts(self) -> List[int]:
        """Simulate dead expert detection."""
        if np.random.random() > 0.9:
            return [np.random.randint(0, 64)]
        return []
    
    def _get_overloaded_experts(self) -> List[int]:
        """Simulate overloaded expert detection."""
        if np.random.random() > 0.95:
            return [np.random.randint(0, 64)]
        return []
    
    def _check_health_gates(self, metrics: Dict) -> Dict[str, bool]:
        """Check all health gates."""
        null = metrics['null_expert']
        health = metrics['routing_health']
        
        return {
            'null_junk_min': null['junk_to_null_rate'] >= self.thresholds.null_junk_min,
            'null_junk_max': null['junk_to_null_rate'] <= self.thresholds.null_junk_max,
            'null_signal_max': null['signal_to_null_rate'] <= self.thresholds.null_signal_max,
            'entropy_min': health['entropy'] >= self.thresholds.entropy_min,
            'gini_max': health['gini_coefficient'] <= self.thresholds.gini_max,
            'no_dead_experts': len(health['dead_experts']) == 0,
            'no_overloaded_experts': len(health['overloaded_experts']) == 0,
        }

    def get_expert_utilization(self) -> Dict[int, float]:
        """Get per-expert utilization."""
        expected = 100 / 64
        utilization = {}
        for i in range(64):
            # Create realistic distribution with some variance
            util = expected * np.random.lognormal(0, 0.3)
            utilization[i] = util
        return utilization

    def get_curriculum_data(self) -> pd.DataFrame:
        """Get curriculum bucket → expert routing data."""
        buckets = ['B0_Trivial', 'B1_Basic', 'B2_Intermediate', 'B3_Advanced', 'B4_Expert', 'B5_Frontier']
        data = []

        for bucket_idx, bucket in enumerate(buckets):
            for expert_id in range(16):  # Show first 16 experts
                # Create realistic patterns: lower buckets → more null routing
                if bucket_idx <= 1:
                    # B0, B1: Heavy null routing
                    if expert_id >= 14:  # Null experts
                        pct = 30 + np.random.uniform(-5, 5)
                    else:
                        pct = np.random.uniform(0, 5)
                else:
                    # Higher buckets: more distributed
                    if expert_id >= 14:
                        pct = 5 - bucket_idx + np.random.uniform(-2, 2)
                    else:
                        pct = (100 - 10) / 14 * np.random.uniform(0.5, 1.5)

                data.append({
                    'Bucket': bucket,
                    'Expert': f'E{expert_id}',
                    'Routing %': max(0, pct)
                })

        return pd.DataFrame(data)

    def get_history_df(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        if not self.history:
            return pd.DataFrame()

        records = []
        for h in self.history:
            records.append({
                'step': h['step'],
                'timestamp': h['timestamp'],
                'Junk → Null (%)': h['null_expert']['junk_to_null_rate'],
                'Signal → Null (%)': h['null_expert']['signal_to_null_rate'],
                'Boilerplate → Null (%)': h['null_expert']['boilerplate_to_null_rate'],
                'Entropy': h['routing_health']['entropy'],
                'Gini': h['routing_health']['gini_coefficient'],
                'Compute Savings (%)': h['null_expert']['compute_savings_pct'],
                'Loss': h['training']['loss'],
                'Throughput': h['training']['throughput'],
                'Stability Score': h['stability']['stability_score'],
            })

        return pd.DataFrame(records)


def compute_health_gates(metrics: Dict, thresholds: ThresholdConfig) -> Dict[str, bool]:
    """Compute health gates for normalized metrics."""
    null = metrics["null_expert"]
    health = metrics["routing_health"]
    return {
        'null_junk_min': null['junk_to_null_rate'] >= thresholds.null_junk_min,
        'null_junk_max': null['junk_to_null_rate'] <= thresholds.null_junk_max,
        'null_signal_max': null['signal_to_null_rate'] <= thresholds.null_signal_max,
        'entropy_min': health['entropy'] >= thresholds.entropy_min,
        'gini_max': health['gini_coefficient'] <= thresholds.gini_max,
        'no_dead_experts': len(health['dead_experts']) == 0,
        'no_overloaded_experts': len(health['overloaded_experts']) == 0,
    }


class RedisMetricsSource:
    """Pull latest metrics from Redis."""

    def __init__(self, redis_url: str, redis_key: str):
        self.redis_url = redis_url
        self.redis_key = redis_key
        self._client = None

    def available(self) -> bool:
        return redis_lib is not None

    def _ensure_client(self):
        if redis_lib is None:
            return None
        if self._client is None:
            self._client = redis_lib.from_url(self.redis_url)
        return self._client

    def get_raw_metrics(self) -> Optional[Dict]:
        client = self._ensure_client()
        if client is None:
            return None
        payload = client.get(self.redis_key)
        if payload is None:
            return None
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None


def _parse_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("%"):
            raw = raw[:-1]
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def normalize_metrics(raw: Dict) -> Dict:
    """Normalize metrics so dashboards can assume numeric values."""
    metrics = copy.deepcopy(raw)

    null_exp = metrics.get("null_expert", {}) or {}
    null_exp["junk_to_null_rate"] = _parse_float(null_exp.get("junk_to_null_rate")) or 0.0
    null_exp["boilerplate_to_null_rate"] = _parse_float(null_exp.get("boilerplate_to_null_rate")) or 0.0
    null_exp["signal_to_null_rate"] = _parse_float(null_exp.get("signal_to_null_rate")) or 0.0
    null_exp["compute_savings_pct"] = _parse_float(null_exp.get("compute_savings_pct")) or 0.0
    metrics["null_expert"] = null_exp

    routing = metrics.get("routing_health", {}) or {}
    routing["entropy"] = _parse_float(routing.get("entropy")) or 0.0
    routing["gini_coefficient"] = _parse_float(routing.get("gini_coefficient")) or 0.0
    routing.setdefault("dead_experts", [])
    routing.setdefault("overloaded_experts", [])
    metrics["routing_health"] = routing

    stability = metrics.get("stability", {}) or {}
    stability["stability_score"] = _parse_float(stability.get("stability_score")) or 0.0
    stability.setdefault("is_stable", False)
    stability.setdefault("lora_ready", False)
    metrics["stability"] = stability

    growth = metrics.get("growth_trigger", {}) or {}
    growth.setdefault("recommend_growth", False)
    growth.setdefault("confidence", 0.0)
    metrics["growth_trigger"] = growth

    training = metrics.get("training", {}) or {}
    training["loss"] = _parse_float(training.get("loss"))
    training["throughput"] = _parse_float(training.get("throughput"))
    training["learning_rate"] = _parse_float(training.get("learning_rate"))
    metrics["training"] = training

    metrics.setdefault("health_gates", {})
    metrics.setdefault("all_gates_pass", False)
    metrics.setdefault("alerts", [])
    return metrics


def update_history(history: List[Dict], metrics: Dict) -> None:
    """Append normalized metrics to history if new."""
    if not metrics:
        return
    step = metrics.get("step")
    if history and history[-1].get("step") == step:
        return
    history.append(metrics)
    if len(history) > 500:
        history.pop(0)


def history_to_df(history: List[Dict]) -> pd.DataFrame:
    """Convert metric history to a DataFrame for charts."""
    if not history:
        return pd.DataFrame()
    records = []
    for h in history:
        records.append({
            "step": h.get("step"),
            "timestamp": h.get("timestamp"),
            "Junk → Null (%)": h["null_expert"]["junk_to_null_rate"],
            "Signal → Null (%)": h["null_expert"]["signal_to_null_rate"],
            "Boilerplate → Null (%)": h["null_expert"]["boilerplate_to_null_rate"],
            "Entropy": h["routing_health"]["entropy"],
            "Gini": h["routing_health"]["gini_coefficient"],
            "Compute Savings (%)": h["null_expert"]["compute_savings_pct"],
            "Loss": h.get("training", {}).get("loss"),
            "Throughput": h.get("training", {}).get("throughput"),
            "Stability Score": h["stability"]["stability_score"],
        })
    return pd.DataFrame(records)


# Initialize simulator (cached across reruns)
@st.cache_resource
def get_simulator():
    return MetricsSimulator()


@st.cache_resource
def get_redis_source(redis_url: str, redis_key: str):
    return RedisMetricsSource(redis_url=redis_url, redis_key=redis_key)


# =============================================================================
# Dashboard Components
# =============================================================================

def render_header(metrics: Dict, source_label: str):
    """Render dashboard header."""
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    
    step_val = metrics.get("step", 0)
    with col1:
        st.title("🎯 Team 7 - MoE Routing Dashboard")
        st.caption(
            f"Step: {step_val:,} | Data Source: {source_label} | "
            f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
        )
    
    with col2:
        all_pass = metrics.get('all_gates_pass', False)
        if all_pass:
            st.success("✓ All Healthy")
        else:
            failed = sum(1 for v in metrics['health_gates'].values() if not v)
            st.error(f"✗ {failed} Gate(s) Failed")
    
    with col3:
        if metrics['stability']['lora_ready']:
            st.success("✓ LoRA Ready")
        else:
            st.warning("○ LoRA Pending")
    
    with col4:
        if metrics['growth_trigger']['recommend_growth']:
            st.success(f"✓ Growth Ready ({metrics['growth_trigger']['confidence']:.0%})")
        else:
            st.info("○ Growth Pending")


def render_key_gauges(metrics: Dict):
    """Render key metric gauges."""
    null_exp = metrics['null_expert']
    health = metrics['routing_health']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        value = null_exp['junk_to_null_rate']
        delta = "Target: 60-80%"
        color = "normal" if 60 <= value <= 80 else "inverse"
        st.metric("🗑️ Junk → Null", f"{value:.1f}%", delta, delta_color="off")
        
        # Mini gauge
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'visible': False},
                'bar': {'color': "#2196f3"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffcdd2"},
                    {'range': [50, 60], 'color': "#fff9c4"},
                    {'range': [60, 80], 'color': "#c8e6c9"},
                    {'range': [80, 100], 'color': "#fff9c4"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'value': value
                }
            }
        ))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        value = null_exp['signal_to_null_rate']
        color = "normal" if value < 10 else "inverse"
        st.metric("⚠️ Signal Leakage", f"{value:.1f}%", "Target: <10%", delta_color="off")
        
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                'axis': {'range': [0, 30], 'visible': False},
                'bar': {'color': "#f44336" if value > 10 else "#4caf50"},
                'steps': [
                    {'range': [0, 10], 'color': "#c8e6c9"},
                    {'range': [10, 15], 'color': "#fff9c4"},
                    {'range': [15, 30], 'color': "#ffcdd2"},
                ],
            }
        ))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        value = health['entropy']
        st.metric("📊 Routing Entropy", f"{value:.3f}", "Target: >0.70", delta_color="off")
        
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                'axis': {'range': [0, 1], 'visible': False},
                'bar': {'color': "#9c27b0"},
                'steps': [
                    {'range': [0, 0.5], 'color': "#ffcdd2"},
                    {'range': [0.5, 0.7], 'color': "#fff9c4"},
                    {'range': [0.7, 1], 'color': "#c8e6c9"},
                ],
            }
        ))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        value = health['gini_coefficient']
        st.metric("⚖️ Load Balance (Gini)", f"{value:.3f}", "Target: <0.50", delta_color="off")
        
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                'axis': {'range': [0, 1], 'visible': False},
                'bar': {'color': "#009688"},
                'steps': [
                    {'range': [0, 0.3], 'color': "#c8e6c9"},
                    {'range': [0.3, 0.5], 'color': "#fff9c4"},
                    {'range': [0.5, 1], 'color': "#ffcdd2"},
                ],
            }
        ))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    with col5:
        value = null_exp['compute_savings_pct']
        st.metric("💰 Compute Savings", f"{value:.1f}%", "FLOPs saved")
        
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=value,
            gauge={
                'axis': {'range': [0, 30], 'visible': False},
                'bar': {'color': "#ff9800"},
                'steps': [
                    {'range': [0, 10], 'color': "#fff9c4"},
                    {'range': [10, 20], 'color': "#c8e6c9"},
                    {'range': [20, 30], 'color': "#a5d6a7"},
                ],
            }
        ))
        fig.update_layout(height=120, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_trend_charts(df: pd.DataFrame):
    """Render trend line charts."""
    if df.empty:
        st.info("Collecting data...")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Null Routing Trends")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['Junk → Null (%)'],
            name='Junk → Null', line=dict(color='#2196f3', width=2),
            fill='tozeroy', fillcolor='rgba(33, 150, 243, 0.1)'
        ))
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['Signal → Null (%)'],
            name='Signal → Null', line=dict(color='#f44336', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['Boilerplate → Null (%)'],
            name='Boilerplate → Null', line=dict(color='#ff9800', width=2)
        ))
        
        # Target zones
        fig.add_hrect(y0=60, y1=80, fillcolor="green", opacity=0.1,
                      annotation_text="Junk Target", annotation_position="top left")
        fig.add_hline(y=10, line_dash="dash", line_color="red",
                      annotation_text="Signal Max")
        
        fig.update_layout(
            height=350,
            xaxis_title="Training Step",
            yaxis_title="Rate (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Routing Health Trends")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['Entropy'],
            name='Entropy', line=dict(color='#9c27b0', width=2),
            fill='tozeroy', fillcolor='rgba(156, 39, 176, 0.1)'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['Gini'],
            name='Gini', line=dict(color='#009688', width=2)
        ), secondary_y=True)
        
        # Thresholds
        fig.add_hline(y=0.70, line_dash="dash", line_color="purple",
                      annotation_text="Entropy Min", secondary_y=False)
        
        fig.update_layout(
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            hovermode="x unified"
        )
        fig.update_xaxes(title_text="Training Step")
        fig.update_yaxes(title_text="Entropy", secondary_y=False)
        fig.update_yaxes(title_text="Gini Coefficient", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)


def render_expert_heatmap(simulator: MetricsSimulator):
    """Render expert utilization heatmap."""
    st.subheader("🔥 Expert Utilization Heatmap")
    
    utilization = simulator.get_expert_utilization()
    
    # Reshape to 8x8 grid
    data = np.array([utilization[i] for i in range(64)]).reshape(8, 8)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[f'E{i}' for i in range(8)],
        y=[f'E{i*8}-{i*8+7}' for i in range(8)],
        colorscale=[
            [0, '#ffcdd2'],      # Under-utilized (red)
            [0.3, '#fff9c4'],    # Low (yellow)
            [0.5, '#c8e6c9'],    # Target (green)
            [0.7, '#fff9c4'],    # High (yellow)
            [1, '#ffcdd2']       # Over-utilized (red)
        ],
        hoverongaps=False,
        hovertemplate='Expert %{x}<br>Utilization: %{z:.2f}%<extra></extra>'
    ))
    
    expected = 100 / 64
    fig.update_layout(
        height=300,
        title=f"Expected utilization: {expected:.2f}% per expert",
        xaxis_title="Expert (column)",
        yaxis_title="Expert (row)",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_curriculum_routing(simulator: MetricsSimulator):
    """Render curriculum bucket routing analysis."""
    st.subheader("📚 Curriculum Bucket → Expert Routing")
    
    df = simulator.get_curriculum_data()
    
    fig = px.density_heatmap(
        df, x='Expert', y='Bucket', z='Routing %',
        color_continuous_scale='RdYlGn',
        title='Token routing distribution by curriculum difficulty'
    )
    
    fig.update_layout(
        height=350,
        xaxis_title="Expert ID",
        yaxis_title="Curriculum Bucket",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**B0-B1 (Trivial/Basic)**: Heavy null routing expected")
    with col2:
        st.info("**B2-B3 (Intermediate/Advanced)**: Balanced distribution")
    with col3:
        st.info("**B4-B5 (Expert/Frontier)**: Specialized experts")


def render_alerts(metrics: Dict):
    """Render active alerts."""
    st.subheader("🚨 Active Alerts")
    
    alerts = []
    
    null_exp = metrics['null_expert']
    health = metrics['routing_health']
    gates = metrics['health_gates']
    
    # Check conditions
    if not gates['null_junk_min']:
        alerts.append({
            'severity': 'warning',
            'title': 'Low Junk → Null Rate',
            'message': f"Current: {null_exp['junk_to_null_rate']:.1f}% (target: ≥60%)",
            'action': 'Consider increasing null expert bias'
        })
    
    if not gates['null_signal_max']:
        alerts.append({
            'severity': 'warning',
            'title': 'High Signal Leakage',
            'message': f"Current: {null_exp['signal_to_null_rate']:.1f}% (target: ≤10%)",
            'action': 'Review router training or decrease null bias'
        })
    
    if not gates['entropy_min']:
        alerts.append({
            'severity': 'critical',
            'title': 'Router Entropy Collapse',
            'message': f"Entropy: {health['entropy']:.3f} (target: ≥0.70)",
            'action': 'URGENT: Check for expert collapse, consider router reset'
        })
    
    if not gates['no_dead_experts']:
        alerts.append({
            'severity': 'warning',
            'title': 'Dead Experts Detected',
            'message': f"Experts {health['dead_experts']} have <1% utilization",
            'action': 'Boost bias for affected experts or reinitialize'
        })
    
    if not gates['no_overloaded_experts']:
        alerts.append({
            'severity': 'warning',
            'title': 'Overloaded Experts',
            'message': f"Experts {health['overloaded_experts']} have >3x expected load",
            'action': 'Check routing distribution, may need rebalancing'
        })
    
    if not alerts:
        st.success("✓ No active alerts - all systems healthy!")
    else:
        for alert in alerts:
            if alert['severity'] == 'critical':
                st.error(f"**🔴 {alert['title']}**\n\n{alert['message']}\n\n*Recommended Action: {alert['action']}*")
            else:
                st.warning(f"**🟡 {alert['title']}**\n\n{alert['message']}\n\n*Recommended Action: {alert['action']}*")


def render_health_gates(metrics: Dict):
    """Render health gate status table."""
    with st.expander("🔍 Health Gate Details", expanded=False):
        gates = metrics['health_gates']
        
        gate_data = []
        gate_descriptions = {
            'null_junk_min': ('Junk → Null ≥ 60%', 'Ensures junk tokens route to null'),
            'null_junk_max': ('Junk → Null ≤ 80%', 'Prevents over-routing to null'),
            'null_signal_max': ('Signal → Null ≤ 10%', 'Limits signal token leakage'),
            'entropy_min': ('Entropy ≥ 0.70', 'Ensures routing diversity'),
            'gini_max': ('Gini ≤ 0.50', 'Ensures load balance'),
            'no_dead_experts': ('No Dead Experts', 'All experts utilized >1%'),
            'no_overloaded_experts': ('No Overloaded Experts', 'No expert >3x expected'),
        }
        
        for gate_name, (description, explanation) in gate_descriptions.items():
            status = gates.get(gate_name, False)
            gate_data.append({
                'Gate': description,
                'Status': '✅ Pass' if status else '❌ Fail',
                'Description': explanation,
            })
        
        df = pd.DataFrame(gate_data)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_training_metrics(df: pd.DataFrame):
    """Render training metrics."""
    with st.expander("📉 Training Metrics", expanded=False):
        if df.empty:
            st.info("Collecting data...")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(df, x='step', y='Loss', title='Training Loss')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(df, x='step', y='Throughput', title='Throughput (tokens/sec)')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)


def render_sidebar_controls(simulator: MetricsSimulator, redis_available: bool) -> Dict:
    """Render sidebar controls and return settings."""
    st.sidebar.header("⚙️ Settings")

    data_source = st.sidebar.radio(
        "Data Source",
        ["Redis", "Simulator"],
        index=0 if redis_available else 1,
        help="Choose where the dashboard reads metrics from"
    )

    redis_url = st.sidebar.text_input(
        "Redis URL",
        value="redis://localhost:6379",
        disabled=data_source != "Redis",
    )
    redis_key = st.sidebar.text_input(
        "Redis Key",
        value="moe_metrics",
        disabled=data_source != "Redis",
    )
    fallback_to_sim = st.sidebar.checkbox(
        "Fallback to simulator if Redis is empty",
        value=True,
        disabled=data_source != "Redis",
    )

    # Refresh rate
    refresh_rate = st.sidebar.slider(
        "Refresh Rate (seconds)",
        min_value=1,
        max_value=30,
        value=2,
        help="How often to update the dashboard"
    )

    st.sidebar.markdown("---")

    # Model info
    st.sidebar.header("📊 Model Info")
    st.sidebar.info("""
    **Model**: 70B MoE-64
    - Routed Experts: 64
    - Shared Experts: 4
    - Null Experts: 2
    - Top-K: 4
    """)

    st.sidebar.markdown("---")

    # Threshold configuration
    st.sidebar.header("🎚️ Thresholds")
    if data_source == "Simulator":
        with st.sidebar.expander("Configure Thresholds"):
            simulator.thresholds.null_junk_min = st.number_input(
                "Min Junk→Null (%)", value=60.0, step=5.0
            )
            simulator.thresholds.null_junk_max = st.number_input(
                "Max Junk→Null (%)", value=80.0, step=5.0
            )
            simulator.thresholds.null_signal_max = st.number_input(
                "Max Signal→Null (%)", value=10.0, step=1.0
            )
            simulator.thresholds.entropy_min = st.number_input(
                "Min Entropy", value=0.70, step=0.05
            )
            simulator.thresholds.gini_max = st.number_input(
                "Max Gini", value=0.50, step=0.05
            )
    else:
        st.sidebar.caption("Thresholds apply only to simulator data.")

    return {
        "data_source": data_source,
        "redis_url": redis_url,
        "redis_key": redis_key,
        "fallback_to_sim": fallback_to_sim,
        "refresh_rate": refresh_rate,
    }


def render_sidebar_status(
    metrics: Optional[Dict],
    history_df: pd.DataFrame,
    data_source: str,
    redis_available: bool,
    source_label: str,
):
    """Render sidebar status and export controls."""
    st.sidebar.markdown("---")
    st.sidebar.header("📡 Data Status")

    if data_source == "Redis":
        if not redis_available:
            st.sidebar.error("Redis client not installed (pip install redis).")
        elif metrics is None:
            st.sidebar.warning("No Redis metrics found yet.")
        else:
            st.sidebar.success(f"Receiving metrics via {source_label}.")
    else:
        st.sidebar.info("Using simulator data.")

    st.sidebar.markdown("---")

    # Export
    st.sidebar.header("📤 Export")

    if metrics is not None and st.sidebar.button("📋 Export Current Metrics"):
        st.sidebar.download_button(
            "Download JSON",
            data=json.dumps(metrics, indent=2, default=str),
            file_name=f"moe_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    if st.sidebar.button("📊 Export History CSV") and not history_df.empty:
        st.sidebar.download_button(
            "Download CSV",
            data=history_df.to_csv(index=False),
            file_name=f"moe_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    """Main dashboard entry point."""
    # Initialize simulator
    simulator = get_simulator()
    redis_available = redis_lib is not None

    # Sidebar controls
    controls = render_sidebar_controls(simulator, redis_available)
    data_source = controls["data_source"]
    refresh_rate = controls["refresh_rate"]

    metrics = None
    history_df = pd.DataFrame()
    source_label = "Simulator"

    if data_source == "Redis":
        redis_source = get_redis_source(controls["redis_url"], controls["redis_key"])
        raw_metrics = redis_source.get_raw_metrics()
        if raw_metrics:
            metrics = normalize_metrics(raw_metrics)
            if not metrics.get("health_gates"):
                metrics["health_gates"] = compute_health_gates(metrics, simulator.thresholds)
                metrics["all_gates_pass"] = all(metrics["health_gates"].values())
            if "redis_history" not in st.session_state:
                st.session_state.redis_history = []
            update_history(st.session_state.redis_history, metrics)
            history_df = history_to_df(st.session_state.redis_history)
            source_label = "Redis"
        elif controls["fallback_to_sim"]:
            metrics = simulator.get_metrics()
            history_df = simulator.get_history_df()
            source_label = "Simulator (fallback)"
        else:
            metrics = None
            history_df = pd.DataFrame()
            source_label = "Redis"
    else:
        metrics = simulator.get_metrics()
        history_df = simulator.get_history_df()
        source_label = "Simulator"

    render_sidebar_status(metrics, history_df, data_source, redis_available, source_label)

    if metrics is None:
        st.warning("Waiting for metrics. Ensure Redis is running and the training script is logging.")
        _schedule_refresh(refresh_rate)
        return

    # Header
    render_header(metrics, source_label)
    
    st.markdown("---")
    
    # Key gauges
    render_key_gauges(metrics)
    
    st.markdown("---")
    
    # Trend charts
    render_trend_charts(history_df)
    
    st.markdown("---")
    
    # Two columns for heatmaps
    col1, col2 = st.columns(2)
    
    with col1:
        if source_label.startswith("Redis"):
            st.caption("Expert heatmap uses simulated data (no per-expert utilization in Redis payload).")
        render_expert_heatmap(simulator)
    
    with col2:
        if source_label.startswith("Redis"):
            st.caption("Curriculum routing uses simulated data (no curriculum data in Redis payload).")
        render_curriculum_routing(simulator)
    
    st.markdown("---")
    
    # Alerts
    render_alerts(metrics)
    
    # Health gates (expandable)
    render_health_gates(metrics)
    
    # Training metrics (expandable)
    render_training_metrics(history_df)
    
    # Auto-refresh
    _schedule_refresh(refresh_rate)


def _schedule_refresh(refresh_rate: int):
    interval_ms = int(refresh_rate * 1000)
    if hasattr(st, "autorefresh"):
        st.autorefresh(interval=interval_ms, key="dashboard_autorefresh")
    else:
        time.sleep(refresh_rate)
        st.rerun()


if __name__ == "__main__":
    main()
