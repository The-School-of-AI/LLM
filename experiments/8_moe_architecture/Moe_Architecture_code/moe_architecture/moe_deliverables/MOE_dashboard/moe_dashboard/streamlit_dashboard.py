#!/usr/bin/env python3
"""
Streamlit Dashboard for MoE Routing Monitoring
==============================================
Production-ready standalone dashboard with real-time updates.

Features:
- Real-time metric visualization
- Interactive charts (Plotly)
- Health gate status indicators
- Alert notifications
- Historical trend analysis
- Export capabilities

Setup:
    pip install streamlit plotly pandas

Run:
    streamlit run streamlit_dashboard.py --server.port 8501

Deploy to cloud:
    - Streamlit Cloud (free): https://streamlit.io/cloud
    - Docker container
    - Any cloud VM with port 8501
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional
import random

# Page configuration
st.set_page_config(
    page_title="Team 7 - MoE Routing Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .status-green { color: #00c853; font-weight: bold; }
    .status-yellow { color: #ffd600; font-weight: bold; }
    .status-red { color: #ff1744; font-weight: bold; }
    .big-number { font-size: 2.5em; font-weight: bold; }
    .alert-box {
        padding: 10px 15px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .alert-critical { background-color: #ffebee; border-left: 4px solid #f44336; }
    .alert-warning { background-color: #fff8e1; border-left: 4px solid #ffc107; }
    .alert-info { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Source (Replace with actual data connection)
# =============================================================================

class MetricsDataSource:
    """
    Data source for dashboard metrics.
    
    In production, replace this with:
    - Redis connection for real-time data
    - REST API call to training server
    - WebSocket connection
    - File-based polling
    """
    
    def __init__(self):
        self.history = []
        self.max_history = 1000
        
    def get_latest_metrics(self) -> Dict:
        """Get latest metrics (simulated for demo)."""
        # In production, fetch from your data source
        step = len(self.history)
        
        # Simulated metrics with realistic patterns
        base_junk_null = 68 + np.sin(step / 50) * 5
        base_entropy = 0.85 + np.sin(step / 100) * 0.05
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'null_expert': {
                'junk_to_null_rate': round(base_junk_null + random.uniform(-2, 2), 1),
                'boilerplate_to_null_rate': round(52 + random.uniform(-3, 3), 1),
                'signal_to_null_rate': round(6 + random.uniform(-1, 1), 1),
                'compute_savings_pct': round(14 + random.uniform(-1, 1), 1),
            },
            'routing_health': {
                'entropy': round(base_entropy + random.uniform(-0.02, 0.02), 3),
                'gini_coefficient': round(0.12 + random.uniform(-0.02, 0.02), 3),
                'dead_experts': [] if random.random() > 0.1 else [random.randint(0, 63)],
                'overloaded_experts': [] if random.random() > 0.05 else [random.randint(0, 63)],
            },
            'stability': {
                'is_stable': random.random() > 0.1,
                'stability_score': round(0.88 + random.uniform(-0.05, 0.05), 2),
                'lora_ready': random.random() > 0.2,
            },
            'health_gates': {
                'null_junk_min': True,
                'null_junk_max': True,
                'null_signal_max': True,
                'entropy_min': base_entropy > 0.70,
                'gini_max': True,
                'no_dead_experts': random.random() > 0.1,
                'no_overloaded_experts': random.random() > 0.05,
            },
            'growth_trigger': {
                'recommend_growth': random.random() > 0.8,
                'confidence': round(random.uniform(0.6, 0.95), 2),
            },
            'training': {
                'loss': round(3.5 - step * 0.001 + random.uniform(-0.1, 0.1), 4),
                'throughput': round(45000 + random.uniform(-2000, 2000), 0),
            }
        }
        
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return metrics
    
    def get_history(self, n: int = 100) -> List[Dict]:
        """Get historical metrics."""
        return self.history[-n:] if self.history else []
    
    def get_expert_utilization(self) -> Dict[int, float]:
        """Get per-expert utilization."""
        # Simulated utilization
        utilization = {}
        for i in range(64):
            utilization[i] = random.uniform(0.8, 1.2) * (1/64) * 100
        return utilization


# Initialize data source
@st.cache_resource
def get_data_source():
    return MetricsDataSource()


# =============================================================================
# Dashboard Components
# =============================================================================

def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title("🎯 Team 7 - MoE Routing Dashboard")
        st.caption("Real-time monitoring of null expert routing and MoE health")
    
    with col2:
        st.metric("Model", "70B MoE-64")
    
    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))


def render_key_metrics(metrics: Dict):
    """Render key metrics as gauges."""
    st.subheader("📊 Key Metrics")
    
    null_exp = metrics.get('null_expert', {})
    health = metrics.get('routing_health', {})
    stability = metrics.get('stability', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        junk_rate = float(null_exp.get('junk_to_null_rate', 0))
        status = "🟢" if 60 <= junk_rate <= 80 else "🟡" if 50 <= junk_rate < 60 else "🔴"
        st.metric(
            f"{status} Junk → Null",
            f"{junk_rate:.1f}%",
            delta=f"Target: 60-80%",
            delta_color="off"
        )
    
    with col2:
        signal_rate = float(null_exp.get('signal_to_null_rate', 0))
        status = "🟢" if signal_rate < 10 else "🟡" if signal_rate < 15 else "🔴"
        st.metric(
            f"{status} Signal Leakage",
            f"{signal_rate:.1f}%",
            delta=f"Target: <10%",
            delta_color="off"
        )
    
    with col3:
        entropy = float(health.get('entropy', 0))
        status = "🟢" if entropy > 0.70 else "🟡" if entropy > 0.50 else "🔴"
        st.metric(
            f"{status} Routing Entropy",
            f"{entropy:.3f}",
            delta=f"Target: >0.70",
            delta_color="off"
        )
    
    with col4:
        savings = float(null_exp.get('compute_savings_pct', 0))
        st.metric(
            "💰 Compute Savings",
            f"{savings:.1f}%",
            delta="FLOPs saved by null routing"
        )


def render_status_indicators(metrics: Dict):
    """Render status indicators."""
    st.subheader("✅ Status")
    
    stability = metrics.get('stability', {})
    growth = metrics.get('growth_trigger', {})
    gates = metrics.get('health_gates', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        lora_ready = stability.get('lora_ready', False)
        if lora_ready:
            st.success("✓ LoRA Ready")
        else:
            st.warning("✗ LoRA Not Ready")
    
    with col2:
        growth_ready = growth.get('recommend_growth', False)
        if growth_ready:
            st.success(f"✓ Growth Ready ({growth.get('confidence', 0):.0%})")
        else:
            st.info("○ Growth Not Ready")
    
    with col3:
        all_gates = all(gates.values())
        if all_gates:
            st.success("✓ All Gates Pass")
        else:
            failed = [k for k, v in gates.items() if not v]
            st.error(f"✗ {len(failed)} Gate(s) Failed")
    
    with col4:
        is_stable = stability.get('is_stable', False)
        score = stability.get('stability_score', 0)
        if is_stable:
            st.success(f"✓ Stable ({score:.2f})")
        else:
            st.warning(f"○ Stabilizing ({score:.2f})")


def render_trend_charts(history: List[Dict]):
    """Render trend charts."""
    if not history:
        st.info("Waiting for data...")
        return
    
    st.subheader("📈 Trends")
    
    # Prepare data
    df = pd.DataFrame([
        {
            'step': h['step'],
            'Junk → Null': h['null_expert']['junk_to_null_rate'],
            'Signal → Null': h['null_expert']['signal_to_null_rate'],
            'Boilerplate → Null': h['null_expert']['boilerplate_to_null_rate'],
            'Entropy': h['routing_health']['entropy'],
            'Gini': h['routing_health']['gini_coefficient'],
            'Loss': h['training']['loss'],
            'Throughput': h['training']['throughput'],
        }
        for h in history
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Null routing rates
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['step'], y=df['Junk → Null'], name='Junk → Null', line=dict(color='#2196f3')))
        fig.add_trace(go.Scatter(x=df['step'], y=df['Signal → Null'], name='Signal → Null', line=dict(color='#f44336')))
        fig.add_trace(go.Scatter(x=df['step'], y=df['Boilerplate → Null'], name='Boilerplate → Null', line=dict(color='#ff9800')))
        
        # Add target zones
        fig.add_hrect(y0=60, y1=80, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Signal Max (10%)")
        
        fig.update_layout(
            title="Null Routing Rates",
            xaxis_title="Step",
            yaxis_title="Rate (%)",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Routing health
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['Entropy'], name='Entropy', line=dict(color='#9c27b0')),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df['step'], y=df['Gini'], name='Gini', line=dict(color='#009688')),
            secondary_y=True,
        )
        
        # Thresholds
        fig.add_hline(y=0.70, line_dash="dash", line_color="purple", annotation_text="Entropy Min")
        
        fig.update_layout(
            title="Routing Health",
            height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.update_yaxes(title_text="Entropy", secondary_y=False)
        fig.update_yaxes(title_text="Gini", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)


def render_expert_utilization(data_source: MetricsDataSource):
    """Render expert utilization chart."""
    st.subheader("🔧 Expert Utilization")
    
    utilization = data_source.get_expert_utilization()
    
    df = pd.DataFrame([
        {'Expert': f'E{k}', 'Utilization': v}
        for k, v in sorted(utilization.items())
    ])
    
    fig = px.bar(
        df, x='Expert', y='Utilization',
        color='Utilization',
        color_continuous_scale=['red', 'yellow', 'green'],
        title='Per-Expert Utilization (%)',
    )
    
    expected = 100 / 64
    fig.add_hline(y=expected, line_dash="dash", annotation_text=f"Expected ({expected:.2f}%)")
    
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_alerts(metrics: Dict):
    """Render active alerts."""
    st.subheader("🚨 Alerts")
    
    alerts = []
    
    null_exp = metrics.get('null_expert', {})
    health = metrics.get('routing_health', {})
    
    # Check conditions
    junk_rate = float(null_exp.get('junk_to_null_rate', 70))
    signal_rate = float(null_exp.get('signal_to_null_rate', 5))
    entropy = float(health.get('entropy', 0.85))
    dead = health.get('dead_experts', [])
    
    if junk_rate < 50:
        alerts.append({
            'severity': 'warning',
            'title': 'Low Junk → Null Rate',
            'message': f'Only {junk_rate:.1f}% of junk tokens routed to null (target: 60-80%)',
            'action': 'Consider increasing null expert bias'
        })
    
    if signal_rate > 15:
        alerts.append({
            'severity': 'warning',
            'title': 'High Signal Leakage',
            'message': f'{signal_rate:.1f}% of signal tokens leaked to null (target: <10%)',
            'action': 'Review router training or decrease null bias'
        })
    
    if entropy < 0.50:
        alerts.append({
            'severity': 'critical',
            'title': 'Router Entropy Collapse',
            'message': f'Entropy at {entropy:.3f} indicates possible expert collapse',
            'action': 'URGENT: Check for dead experts, consider router reset'
        })
    
    if dead:
        alerts.append({
            'severity': 'warning',
            'title': f'Dead Experts Detected',
            'message': f'Experts {dead} have <1% utilization',
            'action': 'Boost bias for affected experts'
        })
    
    if not alerts:
        st.success("✓ No active alerts - system healthy")
    else:
        for alert in alerts:
            if alert['severity'] == 'critical':
                st.error(f"**{alert['title']}**: {alert['message']}\n\n*Action: {alert['action']}*")
            else:
                st.warning(f"**{alert['title']}**: {alert['message']}\n\n*Action: {alert['action']}*")


def render_health_gates(metrics: Dict):
    """Render health gate details."""
    with st.expander("🔍 Health Gate Details"):
        gates = metrics.get('health_gates', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Null Routing Gates:**")
            for gate in ['null_junk_min', 'null_junk_max', 'null_signal_max']:
                status = "✅" if gates.get(gate, False) else "❌"
                st.markdown(f"- {status} {gate.replace('_', ' ').title()}")
        
        with col2:
            st.markdown("**Routing Health Gates:**")
            for gate in ['entropy_min', 'gini_max', 'no_dead_experts', 'no_overloaded_experts']:
                status = "✅" if gates.get(gate, False) else "❌"
                st.markdown(f"- {status} {gate.replace('_', ' ').title()}")


def render_sidebar():
    """Render sidebar with controls."""
    st.sidebar.header("⚙️ Controls")
    
    refresh_rate = st.sidebar.slider(
        "Refresh Rate (seconds)",
        min_value=1,
        max_value=30,
        value=5
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("📋 Thresholds")
    
    st.sidebar.number_input("Min Junk→Null (%)", value=60, key="thresh_junk_min")
    st.sidebar.number_input("Max Junk→Null (%)", value=80, key="thresh_junk_max")
    st.sidebar.number_input("Max Signal→Null (%)", value=10, key="thresh_signal")
    st.sidebar.number_input("Min Entropy", value=0.70, step=0.05, key="thresh_entropy")
    
    st.sidebar.markdown("---")
    
    st.sidebar.header("📊 Data")
    if st.sidebar.button("Export Metrics"):
        st.sidebar.download_button(
            "Download JSON",
            data=json.dumps(st.session_state.get('latest_metrics', {}), indent=2),
            file_name="moe_metrics.json",
            mime="application/json"
        )
    
    return refresh_rate


# =============================================================================
# Main Dashboard
# =============================================================================

def main():
    # Initialize
    data_source = get_data_source()
    
    # Sidebar
    refresh_rate = render_sidebar()
    
    # Header
    render_header()
    
    st.markdown("---")
    
    # Get latest metrics
    metrics = data_source.get_latest_metrics()
    st.session_state['latest_metrics'] = metrics
    
    # Key metrics
    render_key_metrics(metrics)
    
    st.markdown("---")
    
    # Status indicators
    render_status_indicators(metrics)
    
    st.markdown("---")
    
    # Trends
    history = data_source.get_history(100)
    render_trend_charts(history)
    
    st.markdown("---")
    
    # Expert utilization
    render_expert_utilization(data_source)
    
    st.markdown("---")
    
    # Alerts
    render_alerts(metrics)
    
    # Health gate details
    render_health_gates(metrics)
    
    # Auto-refresh
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
