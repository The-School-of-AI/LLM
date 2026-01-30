# 🎯 Team 7 - MoE Routing Dashboard

Real-time monitoring dashboard for MoE (Mixture of Experts) routing diagnostics.

![Dashboard Preview](https://via.placeholder.com/800x400?text=MoE+Routing+Dashboard)

## 📊 Features

| Feature | Description |
|---------|-------------|
| **Null Expert Monitoring** | Track junk/signal/boilerplate → null routing rates |
| **Health Gates** | Visual status of all 7 health gate conditions |
| **Expert Heatmap** | 64-expert utilization visualization |
| **Curriculum Analysis** | B0-B5 bucket routing distribution |
| **Real-time Trends** | Live charts with configurable refresh |
| **Alerts** | Automatic alerts for threshold violations |
| **LoRA/Growth Status** | Milestone tracking for LoRA-readiness |

---

## 🚀 Quick Start

### Option 1: Local Development (Fastest)

```bash
# Install dependencies
pip install streamlit plotly pandas numpy

# Run dashboard
streamlit run app.py
```

Open: http://localhost:8501

### Option 2: Docker (Recommended for Production)

```bash
# Build and run
docker build -t moe-dashboard .
docker run -p 8501:8501 moe-dashboard
```

### Option 3: Full Stack (Grafana + Prometheus)

```bash
# Start all services
docker-compose up -d

# Access:
# - Dashboard:  http://localhost:8501
# - Grafana:    http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9090
# - API:        http://localhost:8000
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Free & Easy)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy!

### AWS/GCP/Azure

```bash
# Build Docker image
docker build -t moe-dashboard .

# Push to registry
docker tag moe-dashboard your-registry/moe-dashboard:latest
docker push your-registry/moe-dashboard:latest

# Deploy to Kubernetes, ECS, Cloud Run, etc.
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: moe-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: moe-dashboard
  template:
    metadata:
      labels:
        app: moe-dashboard
    spec:
      containers:
      - name: dashboard
        image: your-registry/moe-dashboard:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: moe-dashboard
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8501
  selector:
    app: moe-dashboard
```

---

## 🔧 Integration with Training

### Connect Real Data

Replace the `MetricsSimulator` class with your actual data source:

```python
# Option 1: Redis (Real-time)
import redis

class RedisMetricsSource:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.client = redis.from_url(redis_url)
    
    def get_metrics(self):
        data = self.client.get("moe_metrics")
        return json.loads(data) if data else None

# Option 2: REST API
import requests

class APIMetricsSource:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
    
    def get_metrics(self):
        response = requests.get(f"{self.api_url}/metrics")
        return response.json()

# Option 3: WebSocket (Real-time)
import websockets

class WebSocketMetricsSource:
    async def connect(self, url="ws://localhost:8000/ws"):
        async with websockets.connect(url) as ws:
            while True:
                data = await ws.recv()
                yield json.loads(data)
```

### Push Metrics from Training Loop

```python
from moe_tools.diagnostics import RoutingDiagnostics
import redis
import json

# Initialize
diagnostics = RoutingDiagnostics(config)
redis_client = redis.from_url("redis://localhost:6379")

# In training loop
for step in range(num_steps):
    # ... training code ...
    
    # Log routing metrics
    for layer_idx in range(num_layers):
        diagnostics.log_batch(
            layer_idx=layer_idx,
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            token_ids=token_ids,
        )
    
    # Push to dashboard
    snapshot = diagnostics.step()
    metrics = diagnostics.get_dashboard_metrics()
    redis_client.set("moe_metrics", json.dumps(metrics))
```

---

## 📈 Metrics Reference

### Null Expert Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `junk_to_null_rate` | 60-80% | Junk tokens routed to null expert |
| `signal_to_null_rate` | <10% | Signal tokens leaked to null |
| `boilerplate_to_null_rate` | 40-70% | Boilerplate tokens to null |
| `compute_savings_pct` | >10% | FLOPs saved by null routing |

### Routing Health Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| `entropy` | >0.70 | Routing diversity (1.0 = uniform) |
| `gini_coefficient` | <0.50 | Load balance (0 = perfect) |
| `dead_experts` | 0 | Experts with <1% utilization |
| `overloaded_experts` | 0 | Experts with >3x expected load |

### Stability Metrics

| Metric | Description |
|--------|-------------|
| `stability_score` | Routing stability over time |
| `is_stable` | Boolean stability flag |
| `lora_ready` | MoE block ready for LoRA |

---

## 🚨 Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| Low Junk→Null | rate < 50% | Warning |
| High Signal Leakage | rate > 15% | Warning |
| Entropy Collapse | entropy < 0.50 | **Critical** |
| Load Imbalance | gini > 0.50 | Warning |
| Dead Experts | count > 0 | Warning |

---

## 🎨 Customization

### Change Thresholds

Edit in sidebar or modify `ThresholdConfig`:

```python
@dataclass
class ThresholdConfig:
    null_junk_min: float = 60.0
    null_junk_max: float = 80.0
    null_signal_max: float = 10.0
    entropy_min: float = 0.70
    gini_max: float = 0.50
```

### Add Custom Panels

```python
def render_custom_panel(metrics):
    st.subheader("My Custom Panel")
    
    # Your visualization code
    fig = px.line(...)
    st.plotly_chart(fig)
```

---

## 📁 File Structure

```
moe_dashboard/
├── app.py                  # Main Streamlit dashboard
├── fastapi_backend.py      # REST API + WebSocket server
├── wandb_dashboard.py      # W&B integration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build for Streamlit
├── docker-compose.yml      # Full stack deployment
├── grafana_dashboard.json  # Grafana dashboard config
└── README.md               # This file
```

---

## 🔗 Related Tools

| Tool | Purpose |
|------|---------|
| `moe_tools/diagnostics/` | Routing diagnostics engine |
| `moe_tools/estimators/` | FLOPs/memory estimation |
| `moe_tools/profilers/` | Training profiler |

---

## 📝 Team 7 Objectives Supported

✅ **Null Expert Fire Rate**: Track junk/boilerplate/signal routing  
✅ **Token → Expert Mapping**: Curriculum bucket analysis  
✅ **Routing Specialization**: Expert utilization heatmaps  
✅ **LoRA-Readiness**: Stability milestone tracking  
✅ **Growth Triggers**: Expert expansion recommendations  
✅ **Compute Savings**: FLOPs reduction monitoring  

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -am 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Create Pull Request

---

## 📄 License

MIT License - see LICENSE file for details.

---

**Built by Team 8 (Architecture) for Team 7 (Routing Diagnostics)**
