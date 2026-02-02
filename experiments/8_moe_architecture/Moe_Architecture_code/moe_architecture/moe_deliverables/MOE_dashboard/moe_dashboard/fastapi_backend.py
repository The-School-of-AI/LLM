#!/usr/bin/env python3
"""
FastAPI Backend for MoE Routing Dashboard
==========================================
Production-ready REST API + WebSocket server for real-time monitoring.

Features:
- REST API for metrics retrieval
- WebSocket for real-time updates
- Redis integration for distributed metrics
- CORS support for web frontends
- Health check endpoints
- Prometheus metrics export

Setup:
    pip install fastapi uvicorn websockets redis prometheus-client

Run:
    uvicorn fastapi_backend:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health           - Health check
    GET  /metrics          - Latest metrics
    GET  /metrics/history  - Historical metrics
    GET  /experts          - Expert utilization
    WS   /ws               - WebSocket for real-time updates
    GET  /prometheus       - Prometheus metrics
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime
import random

# Optional imports
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="MoE Routing Dashboard API",
    description="Real-time monitoring API for Team 7 MoE routing diagnostics",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Data Models
# =============================================================================

class NullExpertMetrics(BaseModel):
    junk_to_null_rate: float
    boilerplate_to_null_rate: float
    signal_to_null_rate: float
    compute_savings_pct: float


class RoutingHealthMetrics(BaseModel):
    entropy: float
    gini_coefficient: float
    dead_experts: List[int]
    overloaded_experts: List[int]


class StabilityMetrics(BaseModel):
    is_stable: bool
    stability_score: float
    lora_ready: bool


class HealthGates(BaseModel):
    null_junk_min: bool
    null_junk_max: bool
    null_signal_max: bool
    entropy_min: bool
    gini_max: bool
    no_dead_experts: bool
    no_overloaded_experts: bool


class GrowthTrigger(BaseModel):
    recommend_growth: bool
    confidence: float


class DashboardMetrics(BaseModel):
    timestamp: str
    step: int
    null_expert: NullExpertMetrics
    routing_health: RoutingHealthMetrics
    stability: StabilityMetrics
    health_gates: HealthGates
    growth_trigger: GrowthTrigger
    all_gates_pass: bool


class ExpertUtilization(BaseModel):
    expert_id: int
    utilization: float
    status: str  # "healthy", "underutilized", "overloaded"


# =============================================================================
# Metrics Storage (In-memory for demo, use Redis in production)
# =============================================================================

class MetricsStore:
    """In-memory metrics storage with optional Redis backend."""
    
    def __init__(self, redis_url: str = None):
        self.history: List[Dict] = []
        self.max_history = 10000
        self.current_step = 0
        self.redis_client = None
        
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                print(f"Connected to Redis: {redis_url}")
            except Exception as e:
                print(f"Redis connection failed: {e}")
    
    def push_metrics(self, metrics: Dict):
        """Push new metrics to store."""
        self.history.append(metrics)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self.current_step = metrics.get('step', self.current_step + 1)
        
        # Also push to Redis if available
        if self.redis_client:
            try:
                self.redis_client.lpush('moe_metrics', json.dumps(metrics))
                self.redis_client.ltrim('moe_metrics', 0, self.max_history)
            except:
                pass
    
    def get_latest(self) -> Optional[Dict]:
        """Get latest metrics."""
        if self.history:
            return self.history[-1]
        return None
    
    def get_history(self, n: int = 100) -> List[Dict]:
        """Get last N metrics."""
        return self.history[-n:]
    
    def generate_demo_metrics(self) -> Dict:
        """Generate demo metrics for testing."""
        step = self.current_step
        self.current_step += 1
        
        import math
        base_junk = 68 + math.sin(step / 50) * 5
        base_entropy = 0.85 + math.sin(step / 100) * 0.05
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'null_expert': {
                'junk_to_null_rate': round(base_junk + random.uniform(-2, 2), 1),
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
                'null_junk_min': base_junk >= 60,
                'null_junk_max': base_junk <= 80,
                'null_signal_max': True,
                'entropy_min': base_entropy >= 0.70,
                'gini_max': True,
                'no_dead_experts': random.random() > 0.1,
                'no_overloaded_experts': random.random() > 0.05,
            },
            'growth_trigger': {
                'recommend_growth': random.random() > 0.8,
                'confidence': round(random.uniform(0.6, 0.95), 2),
            },
        }
        
        metrics['all_gates_pass'] = all(metrics['health_gates'].values())
        self.push_metrics(metrics)
        
        return metrics


# Initialize store
metrics_store = MetricsStore()


# =============================================================================
# Prometheus Metrics (Optional)
# =============================================================================

if PROMETHEUS_AVAILABLE:
    prom_junk_null_rate = Gauge('moe_null_junk_rate', 'Junk to null routing rate')
    prom_signal_null_rate = Gauge('moe_null_signal_rate', 'Signal to null routing rate')
    prom_entropy = Gauge('moe_routing_entropy', 'Routing entropy')
    prom_dead_experts = Gauge('moe_dead_expert_count', 'Number of dead experts')


# =============================================================================
# WebSocket Connection Manager
# =============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.active_connections.remove(conn)


manager = ConnectionManager()


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MoE Dashboard API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2196f3; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🎯 MoE Routing Dashboard API</h1>
        <p>Real-time monitoring API for Team 7 MoE routing diagnostics</p>
        
        <h2>Endpoints</h2>
        <div class="endpoint">
            <strong>GET</strong> <code>/health</code> - Health check
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/metrics</code> - Latest metrics
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/metrics/history?n=100</code> - Historical metrics
        </div>
        <div class="endpoint">
            <strong>GET</strong> <code>/experts</code> - Expert utilization
        </div>
        <div class="endpoint">
            <strong>WS</strong> <code>/ws</code> - WebSocket for real-time updates
        </div>
        <div class="endpoint">
            <strong>POST</strong> <code>/metrics</code> - Push new metrics (from training)
        </div>
        
        <h2>Interactive Docs</h2>
        <p><a href="/docs">Swagger UI</a> | <a href="/redoc">ReDoc</a></p>
        
        <h2>WebSocket Example</h2>
        <pre>
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onmessage = (event) => {
    const metrics = JSON.parse(event.data);
    console.log(metrics);
};
        </pre>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "metrics_count": len(metrics_store.history),
        "websocket_clients": len(manager.active_connections),
    }


@app.get("/metrics", response_model=DashboardMetrics)
async def get_metrics():
    """Get latest metrics."""
    metrics = metrics_store.get_latest()
    if not metrics:
        # Generate demo metrics if none available
        metrics = metrics_store.generate_demo_metrics()
    return metrics


@app.get("/metrics/history")
async def get_metrics_history(n: int = 100):
    """Get historical metrics."""
    return {
        "count": min(n, len(metrics_store.history)),
        "metrics": metrics_store.get_history(n)
    }


@app.post("/metrics")
async def push_metrics(metrics: DashboardMetrics):
    """Push new metrics from training process."""
    metrics_dict = metrics.dict()
    metrics_dict['timestamp'] = datetime.now().isoformat()
    
    metrics_store.push_metrics(metrics_dict)
    
    # Update Prometheus metrics
    if PROMETHEUS_AVAILABLE:
        prom_junk_null_rate.set(metrics.null_expert.junk_to_null_rate)
        prom_signal_null_rate.set(metrics.null_expert.signal_to_null_rate)
        prom_entropy.set(metrics.routing_health.entropy)
        prom_dead_experts.set(len(metrics.routing_health.dead_experts))
    
    # Broadcast to WebSocket clients
    await manager.broadcast(metrics_dict)
    
    return {"status": "ok", "step": metrics.step}


@app.get("/experts")
async def get_expert_utilization():
    """Get per-expert utilization."""
    # Generate demo data
    experts = []
    for i in range(64):
        util = random.uniform(0.8, 1.2) * (100 / 64)
        status = "healthy"
        if util < 1:
            status = "underutilized"
        elif util > 3:
            status = "overloaded"
        
        experts.append({
            "expert_id": i,
            "utilization": round(util, 2),
            "status": status
        })
    
    return {"experts": experts, "total": 64}


@app.get("/alerts")
async def get_alerts():
    """Get active alerts."""
    metrics = metrics_store.get_latest()
    if not metrics:
        return {"alerts": []}
    
    alerts = []
    
    null_exp = metrics.get('null_expert', {})
    health = metrics.get('routing_health', {})
    
    if null_exp.get('junk_to_null_rate', 70) < 50:
        alerts.append({
            "severity": "warning",
            "title": "Low Junk → Null Rate",
            "message": f"Rate at {null_exp.get('junk_to_null_rate')}%",
        })
    
    if health.get('entropy', 0.85) < 0.50:
        alerts.append({
            "severity": "critical",
            "title": "Entropy Collapse",
            "message": f"Entropy at {health.get('entropy')}",
        })
    
    if health.get('dead_experts'):
        alerts.append({
            "severity": "warning",
            "title": "Dead Experts",
            "message": f"Experts {health.get('dead_experts')} underutilized",
        })
    
    return {"alerts": alerts, "count": len(alerts)}


if PROMETHEUS_AVAILABLE:
    @app.get("/prometheus")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        return PlainTextResponse(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send current metrics immediately
        metrics = metrics_store.get_latest()
        if metrics:
            await websocket.send_json(metrics)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for any message (can be used for client commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Handle client commands
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "get_latest":
                    metrics = metrics_store.get_latest() or metrics_store.generate_demo_metrics()
                    await websocket.send_json(metrics)
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# =============================================================================
# Background Task for Demo Mode
# =============================================================================

async def generate_demo_data():
    """Background task to generate demo data."""
    while True:
        metrics = metrics_store.generate_demo_metrics()
        await manager.broadcast(metrics)
        await asyncio.sleep(2)  # Update every 2 seconds


@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    # Uncomment for demo mode:
    asyncio.create_task(generate_demo_data())
    print("🚀 MoE Dashboard API started")
    print("📊 Demo mode: generating synthetic metrics")


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_ping_interval=30,
        ws_ping_timeout=30,
    )
