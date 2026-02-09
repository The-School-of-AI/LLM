
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Optional
import time
import random
import uvicorn
from pydantic import BaseModel

# ------------------------------------------------------------------------------
# Models
# ------------------------------------------------------------------------------

class RunInfo(BaseModel):
    run_id: str
    status: str
    start_time: str
    context: Dict

class MetricPoint(BaseModel):
    timestamp: float
    step: int
    value: float

class MetricResponse(BaseModel):
    run_id: str
    metric: str
    data: List[MetricPoint]

# ------------------------------------------------------------------------------
# Mock Adapters (Replace these with real clients in Prod)
# ------------------------------------------------------------------------------

class MockClickHouseClient:
    """Simulates fetching historical data from ClickHouse"""
    def get_runs(self):
        return [
            {
                "run_id": "run_2026_02_08_exp1",
                "status": "running",
                "start_time": "2026-02-08T10:00:00Z",
                "context": {"model": "70B", "source": "growth/lora"}
            },
            {
                "run_id": "run_2026_02_07_baseline",
                "status": "completed",
                "start_time": "2026-02-07T08:00:00Z",
                "context": {"model": "70B", "source": "pretrain"}
            }
        ]

    def get_metric_history(self, run_id: str, metric: str, end_time: float):
        """Get data older than end_time"""
        # Generate fake history curve
        history = []
        start_ts = end_time - 3600 # 1 hour back
        for i in range(100):
            ts = start_ts + (i * 30) # every 30s
            val = 2.5 - (i * 0.01) + (random.random() * 0.1)
            history.append({"timestamp": ts, "step": i*10, "value": val})
        return history

class MockPrometheusClient:
    """Simulates fetching live data from Prometheus"""
    def get_metric_range(self, run_id: str, metric: str, start_time: float):
        """Get data newer than start_time"""
        # Generate fake live points
        live = []
        now = time.time()
        for i in range(10):
            ts = start_time + (i * 5) # every 5s
            if ts > now: break
            val = 1.5 + (random.random() * 0.05) # Converged value
            live.append({"timestamp": ts, "step": 1000 + (i), "value": val})
        return live

# ------------------------------------------------------------------------------
# App Logic
# ------------------------------------------------------------------------------

app = FastAPI(title="Training Dashboard Aggregator")
ch_client = MockClickHouseClient()
prom_client = MockPrometheusClient()

@app.get("/runs", response_model=Dict[str, List[RunInfo]])
async def list_runs():
    """List all available training runs from ClickHouse"""
    runs = ch_client.get_runs()
    return {"runs": runs}

@app.get("/metrics", response_model=MetricResponse)
async def get_metrics(
    run_id: str, 
    metric: str, 
    window: str = Query("1h", description="Time window e.g. 1h, 24h")
):
    """
    Unified Metric Query:
    Merges Cold Data (ClickHouse) + Hot Data (Prometheus)
    """
    now = time.time()
    cutoff_time = now - 300 # Last 5 minutes is "Hot"
    
    # 1. Fetch Cold Data (History)
    cold_data = ch_client.get_metric_history(run_id, metric, end_time=cutoff_time)
    
    # 2. Fetch Hot Data (Live)
    hot_data = prom_client.get_metric_range(run_id, metric, start_time=cutoff_time)
    
    # 3. Merge & Sort
    merged = cold_data + hot_data
    merged.sort(key=lambda x: x['timestamp'])
    
    return {
        "run_id": run_id,
        "metric": metric,
        "data": merged
    }

if __name__ == "__main__":
    print("Starting Dashboard Backend on http://localhost:8081")
    uvicorn.run(app, host="0.0.0.0", port=8081)
