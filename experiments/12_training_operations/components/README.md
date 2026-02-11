# Component Integration Guide

This directory contains the "Self-Hosted Observability Stack" designed for 70B+ LLM training.
It replaces SaaS tools with high-performance, local-first components.

## 📁 Components

1.  **`json_logger.py`**: A non-blocking structured logger (The "Producer").
2.  **`vector.toml`**: A configuration for the data shipper (The "Shipper").
3.  **`watchdog.py`**: A control plane service (The "Enforcer").
4.  **`metrics_server.py`**: Custom JSON API server for system/training metrics (The "Exporter").
5.  **`dashboard_backend.py`**: Aggregation API for the frontend (The "API").
6.  **`checkpoint_registry.py`**: Database for checkpoint governance (The "Registry").
7.  **`system_architecture.md`**: Master architecture document and data flow diagram.

---

## 🚀 Quick Start: The Training Logger

### 1. Initialization
In your `train.py`, initialize the logger **once** before the loop.
Pass static metadata (like `source`, `model_size`) in `default_context`.

```python
from components.json_logger import JSONLogger

# Define "Static" metadata here
context = {
    "source": "growth_team/lora_experiments",
    "model": "70B_v4",
    "cluster": "us-east-1-p4d",
    "hyperparams": {
        "lr": 0.001,
        "batch_size": 256
    }
}

# Initialize (Rank-Aware)
logger = JSONLogger(
    base_dir="/tmp/training_logs", 
    run_id="run_2026_02_08_exp1",
    rank=int(os.environ.get("RANK", 0)),
    default_context=context
)
```

### 2. Logging in the Loop
Call `log_step` freely. It is non-blocking (async).

```python
for step, batch in enumerate(dataloader):
    # ... training logic ...
    
    if step % 10 == 0:
        # Log Scalars (Loss, LR)
        metrics = {
            "loss": loss.item(),
            "lr": scheduler.get_last_lr()[0],
            "tokens_per_sec": tps
        }
        
        # Log Rich Data (Routing Distribs, Text)
        # Javascript/ClickHouse handles the arrays automatically
        rich_data = {
            "routing_dist": routing_probs.mean(0).cpu().numpy(),
            "grad_norm": grad_norm.item()
        }
        
        logger.log_step(step=step, metrics=metrics, context=rich_data)

# Always close at end
logger.close()
```

---

## 🚛 The Sidecar (Vector)
The logger writes to disk. **Vector** ships it to ClickHouse.

1.  **Install Vector**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.vector.dev | sh`
2.  **Run with Config**:
    ```bash
    vector --config components/vector.toml
    ```
3.  **Verify**: You should see logs printing to the console (Debug Sink).
4.  **Deploy**: Uncomment the `[sinks.clickhouse_prod]` section in `vector.toml` to ship to DB.

---

## 📊 The Metrics Server (Custom JSON API)
Exposes "Hot" data (CPU/RAM, Loss) for real-time querying.

### Usage
In `train.py`, start the server and push updates:

```python
from components.metrics_server import get_metrics_server

# 1. Start Server (Default Port 8000)
metrics = get_metrics_server("config.yaml")
metrics.start()

# 2. Update in Loop
metrics.update_training_metrics(
    loss=loss.item(),
    lr=scheduler.get_last_lr()[0],
    step=step
)
```

### Endpoints
- `GET /metrics` — Full snapshot of all current metric values (JSON)
- `GET /query?metric=training_loss` — Single metric current value
- `GET /history?metric=training_loss&since=<epoch>` — Time-series history
- `GET /health` — Liveness probe

---

## 🛡️ The Watchdog (Active Control)
The Watchdog enforces layout safety (SEV-1 actions).

1.  **Run Service**: `python components/watchdog.py`
2.  **Modify Training Loop**: You MUST check for the pause flag.

```python
# In train.py
import time
from pathlib import Path

CONTROL_FILE = Path("/tmp/training_control.flag")

def check_control_plane():
    while CONTROL_FILE.exists():
        print("⏸️  PAUSED BY WATCHDOG. Waiting for resume...")
        time.sleep(10)

# Inside loop
for step in ...:
    check_control_plane() # <--- Add this
    train_step()
```

---

## 🏛️ The Checkpoint Registry (Governance)
Enforces "No Delete" rules for Growth/LoRA checkpoints.

### Usage
```python
from components.checkpoint_registry import CheckpointRegistry

# 1. Connect (Local SQLite or AWS RDS)
# db_url = "postgresql://user:pass@rds-endpoint:5432/training_db"
db_url = "sqlite:///checkpoints.db"
registry = CheckpointRegistry(db_url)

# 2. Register after S3 Upload
registry.register_checkpoint(
    run_id="run_1",
    step=1000,
    s3_key="s3://bucket/ckpt_1000.pt",
    loss=0.15,
    tag="growth" # <--- Automatically PROTECTED
)

# 3. Check before Deletion
if registry.can_delete("s3://bucket/ckpt_1000.pt"):
    resize_s3_bucket() # Safe to delete
else:
    print("Skipping protected checkpoint")
```
