# Component Integration Guide

This directory contains the "Self-Hosted Observability Stack" designed for 70B+ LLM training.
It replaces SaaS tools with high-performance, local-first components.

## 📁 Components

| # | File | Role |
|---|------|------|
| 1 | `train_logger/json_logger.py` | Non-blocking structured logger (The "Producer") |
| 2 | `system_metrics/collector.py` | System metrics → JSONL → ClickHouse (The "System Probe") |
| 3 | `metrics_server.py` | Custom JSON API for live metrics (The "Exporter") |
| 4 | `sidecar_agent/vector.toml` | Data shipper config (The "Shipper") |
| 5 | `watchdog/watchdog.py` | Control plane service (The "Enforcer") |
| 6 | `aggregation_api/dashboard_backend.py` | Aggregation API for the frontend (The "API") |
| 7 | `checkpoint_registry/checkpoint_registry.py` | Checkpoint governance (The "Registry") |

---

## 🖥️ Training Instance Setup

### 1. Python Packages

```bash
pip install psutil pyyaml numpy
pip install pynvml   # optional — only needed for GPU metrics
```

### 2. Files to Copy

```
components/
├── __init__.py
├── json_logger.py                     # re-export
├── train_logger/
│   └── json_logger.py                 # structured logger
├── metrics_server.py                  # custom JSON API metrics server
├── system_metrics/
│   ├── __init__.py
│   └── collector.py                   # system metrics → JSONL
└── sidecar_agent/
    └── vector.toml                    # Vector config
```

### 3. Vector Sidecar

```bash
# Install (once)
curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local

# Run
CLICKHOUSE_HTTP_ENDPOINT="http://<DB_INSTANCE_PRIVATE_IP>:8123" \
  vector --config /path/to/vector.toml --data-dir /tmp/vector-data
```

---

## � train.py Integration

Complete example — copy this into your `train.py`:

```python
import os
import time
from pathlib import Path
from components.json_logger import JSONLogger
from components.metrics_server import get_metrics_server
from components.system_metrics import SystemMetricsCollector

RUN_ID = "run_2026_02_11_exp1"
RANK = int(os.environ.get("RANK", 0))
LOG_DIR = "/tmp/training_logs"

# --- 1. Structured Logger (training metrics → JSONL → ClickHouse) ---
logger = JSONLogger(
    base_dir=LOG_DIR,
    run_id=RUN_ID,
    rank=RANK,
    default_context={"model": "70B_v4", "cluster": "us-east-1-p4d"},
)

# --- 2. System Metrics Collector (CPU/RAM/GPU/Disk/Net → JSONL → ClickHouse) ---
sys_collector = SystemMetricsCollector(
    log_dir=LOG_DIR,
    run_id=RUN_ID,
    rank=RANK,
    interval=5.0,
)

# --- 3. Metrics Server (live JSON API on port 8000 for watchdog / dashboard) ---
metrics = get_metrics_server()
metrics.start(system_collector=sys_collector)

# --- 4. Watchdog Control Plane Check ---
CONTROL_FILE = Path("/tmp/training_control.flag")

def check_control_plane():
    while CONTROL_FILE.exists():
        print("⏸️  PAUSED BY WATCHDOG. Waiting for resume...")
        time.sleep(10)

# --- 5. Training Loop ---
for step, batch in enumerate(dataloader):
    check_control_plane()
    loss = train_step(batch)

    if step % 10 == 0:
        logger.log_step(
            step=step,
            metrics={"loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
        )
        metrics.update_training_metrics(
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            step=step,
        )

# --- 6. Cleanup ---
logger.close()
metrics.stop()   # stops both the HTTP server and the system collector
```

---

## 📊 Metrics Server Endpoints

The metrics server starts on **port 8000** by default (no config file needed).

| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | Full snapshot of all current metric values (JSON) |
| `GET /query?metric=training_loss` | Single metric current value |
| `GET /history?metric=training_loss&since=<epoch>` | Time-series history |
| `GET /health` | Liveness probe |

To use a custom port, create a `config.yaml`:

```yaml
training:
  metrics_port: 9090
```

Then pass it: `get_metrics_server("config.yaml")`

---

## 🖥️ System Metrics Collected

All metrics land in ClickHouse `metric_points` table via the Vector pipeline.

| Category | Metrics |
|----------|---------|
| **CPU** | `sys.cpu_percent`, `sys.cpu_freq_mhz`, `sys.cpu_count`, `sys.load_1m/5m/15m` |
| **Memory** | `sys.mem_percent`, `sys.mem_used_bytes`, `sys.mem_available_bytes`, `sys.swap_percent` |
| **Disk** | `sys.disk.<mount>.percent`, `sys.disk.<mount>.free_bytes`, `sys.disk.<mount>.used_bytes` |
| **Network** | `sys.net.<iface>.sent_bytes_per_s`, `sys.net.<iface>.recv_bytes_per_s` |
| **GPU** | `sys.gpu.<idx>.util_percent`, `sys.gpu.<idx>.mem_percent`, `sys.gpu.<idx>.temperature_c`, `sys.gpu.<idx>.power_w` |

GPU metrics require `pynvml` (`pip install pynvml`). Gracefully disabled if not installed.

---

## 🛡️ The Watchdog (Active Control)

The Watchdog polls the metrics server and pauses training on anomalies (e.g. loss divergence).

```bash
python -m components.watchdog.watchdog
```

The watchdog writes to `/tmp/training_control.flag` when triggered. The `check_control_plane()` function in the train.py example above reads this flag.

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
