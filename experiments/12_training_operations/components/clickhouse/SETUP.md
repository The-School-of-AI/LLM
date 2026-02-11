# Observability Stack Setup

Two instances: **DB instance** (ClickHouse) and **Training instance** (logger + Vector sidecar + system metrics collector).

## Architecture

```
Training Instance                         DB Instance
┌──────────────────────┐                  ┌──────────────────┐
│ Training loop        │                  │ ClickHouse       │
│   → JSONLogger ──────┼──┐              │   logs           │
│                      │  ├─ .jsonl ──┐  │   metric_points  │
│ SystemMetricsCollector──┘           │  │   metric_arrays  │
│   (CPU/RAM/GPU/Disk/Net)            │  │   events         │
│                      │  HTTP :8123  │  │   runs           │
│ Vector sidecar ──────┼──────────────┼─▶│                  │
│   (fan-out)          │              │  └──────────────────┘
│                      │              │         ▲
│ MetricsServer :8000  │              │  Dashboard API queries
│   (live JSON API)    │              │  metric_points
└──────────────────────┘              │
                                      └─ /tmp/training_logs/
```

## DB Instance Setup

### 1) Start ClickHouse (Docker)

```bash
cd experiments/12_training_operations/components/clickhouse
sudo docker compose up -d
```

This auto-applies schema from `initdb.d/` on first start (creates `training_observability` database and all tables).

**Gotcha:** The official Docker image restricts the `default` user to localhost. We override this via `users.d/default-allow-remote.xml` (mounted in docker-compose) to allow remote connections from Vector and the Dashboard API.

### 2) Verify

```bash
sudo docker exec p12-clickhouse clickhouse-client --query "SHOW TABLES FROM training_observability"
# Expected: events, logs, metric_arrays, metric_points, runs

curl -s http://localhost:8123/ping
# Expected: Ok.
```

### 3) Wipe and restart (if needed)

```bash
sudo docker compose down -v   # -v removes volumes (all data)
sudo docker compose up -d     # fresh start, schema re-applied
```

### 4) AWS Security Group

Open **TCP 8123** inbound from the training instance IP/security group. Without this, Vector gets `Connection refused`.

---

## Training Instance Setup

### 1) Install Python packages

```bash
pip install psutil pyyaml numpy
pip install pynvml   # optional — only needed for GPU metrics
```

### 2) Install Vector (>= 0.30)

```bash
curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local
vector --version
```

### 3) Copy files to the training instance

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

### 4) Create Vector data directory

```bash
sudo mkdir -p /var/lib/vector && sudo chown $(whoami) /var/lib/vector
```

Or use `--data-dir /tmp/vector-data` flag instead.

### 5) Verify connectivity (before starting training)

```bash
curl -s http://<DB_INSTANCE_PRIVATE_IP>:8123/ping
# Must return: Ok.
```

If this fails → check AWS security group (port 8123).

### 6) Run Vector

```bash
CLICKHOUSE_HTTP_ENDPOINT="http://<DB_INSTANCE_PRIVATE_IP>:8123" \
  vector --config /path/to/vector.toml --data-dir /tmp/vector-data
```

### 7) Integrate into train.py

```python
import os
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
    interval=5.0,       # collect every 5 seconds
)

# --- 3. Metrics Server (live JSON API for watchdog / dashboard) ---
metrics = get_metrics_server()          # no config file needed, defaults to port 8000
metrics.start(system_collector=sys_collector)

# --- 4. Training Loop ---
for step, batch in enumerate(dataloader):
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

# --- 5. Cleanup ---
logger.close()
metrics.stop()   # stops both the HTTP server and the system collector
```

---

## Data Flow

Vector reads `/tmp/training_logs/**/*.jsonl` and does two things per log line:

- **`logs` table** — raw row with `metrics`/`context` as JSON strings (audit trail)
- **`metric_points` table** — one row per numeric metric key (dashboard-friendly)

Only numeric metric values land in `metric_points`. Non-numeric values (strings, arrays) are skipped.

---

## Dashboard API

The custom Dashboard API (`dashboard_backend.py`) queries ClickHouse directly for historical data and the custom metrics server for live data.

Query **`training_observability.metric_points`** for both training and system metrics:

- **Time column:** `event_time`
- **Metric name:** `metric` (e.g. `loss`, `lr`, `sys.cpu_percent`, `sys.mem_percent`)
- **Value:** `value` (Float64)
- **Filters:** `run_id`, `host`, `rank`

---

## Verify Ingestion

On the DB instance:

```bash
sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT count() FROM training_observability.logs"

sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT metric, count(), min(value), max(value) FROM training_observability.metric_points GROUP BY metric"
```

---

## Tables

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `logs` | Raw audit trail | `event_time`, `timestamp`, `step`, `metrics` (JSON), `context` (JSON), `run_id`, `host`, `rank` |
| `metric_points` | Typed scalars (training + system) | `event_time`, `metric`, `value`, `step`, `run_id`, `host`, `rank` |
| `metric_arrays` | Array metrics (grad norms, etc.) | `event_time`, `metric`, `keys`, `values`, `step`, `run_id` |
| `events` | Discrete events (checkpoint, OOM) | `event_time`, `event_type`, `severity`, `message`, `run_id` |
| `runs` | Run metadata | `run_id`, `status`, `model_name`, `cluster` |

---

## Logging (Training Code)

```python
from components.json_logger import JSONLogger

logger = JSONLogger(
    base_dir="/tmp/training_logs",
    run_id="run_001",
    rank=0,
    default_context={"model": "1B_v4", "cluster": "us-east-1"},
)

# Metrics must be numeric (float/int) — not raw tensors
logger.log_step(step=1, metrics={"loss": loss.item(), "lr": 0.001})
logger.close()
```

**Important:** Call `.item()` on tensors before logging. The serializer handles this automatically for PyTorch tensors, but explicit `.item()` is preferred.
