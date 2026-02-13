# Observability Stack Setup

Two instances: **DB instance** (ClickHouse) and **Training instance** (TrainingOps + Vector sidecar).

## Architecture

```
Training Instance                              DB Instance
┌────────────────────────────────┐             ┌──────────────────────┐
│                                │             │ ClickHouse           │
│  TrainingOps (single object)   │             │                      │
│    ├─ JSONLogger ──────┐       │             │   logs               │
│    ├─ SystemMetrics ───┤ .jsonl│             │   metric_points      │
│    ├─ MetricsServer    │       │             │   checkpoints        │
│    └─ CheckpointRegistry       │             │   events             │
│                        │       │             │   runs               │
│  Vector sidecar ───────┼───────┼── HTTP ───▶ │                      │
│    ├─ to_raw_logs      │       │    :8123    └──────────────────────┘
│    ├─ to_metric_points │       │
│    └─ to_checkpoints   │       │
│                        │       │
│  /tmp/training_logs/*.jsonl    │
└────────────────────────────────┘
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
├── training_ops.py                    # TrainingOps facade
├── json_logger.py                     # re-export
├── train_logger/
│   └── json_logger.py                 # structured logger
├── metrics_server.py                  # custom JSON API metrics server
├── system_metrics/
│   ├── __init__.py
│   └── collector.py                   # system metrics → JSONL
├── checkpoint_registry/
│   ├── __init__.py
│   └── checkpoint_registry.py         # ClickHouse-backed governance
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
# Vector MUST be running before training starts — TrainingOps checks for it.
CLICKHOUSE_HTTP_ENDPOINT="http://<DB_INSTANCE_PRIVATE_IP>:8123" \
  vector --config /path/to/vector.toml --data-dir /tmp/vector-data
```

### 7) Integrate into train.py

`TrainingOps` is the single entry point. One import, one init, two methods, one cleanup.

```python
import os
from components import TrainingOps

# --- 1. Initialize (starts all backend services, runs preflight checks) ---
ops = TrainingOps(
    run_id="run_2026_02_13_70b_v4",
    rank=int(os.environ.get("RANK", 0)),
    log_dir="/tmp/training_logs",
    default_context={"model": "70B_v4", "cluster": "us-east-1-p4d"},
)

# --- 2. Training Loop ---
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    # Log every N steps
    if step % 10 == 0:
        ops.log_step(
            step=step,
            metrics={"loss": loss.item(), "lr": scheduler.get_last_lr()[0]},
        )

    # After saving a checkpoint
    if step % 1000 == 0:
        path = f"checkpoints/ckpt_step_{step}.pt"
        torch.save(checkpoint_state, path)
        ops.log_checkpoint(
            step=step,
            path=path,
            s3_key=f"s3://bucket/ckpt_step_{step}.pt",  # omit if local-only
            loss=loss.item(),
            tag="temporary",
        )

# --- 3. Cleanup ---
ops.shutdown()
```

---

## Data Flow

Vector reads `/tmp/training_logs/**/*.jsonl` and routes each log line to multiple ClickHouse tables:

- **`logs` table** — raw row with `metrics`/`context` as JSON strings (audit trail)
- **`metric_points` table** — one row per numeric metric key (dashboard-friendly)
- **`checkpoints` table** — checkpoint events only (filtered by `context.event == "checkpoint_saved"`)

Only numeric metric values land in `metric_points`. Non-numeric values (strings, arrays) are skipped.

Checkpoint registration has a **dual path**: the JSONL → Vector route is the durable guarantee (buffered, retried). A best-effort direct HTTP INSERT provides immediate query-ability when ClickHouse is reachable.

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

sudo docker exec p12-clickhouse clickhouse-client --query \
  "SELECT run_id, step, s3_key, tag, is_protected, status FROM training_observability.checkpoints FINAL"
```

---

## Tables

| Table | Purpose | Key columns |
|-------|---------|-------------|
| `logs` | Raw audit trail | `event_time`, `timestamp`, `step`, `metrics` (JSON), `context` (JSON), `run_id`, `host`, `rank` |
| `metric_points` | Typed scalars (training + system) | `event_time`, `metric`, `value`, `step`, `run_id`, `host`, `rank` |
| `metric_arrays` | Array metrics (grad norms, etc.) | `event_time`, `metric`, `keys`, `values`, `step`, `run_id` |
| `checkpoints` | Checkpoint governance (ReplacingMergeTree) | `run_id`, `s3_key`, `step`, `loss`, `tag`, `is_protected`, `status` |
| `events` | Discrete events (OOM, etc.) | `event_time`, `event_type`, `severity`, `message`, `run_id` |
| `runs` | Run metadata | `run_id`, `status`, `model_name`, `cluster` |
