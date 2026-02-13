# Component Integration Guide

This directory contains the "Self-Hosted Observability Stack" designed for 70B+ LLM training.
It replaces SaaS tools with high-performance, local-first components.

## Architecture

```
Training Instance                              DB Instance
┌────────────────────────────────┐             ┌──────────────────────┐
│                                │             │ ClickHouse           │
│  TrainingOps (single object)   │             │                      │
│    ├─ JSONLogger ──────┐       │             │   logs               │
│    ├─ SystemMetrics ───┤ .jsonl│             │   metric_points      │
│    ├─ MetricsServer    │       │             │   checkpoints  ◄─────┤
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

**Data flows through JSONL → Vector → ClickHouse for everything.** The training instance never needs a direct connection to ClickHouse. Vector handles buffering, retries, and delivery.

## Components

| # | File | Role |
|---|------|------|
| 1 | `training_ops.py` | **Facade** — single entry point for the training team |
| 2 | `train_logger/json_logger.py` | Non-blocking structured logger (The "Producer") |
| 3 | `system_metrics/collector.py` | System metrics → JSONL → ClickHouse (The "System Probe") |
| 4 | `metrics_server.py` | Custom JSON API for live metrics (The "Exporter") |
| 5 | `checkpoint_registry/checkpoint_registry.py` | Checkpoint governance — ClickHouse-backed (The "Registry") |
| 6 | `sidecar_agent/vector.toml` | Data shipper config (The "Shipper") |
| 7 | `watchdog/watchdog.py` | Control plane service (The "Enforcer") |
| 8 | `aggregation_api/dashboard_backend.py` | Aggregation API for the frontend (The "API") |

---

## Training Instance Setup

### 1. Python Packages

```bash
pip install psutil pyyaml numpy
pip install pynvml   # optional — only needed for GPU metrics
```

### 2. Files to Copy

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

### 3. Vector Sidecar

```bash
# Install (once)
curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local

# Run (Vector MUST be running before training starts — TrainingOps checks for it)
CLICKHOUSE_HTTP_ENDPOINT="http://<DB_INSTANCE_PRIVATE_IP>:8123" \
  vector --config /path/to/vector.toml --data-dir /tmp/vector-data
```

---

## train.py Integration

`TrainingOps` is the **only thing the training team needs to interact with**. One import, one init, two methods in the loop, one cleanup call.

```python
from components import TrainingOps

# --- 1. Initialize (starts all backend services, runs preflight checks) ---
ops = TrainingOps(
    run_id="run_2026_02_13_70b_v4",
    rank=int(os.environ.get("RANK", 0)),
    log_dir="/tmp/training_logs",
    default_context={"model": "70B_v4", "cluster": "us-east-1-p4d"},
)
# Preflight: Vector running? → FATAL if not.
#            ClickHouse reachable? → warn (Vector buffers until recovery).
# Starts:   JSONLogger, SystemMetricsCollector, MetricsServer(:8000),
#           CheckpointRegistry.

# --- 2. Training Loop ---
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    # Log every N steps (writes JSONL + updates live dashboard gauges)
    if step % 10 == 0:
        ops.log_step(
            step=step,
            metrics={
                "loss": loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "tokens_per_second": tok_sec,
            },
        )

    # After saving a checkpoint (record the path — TrainingOps handles the rest)
    if step % 1000 == 0:
        path = f"checkpoints/ckpt_step_{step}.pt"
        torch.save(checkpoint_state, path)

        ops.log_checkpoint(
            step=step,
            path=path,
            s3_key=f"s3://bucket/{run_id}/ckpt_step_{step}.pt",  # or omit if local-only
            loss=loss.item(),
            tag="temporary",       # "growth" / "lora" / "release_candidate" → auto-protected
        )

# --- 3. Cleanup ---
ops.shutdown()
```

### What happens behind the scenes

| `ops.log_step(...)` | `ops.log_checkpoint(...)` |
|---|---|
| Writes JSONL → Vector → ClickHouse `logs` + `metric_points` | Writes JSONL → Vector → ClickHouse `logs` + `metric_points` + `checkpoints` |
| Updates MetricsServer gauges (live HTTP API) | Best-effort direct INSERT to `checkpoints` table (fast path) |
| | Updates checkpoint counter on MetricsServer |

### Checkpoint data flow (dual path)

```
ops.log_checkpoint()
  │
  ├─► JSONL file (durable, guaranteed)
  │     └─► Vector to_checkpoints transform
  │           └─► ClickHouse checkpoints table
  │
  └─► Direct HTTP INSERT (best-effort, immediate)
        └─► ClickHouse checkpoints table
```

If ClickHouse is temporarily unreachable, the direct insert fails silently — the JSONL → Vector path guarantees delivery with buffering and retries.

---

## Checkpoint Registry (Governance)

Backed by ClickHouse `checkpoints` table (`ReplacingMergeTree`). No SQLAlchemy, no separate database.

**Auto-protection policy:** tags `growth`, `lora`, `release_candidate` are automatically protected and cannot be deleted.

```python
from components.checkpoint_registry import CheckpointRegistry

registry = CheckpointRegistry()  # connects to ClickHouse

# Query checkpoints
registry.list_checkpoints("run_001")
registry.best_checkpoint("run_001", top_n=3)
registry.get_checkpoint("s3://bucket/ckpt_1000.pt")

# Governance
registry.can_delete("s3://bucket/ckpt_1000.pt")    # False if protected
registry.mark_for_deletion("s3://bucket/ckpt_1000.pt")  # soft-delete (appends status='deleted')
```

---

## Metrics Server Endpoints

The metrics server starts on **port 8000** by default (no config file needed).

| Endpoint | Description |
|----------|-------------|
| `GET /metrics` | Full snapshot of all current metric values (JSON) |
| `GET /query?metric=training_loss` | Single metric current value |
| `GET /history?metric=training_loss&since=<epoch>` | Time-series history |
| `GET /health` | Liveness probe |

---

## System Metrics Collected

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

## The Watchdog (Active Control)

The Watchdog polls the metrics server and pauses training on anomalies (e.g. loss divergence).

```bash
python -m components.watchdog.watchdog
```

The watchdog writes to `/tmp/training_control.flag` when triggered. Halting integration with `TrainingOps` is planned.
