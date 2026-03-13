# Observability Stack Setup

Two instances: **DB instance** (ClickHouse) and **Training instance** (TrainingOps + Vector sidecar).

## Architecture

```
Training Instance                              DB Instance
┌────────────────────────────────┐             ┌──────────────────────┐
│                                │             │ ClickHouse :8443     │
│  TrainingOps (single object)   │             │   logs               │
│    ├─ JSONLogger ──────┐       │             │   metric_points      │
│    ├─ SystemMetrics ───┤ .jsonl│             │   checkpoints        │
│    ├─ MetricsServer    │       │             │   events, runs       │
│    └─ CheckpointRegistry       │             │                      │
│                        │       │             │ Grafana :3000        │
│  Vector sidecar ───────┼───────┼─ HTTPS ──▶  │   Training Overview  │
│    ├─ to_raw_logs      │       │  :8443      │   (auto-provisioned) │
│    ├─ to_metric_points │       │  (TLS+auth) │                      │
│    └─ to_checkpoints   │       │             │  Users:              │
│                        │       │             │   p12_writer (write) │
│  /tmp/training_logs/*.jsonl    │             │   p12_reader (read)  │
└────────────────────────────────┘             └──────────────────────┘
```

---

## DB Instance Setup

### 1) Generate Auth Credentials + TLS Certs

Run the setup script **once** on the DB instance. It generates TLS certificates, ClickHouse user configs, and `.env` files for the training and dashboard teams.

```bash
cd experiments/12_training_operations/components/clickhouse
bash setup-auth.sh
```

You will be prompted for two passwords:
- **p12_writer** — used by Vector and CheckpointRegistry on the training instance
- **p12_reader** — used by dashboards and query tools (read-only)

The script generates:
| File | Purpose | Committed to git? |
|------|---------|-------------------|
| `tls/server.crt`, `tls/server.key`, `tls/ca.crt` | TLS certificates | No |
| `users.d/p12-users.xml` | ClickHouse users with password hashes | No |
| `training-instance.env` | Env file for the training team | No |
| `dashboard.env` | Env file for the dashboard team | No |

### 2) Start ClickHouse

```bash
sudo docker compose up -d
```

This auto-applies schema from `initdb.d/` on first start (creates `training_observability` database and all tables).

### 3) Verify

```bash
# Local admin access (no auth needed — localhost only)
sudo docker exec p12-clickhouse clickhouse-client --query "SHOW TABLES FROM training_observability"
# Expected: checkpoints, events, logs, metric_arrays, metric_points, runs

# HTTPS endpoint (requires auth)
curl -sk https://localhost:8443/ping
# Expected: Ok.

# Authenticated query
curl -sk "https://localhost:8443/?user=p12_reader&password=<password>&query=SELECT+1"
# Expected: 1
```

### 4) AWS Security Group

Open **TCP 8443** (HTTPS) inbound from the training instance IP/security group. Port 8123 (HTTP) should **not** be open publicly — it's localhost-only for admin access.

### 5) Wipe and Restart (if needed)

```bash
sudo docker compose down -v   # -v removes volumes (all data)
sudo docker compose up -d     # fresh start, schema re-applied
```

---

## Training Instance Setup

### 1) Install Python packages

```bash
pip install psutil pyyaml numpy pynvml
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

### 4) Copy credentials from the DB instance

```bash
# CA certificate (for TLS verification)
scp db-instance:/path/to/clickhouse/tls/ca.crt /etc/p12/ca.crt

# Environment file
scp db-instance:/path/to/clickhouse/training-instance.env ~/.p12.env
```

Edit `~/.p12.env` and replace `<DB_INSTANCE_IP>` with the actual private IP of the DB instance.

### 5) Set environment variables

```bash
export $(cat ~/.p12.env | grep -v '^#' | xargs)
```

Required variables:

| Variable | Example | Used by |
|----------|---------|---------|
| `CLICKHOUSE_HTTPS_ENDPOINT` | `https://10.0.1.5:8443` | Vector, TrainingOps, CheckpointRegistry |
| `CLICKHOUSE_USER` | `p12_writer` | Vector, TrainingOps, CheckpointRegistry |
| `CLICKHOUSE_PASSWORD` | `<password>` | Vector, TrainingOps, CheckpointRegistry |
| `CLICKHOUSE_CA_CERT` | `/etc/p12/ca.crt` | Vector, TrainingOps, CheckpointRegistry |

### 6) Verify connectivity

```bash
curl -sk --cacert /etc/p12/ca.crt \
  "https://<DB_INSTANCE_IP>:8443/?user=p12_writer&password=<password>&query=SELECT+1"
# Must return: 1
```

### 7) Create Vector data directory

```bash
sudo mkdir -p /var/lib/vector && sudo chown $(whoami) /var/lib/vector
```

### 8) Run Vector

```bash
# Env vars must be set first (step 5). Vector reads them from the environment.
vector --config /path/to/vector.toml --data-dir /tmp/vector-data
```

### 9) Integrate into train.py

See `train_sample/README.md` for the full integration guide. The short version:

```python
from components import TrainingOps

# All credentials come from environment variables — no passwords in code.
ops = TrainingOps(
    run_id="run_2026_02_13_70b_v4",
    rank=int(os.environ.get("RANK", 0)),
    default_context={"model": "70B_v4", "cluster": "us-east-1-p4d"},
)

for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    if step % 10 == 0:
        ops.log_step(step=step, metrics={"loss": loss.item(), "lr": lr})

ops.shutdown()
```

---

## Grafana Dashboard

Grafana is included in the docker-compose and auto-provisions with a pre-built **Training Overview** dashboard.

### 1) Start (included in docker-compose)

```bash
# Set Grafana admin password (optional, default: p12training)
export GRAFANA_ADMIN_PASSWORD=your_secure_password

# Start both ClickHouse and Grafana
sudo docker compose up -d
```

Grafana starts on **port 3000** after ClickHouse passes its healthcheck.

### 2) Access

Open `http://<DB_INSTANCE_IP>:3000` in your browser.
- Default login: `admin` / `p12training` (or your `GRAFANA_ADMIN_PASSWORD`)
- The **Training Overview** dashboard loads as the home dashboard automatically

### 3) What's in the dashboard

The pre-built dashboard has **20 panels** across these sections:

| Row | Panels |
|-----|--------|
| **Training curves** | Loss (multi-rank), Learning Rate |
| **Throughput** | Tokens/sec, Step Time (ms) |
| **GPU health** | Utilization %, Memory Used (GB), Temperature, Power (W) |
| **System** | CPU/RAM %, GPU Alloc/Peak Reserved, Gradient Norm |
| **OPUS metrics** | Alignment, Redundancy, Entropy, Phase Timings (stacked bar) |
| **Governance** | Checkpoints table (step, loss, tag, protected, size), Events table (severity-colored) |
| **Stats bar** | Current Loss, Step, Tok/s, Checkpoints count, Log entries, Max GPU Temp |
| **Extra** | Loss vs Step scatter |

All panels are parameterized by **Run ID** and **Rank** dropdowns at the top.

### 4) ClickHouse datasource

The datasource auto-provisions via `grafana/provisioning/datasources/clickhouse.yml`. It connects to ClickHouse over the Docker internal network (`clickhouse:8123`, HTTP, no TLS needed for intra-compose).

By default it uses the `default` user (localhost, no password). To use `p12_reader` instead:

```bash
export CLICKHOUSE_GRAFANA_USER=p12_reader
export CLICKHOUSE_GRAFANA_PASSWORD=<reader_password>
sudo docker compose up -d
```

### 5) AWS Security Group

Open **TCP 3000** (Grafana UI) inbound from your IP/VPN. Or use SSH tunnel:

```bash
ssh -L 3000:localhost:3000 user@db-instance
# Then open http://localhost:3000
```

### 6) Custom dashboards

The provisioned dashboard is editable. You can also create new dashboards using the ClickHouse datasource — all 6 tables are available:

| Table | Best for |
|-------|----------|
| `metric_points` | Time-series charts (loss, LR, GPU, throughput) |
| `checkpoints` | Governance tables (use `FINAL` for deduplication) |
| `events` | Event feeds, alerts |
| `logs` | Raw audit trail, debugging |
| `metric_arrays` | Histograms, distributions |
| `runs` | Run metadata |

---

## Dashboard Team Setup (CLI / External Tools)

The dashboard team uses the **`p12_reader`** user (SELECT-only access).

### 1) Get credentials

```bash
scp db-instance:/path/to/clickhouse/dashboard.env ~/.p12-dashboard.env
scp db-instance:/path/to/clickhouse/tls/ca.crt /etc/p12/ca.crt
```

Edit the `.env` file and replace `<DB_INSTANCE_IP>` with the actual IP.

### 2) Connect

Any ClickHouse client, BI tool, or HTTP client can connect:

```bash
# CLI example
curl -sk --cacert /etc/p12/ca.crt \
  "https://<DB_INSTANCE_IP>:8443/?user=p12_reader&password=<password>" \
  -d "SELECT metric, count(), avg(value) FROM training_observability.metric_points GROUP BY metric"
```

### 3) Example queries

```sql
-- Loss curve for a run
SELECT step, value FROM training_observability.metric_points
WHERE run_id = 'my_run' AND metric = 'loss' ORDER BY step;

-- Latest checkpoints
SELECT run_id, step, s3_key, loss, tag, is_protected, status
FROM training_observability.checkpoints FINAL
WHERE run_id = 'my_run' ORDER BY step DESC;

-- System metrics (GPU utilization)
SELECT event_time, value FROM training_observability.metric_points
WHERE metric LIKE 'sys.gpu%util%' AND run_id = 'my_run' ORDER BY event_time;
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

## Verify Ingestion

On the DB instance (local admin access):

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

---

## Redeploying to a New Instance

The entire DB stack is self-contained in the `clickhouse/` directory. To redeploy:

### 1) On the new instance

```bash
# Install Docker
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2

# Copy the clickhouse/ directory
scp -r clickhouse/ new-instance:~/clickhouse/

# Generate fresh credentials (or copy existing ones)
cd ~/clickhouse
bash setup-auth.sh

# Start
sudo docker compose up -d
```

### 2) If migrating data

```bash
# On the old instance — export data
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.logs FORMAT Native" > logs.native
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.metric_points FORMAT Native" > metric_points.native
sudo docker exec p12-clickhouse clickhouse-client \
  --query "SELECT * FROM training_observability.checkpoints FORMAT Native" > checkpoints.native

# Copy to new instance, then import
cat logs.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.logs FORMAT Native"
cat metric_points.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.metric_points FORMAT Native"
cat checkpoints.native | sudo docker exec -i p12-clickhouse clickhouse-client \
  --query "INSERT INTO training_observability.checkpoints FORMAT Native"
```

### 3) Update training instances

- Update `CLICKHOUSE_HTTPS_ENDPOINT` in `~/.p12.env` with the new IP
- Copy the new `ca.crt` if you regenerated certs
- Restart Vector

---

## Instance Sizing Recommendation

ClickHouse is extremely efficient. For training observability (not serving production queries), the requirements are modest.

### Workload Profile

- **Write rate:** ~10-50 rows/second (training steps + system metrics every 5s)
- **Storage:** ~1-5 GB/month per active training run (compressed by MergeTree)
- **Query pattern:** Occasional dashboard reads, not high-concurrency OLTP

### Recommended Instance Sizes

| Scenario | AWS Instance | vCPUs | RAM | Storage | Monthly Cost |
|----------|-------------|-------|-----|---------|-------------|
| **1-2 concurrent training runs** | `t3.small` | 2 | 2 GB | 20 GB gp3 | ~$15 |
| **3-5 concurrent runs + dashboard** | `t3.medium` | 2 | 4 GB | 50 GB gp3 | ~$30 |
| **Heavy use (10+ runs, long retention)** | `t3.large` | 2 | 8 GB | 100 GB gp3 | ~$60 |

**Key insight:** ClickHouse compresses data 5-10x with MergeTree. A month of 70B training metrics (logging every 10 steps) is typically under 1 GB compressed. A `t3.small` is more than sufficient for most setups.

### Storage

- Use **gp3** EBS volumes (not gp2) — gp3 has consistent baseline IOPS regardless of size
- 20 GB is plenty to start; ClickHouse data is highly compressible
- Docker volume is at `/var/lib/docker/volumes/` — monitor with `df -h`

### When to Scale Up

- **RAM:** If ClickHouse starts swapping (check `sys.mem_percent` in your own metrics)
- **Storage:** If disk usage exceeds 80%
- **CPU:** Only if running heavy analytical queries concurrently with writes
