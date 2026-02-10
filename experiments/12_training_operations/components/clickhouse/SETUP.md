# ClickHouse Setup (P12 Training Operations)

This guide is for validating that:

- ClickHouse is running
- Training logs are being produced as JSONL
- (Later) Vector ships logs into ClickHouse
- ClickHouse can be queried locally and from outside

## 1) Start ClickHouse (bare metal)

### Install (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
sudo systemctl enable --now clickhouse-server
```

### Verify

```bash
systemctl is-active clickhouse-server
clickhouse-client --query "SELECT version()"
```

## 2) Apply schema

From repo root:

```bash
bash experiments/12_training_operations/components/clickhouse/apply_schema.sh
```

Verify:

```bash
clickhouse-client --query "SHOW TABLES FROM training_observability"
```

## 3) Produce JSONL logs (no sidecar yet)

Example:

```python
from components.json_logger import JSONLogger

logger = JSONLogger(
    base_dir="/tmp/training_logs",
    run_id="run_dev_001",
    rank=0,
    default_context={"model_size": "1B", "source": "dev"},
)

for step in range(3):
    logger.log_step(
        step=step,
        metrics={"loss/train": 2.3 - step * 0.1, "throughput/tokens_per_sec": 1000 + step},
        context={"routing_dist": [0.1, 0.2, 0.7]},
    )

logger.close()
```

You should see files like:

- `/tmp/training_logs/run_dev_001_rank_0.jsonl`

## 4) Confirm ClickHouse is queryable (local)

At this stage ClickHouse will not receive anything until Vector is running.

You can still validate ClickHouse is healthy:

```bash
clickhouse-client --query "SELECT count() FROM training_observability.logs"
clickhouse-client --query "SELECT count() FROM training_observability.metric_points"
```

## 5) Query from outside (safe dev method)

ClickHouse is configured to listen on localhost by default.

Use an SSH tunnel from your laptop:

```bash
ssh -L 8123:127.0.0.1:8123 -L 9000:127.0.0.1:9000 ubuntu@<server_ip>
```

Then on your laptop:

- HTTP endpoint: `http://127.0.0.1:8123`
- Native port: `127.0.0.1:9000`

Example HTTP query:

```bash
curl 'http://127.0.0.1:8123/?query=SELECT%20version()'
```

## 6) Next step (sidecar)

Run Vector with:

```bash
vector --config experiments/12_training_operations/components/sidecar_agent/vector.toml
```

Then verify ingestion:

```bash
clickhouse-client --query "SELECT run_id, rank, step, host FROM training_observability.logs ORDER BY _ingest_time DESC LIMIT 10"
clickhouse-client --query "SELECT metric, value, step, host, rank FROM training_observability.metric_points ORDER BY event_time DESC LIMIT 50"
```
