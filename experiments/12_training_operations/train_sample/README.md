# Training Scripts — P12 Observability Integration

These training scripts are instrumented with **P12 TrainingOps** for automatic observability. The training team does not need to manage any backend services manually — `TrainingOps` handles everything.

## What You Get (Automatically)

Once `TrainingOps` is initialized, every training run automatically gets:

- **Structured JSONL logs** shipped to ClickHouse via Vector (loss, LR, throughput, etc.)
- **System metrics** (CPU, RAM, GPU, disk, network) collected every 5 seconds
- **Live metrics HTTP API** on port 8000 (for dashboards and watchdog)
- **Checkpoint registration** in ClickHouse with governance (auto-protection for growth/lora tags)

## Quick Start

### 1. Set Environment Variables

Before launching training, source the env file provided by the infra team:

```bash
export $(cat ~/.p12.env | grep -v '^#' | xargs)
```

Or set them manually:

```bash
export CLICKHOUSE_ENDPOINT="http://<DB_INSTANCE_IP>:8123"
export CLICKHOUSE_USER="p12_writer"
export CLICKHOUSE_PASSWORD="<password>"
```

If you run ClickHouse over HTTPS, use:

```bash
export CLICKHOUSE_ENDPOINT="https://<DB_INSTANCE_IP>:8443"
export CLICKHOUSE_CA_CERT="/etc/p12/ca.crt"
```

### 2. Ensure Vector Is Running

Vector **must** be running before training starts. `TrainingOps` will check and exit if it's not.

```bash
CLICKHOUSE_ENDPOINT="http://<DB_INSTANCE_IP>:8123" \
CLICKHOUSE_USER="p12_writer" \
CLICKHOUSE_PASSWORD="<password>" \
  vector --config /path/to/vector.toml
```

### 3. Run Training

```bash
python train_recurrence_70b.py
```

For a fast synthetic pipeline check (no tokenizer/dataset), run:

```bash
python train_dry_run.py
```

That's it. All observability is handled automatically.

## Integration Guide (For New Scripts)

Adding P12 observability to a new training script requires **3 lines of code**:

```python
from components import TrainingOps

# 1. Initialize (before the training loop)
ops = TrainingOps(
    run_id="my_run_001",
    rank=int(os.environ.get("RANK", 0)),
    default_context={"model": "my_model", "experiment": "baseline"},
)

# 2. In the training loop — log metrics
for step, batch in enumerate(dataloader):
    loss = train_step(batch)

    ops.log_step(step=step, metrics={
        "loss": loss.item(),
        "lr": optimizer.param_groups[0]["lr"],
        "tokens_per_second": tok_sec,
    })

# 3. Cleanup (after the training loop)
ops.shutdown()
```

### Rich Metrics + Events (recommended)

`TrainingOps` now supports explicit APIs for structured events and array metrics:

```python
# Typed event -> training_observability.events
ops.log_event(
    step=step,
    event_type="stage_transition",
    message="entered_train_phase",
    payload={"stage": "train"},
)

# Array metric -> training_observability.metric_arrays
ops.log_metric_array(
    step=step,
    metric="moe/routing_dist_mean",
    keys=["expert_0", "expert_1", "expert_2"],
    values=[0.41, 0.33, 0.26],
    unit="ratio",
)
```

Suggested scalar metric names (go to `metric_points` via `ops.log_step`):

- `loss/train`
- `loss/train_t_plus_1`
- `loss/train_t_plus_2`
- `loss/val`
- `loss/router_null`
- `loss/router_moe`
- `throughput/tokens_per_sec`
- `throughput/batches_per_sec`
- `tokens/processed_total`
- `router/null_ratio`
- `cpu/idle_percent`

Suggested array metric names (go to `metric_arrays` via `ops.log_metric_array`):

- `moe/routing_dist_mean`
- `moe/favorite_tokens_topk`
- `moe/fourier_bucket_energy`
- `gpu/utilization` (per device)

Suggested event types (go to `events` via `ops.log_event`):

- `checkpoint_saved`
- `checkpoint_uploaded`
- `checkpoint_benchmarked`
- `stage_transition`
- `sample_generated`

Table routing summary:

- `ops.log_step(...)` -> `logs`, `metric_points`
- `ops.log_metric_array(...)` -> `metric_arrays` (+ raw audit copy in `logs`)
- `ops.log_event(...)` -> `events` (+ raw audit copy in `logs`)
- `ops.log_checkpoint(...)` -> `checkpoints` (+ `checkpoint_saved` in `events` + audit copy in `logs`)

### Checkpointing

The existing `save_checkpoint()` in `training.py` now accepts an optional `ops` parameter:

```python
from training import save_checkpoint

save_checkpoint(
    model, optimizer, lr_scheduler,
    step=step, loss=loss.item(),
    embedding_type="kronecker",
    save_dir="checkpoints",
    ops=ops,                                    # ← pass TrainingOps
    s3_key="s3://bucket/ckpt_step_1000.pt",     # ← optional S3 path
    tag="temporary",                            # ← governance tag
)
```

If `ops` is `None` (or not passed), checkpointing works exactly as before — no observability, no breaking change.

### Governance Tags

| Tag | Auto-Protected | Meaning |
|-----|---------------|---------|
| `temporary` | No | Regular checkpoint, can be cleaned up |
| `growth` | **Yes** | Growth-phase checkpoint, cannot be deleted |
| `lora` | **Yes** | LoRA adapter checkpoint, cannot be deleted |
| `release_candidate` | **Yes** | Release candidate, cannot be deleted |

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `CLICKHOUSE_ENDPOINT` | Yes | ClickHouse endpoint, e.g. `http://10.0.1.5:8123` or `https://10.0.1.5:8443` |
| `CLICKHOUSE_USER` | Yes | ClickHouse username (`p12_writer`) |
| `CLICKHOUSE_PASSWORD` | Yes | ClickHouse password |
| `CLICKHOUSE_CA_CERT` | HTTPS only | Path to CA certificate for TLS verification |
| `CLICKHOUSE_HTTPS_ENDPOINT` | Legacy fallback | Older variable still accepted by `TrainingOps` |
| `CLICKHOUSE_HTTP_ENDPOINT` | Legacy fallback | Older variable still accepted by `TrainingOps` |
| `RANK` | No | Distributed training rank (default: 0) |

## Files

| File | Description |
|------|-------------|
| `training.py` | Training utilities (optimizer, LR schedule, checkpoint save/load) |
| `train_recurrence_70b.py` | 70B recurrence model training — **integrated with TrainingOps** |
| `train_recurrence_1b.py` | 1B recurrence model training |
| `data.py`, `data_utils.py` | Dataset loading utilities |
| `model_gated_multitoken.py` | Gated multi-token prediction model |
| `recurrence_model_70b.py` | 70B recurrence model definition |
| `recurrence_model_1b.py` | 1B recurrence model definition |
| `reversible_ops_midpoint.py` | Reversible operations for memory efficiency |
