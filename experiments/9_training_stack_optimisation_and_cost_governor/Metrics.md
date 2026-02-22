# Training Metrics Guide

All metrics are written to a single file: `{output_dir}/metrics.jsonl`

The output directory depends on which config you run:

| Config | Metrics file location |
|---|---|
| `config.yaml` | `./checkpoints/metrics.jsonl` |
| `config_fix_oom.yaml` | `./checkpoints_oom_fixed/metrics.jsonl` |
| `config_4k_throughput.yaml` | `./checkpoints_4k/metrics.jsonl` |
| `config_8k_throughput.yaml` | `./checkpoints_8k/metrics.jsonl` |
| `config_reversible.yaml` | `./checkpoints_reversible/metrics.jsonl` |

**If you're not sure where the file is:**
```bash
find . -name "metrics.jsonl"
```

---

## How to Run Training

```bash
# Single GPU
deepspeed main.py

# 4 GPUs
deepspeed --num_gpus=4 main.py

# With a specific config
deepspeed --num_gpus=4 main.py --config config_fix_oom.yaml
```

The metrics file is created automatically once the first optimizer step completes.

---

## Metric Types

Every line in the JSONL has an `event` field. There are 4 event types:

### 1. `train_step` — every optimizer step

| Field | What it means |
|---|---|
| `global_step` | Current step number |
| `epoch` | Current epoch |
| `loss` | Total combined loss |
| `loss_ntp` | Next token prediction loss (t+1) |
| `loss_mtp` | Multi-token prediction loss (t+2) |
| `loss_aux` | Combined router loss (MoE only) |
| `loss_null_router` | NULL expert router loss — `null` on dense 1B model, populated on 70B MoE when model returns split aux_loss |
| `loss_moe_router` | MoE router loss — same as above |
| `tokens_per_sec` | Training throughput in tokens/second |
| `batches_per_sec` | Training throughput in batches/second |
| `tokens` | Tokens processed in this step |
| `total_tokens_processed` | Running total tokens across all epochs |
| `step_time_s` | Wall time for this optimizer step (seconds) |
| `gpu_util_pct` | GPU utilization % (requires `enable_system_metrics: true`) |
| `gpu_idle_pct` | GPU idle % — `100 - gpu_util_pct` |
| `gpu_mem_gb` | GPU memory used in GB |
| `cpu_util_pct` | CPU utilization % |
| `cpu_idle_pct` | CPU idle % |
| `lr` | Current learning rate |
| `timestamp` | Unix timestamp |

### 2. `evaluation` — after every validation and test run

| Field | What it means |
|---|---|
| `phase` | `"Validation"` or `"Test"` |
| `avg_loss` | Average loss on the eval set |
| `avg_perplexity` | Average perplexity |
| `global_step` | Step it was run at |
| `timestamp` | Unix timestamp |

### 3. `checkpoint_saved` — every time a checkpoint is saved

| Field | What it means |
|---|---|
| `tag` | Checkpoint name e.g. `epoch0_step200` |
| `duration_s` | How long the save took in seconds |
| `global_step` | Step it was saved at |
| `timestamp` | Unix timestamp |

### 4. `generated_sample` — every N steps (if configured)

| Field | What it means |
|---|---|
| `prompt` | Input prompt used |
| `generated_text` | Model output |
| `epoch` | Current epoch |
| `global_step` | Step it was generated at |
| `timestamp` | Unix timestamp |

To enable generated samples, set in your config:
```yaml
generation:
  generation_interval: 500  # generate every 500 steps
  generation_prompt: "The history of artificial intelligence begins with"
```

---

## How to Watch Metrics Live

### Option 1 — Pretty printed (recommended)

Run this in a second terminal while training:

```bash
tail -f checkpoints_oom_fixed/metrics.jsonl | python3 -c "
import sys, json
for line in sys.stdin:
    d = json.loads(line)
    if d.get('event') == 'train_step':
        print(f\"step={d['global_step']:4d} | loss={d['loss']:.3f} | ntp={d['loss_ntp']:.3f} | tok/s={d['tokens_per_sec']:6.0f} | gpu={d['gpu_util_pct']}% | idle={d['gpu_idle_pct']}% | mem={d['gpu_mem_gb']:.1f}GB | cpu_idle={d['cpu_idle_pct']}% | total_tok={d['total_tokens_processed']:,}\")
    elif d.get('event') == 'evaluation':
        print(f\">>> EVAL [{d['phase']}] loss={d['avg_loss']:.3f} ppl={d['avg_perplexity']:.2f}\")
    elif d.get('event') == 'checkpoint_saved':
        print(f\">>> CHECKPOINT saved: {d['tag']} in {d['duration_s']:.1f}s\")
"
```

### Option 2 — Raw JSON (for debugging)

```bash
tail -f checkpoints_oom_fixed/metrics.jsonl
```

### Option 3 — Watch all 4 GPUs

```bash
watch -n 2 nvidia-smi
```

Shows per-GPU memory, utilization, temperature and power draw updated every 2 seconds.

---

## How to Enable GPU/CPU Metrics

By default `gpu_util_pct`, `gpu_idle_pct`, `cpu_util_pct`, `cpu_idle_pct` are `null`.

To enable them, set in your config yaml:

```yaml
training:
  enable_system_metrics: true
```

Requires `pynvml` installed:
```bash
pip install pynvml
```

---

## How to Query Past Metrics

After training, load the JSONL and filter by event type:

```python
import json

with open("checkpoints_oom_fixed/metrics.jsonl") as f:
    records = [json.loads(line) for line in f]

# Training steps only
train_steps = [r for r in records if r["event"] == "train_step"]

# Validation results only
evals = [r for r in records if r["event"] == "evaluation" and r["phase"] == "Validation"]

# Print loss over time
for r in train_steps:
    print(f"step {r['global_step']}: loss={r['loss']:.4f}")

# Print all validation results
for r in evals:
    print(f"step {r['global_step']}: val_loss={r['avg_loss']:.4f} ppl={r['avg_perplexity']:.2f}")
```

---

## What the Numbers Mean

| Metric | Healthy | Warning |
|---|---|---|
| `gpu_util_pct` | 95-100% | Below 80% = GPU is waiting (data bottleneck) |
| `gpu_idle_pct` | 0-5% | Above 20% = wasted compute |
| `gpu_mem_gb` | Stable | Growing every step = memory leak |
| `cpu_idle_pct` | Above 70% | Below 30% = CPU is the bottleneck |
| `loss` (early) | Spiky/high | Fine during warmup |
| `loss` (later) | Smoothly decreasing | Flat or increasing = problem |
| `loss_null_router` | `null` on 1B dense | Populated on 70B MoE only |
| `loss_moe_router` | `null` on 1B dense | Populated on 70B MoE only |
