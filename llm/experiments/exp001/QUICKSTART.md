# OPUS Experiment — Quick Start Guide

Training with and without OPUS data selection on the recurrence model.

## Prerequisites

```bash
cd llm/experiments/exp001

# 1. Tokenizer — place files in _data/tokenizer/
ls _data/tokenizer/  # should contain tokenizer.json, etc.

# 2. Proxy dataset — download synthetic data
uv run download_synth_shard.py -o _data/synth_local_en

# 3. Dependencies — install via uv (from llm/ root)
cd ../../ && uv sync && cd experiments/exp001
```

## Quick Launch (auto-detects GPUs)

```bash
bash run.sh
```

This runs `deepspeed main.py --config config.yaml`, auto-detecting GPU count.

## Training WITH OPUS (default)

OPUS is **enabled** when `opus.candidate_multiplier > 1`. The default config uses multiplier=2.

```bash
uv run deepspeed --num_gpus=1 main.py --config config.yaml
```

**What happens at each step:**
1. Load `candidate_multiplier × micro_batch` candidate samples (the pool)
2. Sample `n_proxy_per_gpu` proxy sequences from high-quality synthetic data
3. **Scoring pass** — forward+backward over [proxy + candidates] with ghost hooks
4. **Boltzmann selection** — pick best candidates based on alignment scores
5. **Assemble training batch** — `proxy samples + OPUS-selected candidates`
6. **Training pass** — forward+backward+optimizer step on the assembled batch

**Per-step log output (every step):**
```
[step 1/20] loss=8.1234 | lr=3.00e-04 | step_time=142.3ms
  OPUS: proxy_sample=2.1ms | scoring_fwd=45.3ms | scoring_bwd=38.2ms | zero_grad=0.5ms | boltzmann=12.1ms | precond_refresh=1.2ms | total=99.8ms
  OPUS scores: alignment=0.7234 | redundancy=0.0512 | entropy=2.3410 | selector_time=11.8ms
  Batch: candidates_in=2 (seq_len=128) | proxy_in_batch=1 | selected=0 | training_batch=1 (seq_len=128) | train_tokens=128
  Timing: data_load=0.3ms | batch_asm=0.1ms | train_fwd=22.5ms | train_bwd=18.1ms | optim=1.9ms
  GPU: alloc=512MB | reserved=1024MB | peak=1280MB
```

## Training WITHOUT OPUS (baseline)

Set `opus.candidate_multiplier: 1` to disable OPUS. All candidates go directly to training — no scoring pass, no selection.

### Option A: Edit config.yaml

```yaml
opus:
  candidate_multiplier: 1  # disables OPUS
```

Then run normally:
```bash
uv run deepspeed --num_gpus=1 main.py --config config.yaml
```

### Option B: CLI override (no file changes)

```bash
uv run deepspeed --num_gpus=1 main.py --config config.yaml \
  --config.opus.candidate_multiplier=1
```

**Per-step log output (bypass mode):**
```
[step 1/20] loss=8.5678 | lr=3.00e-04 | step_time=42.1ms
  OPUS: DISABLED (bypass) | bypass_time=0.0ms
  Batch: candidates_in=1 (seq_len=128) | proxy_in_batch=0 | selected=1 | training_batch=1 (seq_len=128) | train_tokens=128
  Timing: data_load=0.2ms | batch_asm=0.0ms | train_fwd=22.0ms | train_bwd=17.5ms | optim=1.8ms
  GPU: alloc=480MB | reserved=960MB | peak=1100MB
```

## Key OPUS Config Knobs

All experiment parameters live under the `opus:` section in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `candidate_multiplier` | 2 | Pool size = micro_batch × this. **Set to 1 to disable OPUS.** |
| `n_proxy_total` | 1 | Total proxy samples across all GPUs (auto-divided by world_size) |
| `scoring_seq_len` | 512 | Token length for the ghost scoring pass |
| `train_seq_len` | 128 | Training seq_len for proxy samples (should match `data.block_sizes`) |
| `include_proxy_in_training` | true | Train on proxy samples alongside selected candidates |
| `temperature` | 0.9 | Boltzmann τ (higher = more diverse selection, lower = greedier) |
| `sketch_dim` | 512 | CountSketch projection dimension |
| `strict_shard_preconditioner` | false | Set false for ZeRO-2 (scalar fallback for partitioned state) |

## Multi-GPU (Production)

For 8 GPUs with micro_batch=4 per GPU:

**deepspeed_config.yaml:**
```yaml
train_batch_size: 32
train_micro_batch_size_per_gpu: 4
gradient_accumulation_steps: 1
zero_optimization:
  stage: 2  # enable ZeRO-2
```

**config.yaml opus section:**
```yaml
opus:
  candidate_multiplier: 4     # 4×4=16 candidate pool per GPU
  n_proxy_total: 16           # 16/8=2 proxy per GPU
  scoring_seq_len: 512
  train_seq_len: 128          # match data.block_sizes
  include_proxy_in_training: true
  strict_shard_preconditioner: false  # required for ZeRO-2
```

**Training batch per GPU:** 2 proxy + 2 OPUS-selected = 4 (= micro_batch)

```bash
uv run deepspeed --num_gpus=8 main.py --config config.yaml
```

## Understanding the Log Output

### Line 1 — Step Summary
`[step 1/20] loss=8.1234 | lr=3.00e-04 | step_time=142.3ms`

- **loss**: training loss (NTP cross-entropy + auxiliary)
- **lr**: current learning rate from scheduler
- **step_time**: total wall-clock time for the entire step

### Line 2 — OPUS Pipeline Timing
`OPUS: proxy_sample=2.1ms | scoring_fwd=45.3ms | scoring_bwd=38.2ms | ...`

- **proxy_sample**: time to draw proxy samples from the synthetic data loader
- **scoring_fwd/bwd**: forward and backward pass for ghost gradient scoring
- **zero_grad**: clearing scoring gradients before training pass
- **boltzmann**: distributed Boltzmann selection loop
- **precond_refresh**: snapshot AdamW preconditioner state (v_hat)
- **total**: end-to-end OPUS overhead

### Line 3 — Batch Composition
`Batch: candidates_in=2 | proxy_in_batch=1 | selected=0 | training_batch=1 | train_tokens=128`

- **candidates_in**: size of candidate pool loaded from data
- **proxy_in_batch**: proxy samples added to training batch
- **selected**: OPUS-selected candidates from the pool
- **training_batch**: total samples in the training forward pass
- **train_tokens**: total tokens trained on this step

### Line 4 — Training Pass Timing
`Timing: data_load=0.3ms | batch_asm=0.1ms | train_fwd=22.5ms | train_bwd=18.1ms | optim=1.9ms`

- **data_load**: moving batch to GPU
- **batch_asm**: assembling proxy + selected into training tensor
- **train_fwd/bwd**: actual training forward and backward
- **optim**: optimizer step (weight update)

### Line 5 — GPU Memory
`GPU: alloc=512MB | reserved=1024MB | peak=1280MB`

- **alloc**: currently allocated by tensors
- **reserved**: total reserved by CUDA allocator
- **peak**: maximum allocated since last reset

## Comparing OPUS vs Baseline

Run both configurations and compare the logs:

```bash
# Run with OPUS (default)
uv run deepspeed --num_gpus=1 main.py --config config.yaml 2>&1 | tee _data/train_opus.log

# Run without OPUS
uv run deepspeed --num_gpus=1 main.py --config config.yaml \
  --config.opus.candidate_multiplier=1 2>&1 | tee _data/train_baseline.log
```

Key metrics to compare:
- **Loss convergence** — does OPUS selection improve loss?
- **step_time** — OPUS adds scoring overhead (~2× per step), but may converge in fewer steps
- **GPU peak memory** — OPUS scoring pass uses extra memory for the larger combined batch
- **alignment scores** — higher alignment means OPUS is finding candidates that match the proxy gradient direction

## Running Tests

```bash
cd llm/
.venv/bin/python -m pytest experiments/exp001/tests/ -v
```

57 tests covering config, logging, selection pipeline, and batch assembly. All run on CPU with no GPU required.

## Profiler Output

When `train.profile_steps` is set, detailed kernel-level profiling is written to:
- `_data/profiler/profile.jsonl` — per-step timing data
- `_data/profiler/profile_report.txt` — human-readable summary
- `_data/profiler/pipeline_report.txt` — pipeline stage breakdown
