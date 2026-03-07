# LLM Training Logging Updates

This repository includes a production-oriented training log pipeline update for DeepSpeed-based runs.

## What Changed

### 1. Training launcher logging (`llm/run.sh`)

- Writes a full combined stream to:
  - `llm/_data/results/run/training_combined.log`
- Writes filtered rank-sensitive issues to:
  - `llm/_data/results/run/rank_issues.log`
- Adds timestamp prefixing for streamed output:
  - uses `ts` (from `moreutils`) when available
  - falls back to Python-based UTC timestamping
- Supports optional per-process NCCL files:
  - enabled with `NCCL_PER_PROCESS_LOGS=1`
  - outputs `nccl.%h.%p.log` in the run log directory
- Extracts issue lines from per-rank temporary logs and prefixes with rank information.

### 2. Rank startup metadata (`llm/main.py`)

Each process prints one startup line with:

- `run_id`
- `rank`
- `local_rank`
- `world_size`
- `pid`
- `host`

Timestamp format is timezone-aware UTC (`...Z`).

### 3. Logrotate template (`infra/logging/logrotate.training.conf`)

- Machine-agnostic template using `__RUN_LOG_DIR__`
- Includes:
  - `training_combined.log`
  - `rank_issues.log`
  - `nccl.*.log`
- Policy:
  - `size 200M`
  - `rotate 10`
  - `daily`
  - `compress` + `delaycompress`
  - `copytruncate`
  - `sharedscripts`

## Run-Time Logrotate Rendering

`llm/run.sh` renders a machine-specific config on each run to:

- `llm/_data/results/run/logrotate.training.rendered.conf`

Optional behavior:

- `AUTO_INSTALL_LOGROTATE=1` installs rendered config to `/etc/logrotate.d/llm-training` (or `LOGROTATE_TARGET`)
- `VERIFY_LOGROTATE=1` runs `logrotate -d` validation

Defaults are safe for training:

- no auto-install (`AUTO_INSTALL_LOGROTATE=0`)
- no validation (`VERIFY_LOGROTATE=0`)

## Usage

Standard:

```bash
./llm/run.sh
```

Enable NCCL per-process files:

```bash
NCCL_PER_PROCESS_LOGS=1 ./llm/run.sh
```

Auto-install rendered logrotate config:

```bash
AUTO_INSTALL_LOGROTATE=1 ./llm/run.sh
```

Render + validate logrotate config:

```bash
VERIFY_LOGROTATE=1 ./llm/run.sh
```

## Environment Knobs

- `RANK_ISSUE_REGEX`
- `NCCL_PER_PROCESS_LOGS`
- `LOGROTATE_TEMPLATE`
- `AUTO_INSTALL_LOGROTATE`
- `LOGROTATE_TARGET`
- `VERIFY_LOGROTATE`
- `NCCL_DEBUG`
- `NCCL_ASYNC_ERROR_HANDLING`
- `PYTHONUNBUFFERED`

## Important Note

`llm/run.sh` currently points to `llm/configs/config.yaml` by default.  
If you use `llm/configs/config-mini.yaml`, either:

- update `CFG` in `llm/run.sh`, or
- create/point a `config.yaml` for your run.
