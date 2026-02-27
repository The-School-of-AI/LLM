# Coreset Selection - Sharded Execution

This directory contains scripts for building coresets at scale using sharded, parallel processes.

## Sharded Execution with `shard.sh`

The `shard.sh` script (located in `coreset_engine_v5/`) allows you to run multiple parallel shards of the coreset selection pipeline.

### Prerequisites

Ensure you have your virtual environment activated and dependencies installed:

```bash
cd experiments/3_coreset_engineering/
source .venv/bin/activate
```

### 1. Fresh Start (Standard Run)

Use this command if it is your first time running the job or if you want to wipe previous results and start from scratch. It will delete the `output/checkpoints` and `output/coresets` folders before starting.

```bash
cd coreset_engine_v5
bash shard.sh \
  --num-shards 8 \
  --input-path "data/combined/bands/" \
  --total-tokens 4523096944
```

### 2. Resume (Continue Previous Job)

Use this command if your job was interrupted (e.g., manual stop or server crash). It will **keep** existing progress and skip any data that has already been processed by each shard.

```bash
cd coreset_engine_v5
bash shard.sh \
  --num-shards 8 \
  --input-path "data/combined/bands/" \
  --total-tokens 4523096944 \
  --resume
```

### Key Parameters

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input-path` | (Required) | Path to input data directory or file. |
| `--num-shards` | 4 | Number of parallel shards to launch. |
| `--total-tokens` | (Required) | Total token count of the input dataset. |
| `--resume` | false | Keeps previous outputs and resumes from last checkpoints. |
| `--stages` | "1B 3B 8B 70B" | Specific weight stages to process. |
