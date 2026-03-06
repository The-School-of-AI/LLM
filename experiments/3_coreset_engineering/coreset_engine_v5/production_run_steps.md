# Coreset Engine Production Run Playbook

This document outlines the high-level steps executed by the [commands.sh](file:///Users/pankajkumar/Documents/git/TSAI/ERA4/final-capstone/P3/LLM/experiments/3_coreset_engineering/coreset_engine_v5/scripts/commands.sh) script to run the coreset production pipeline.

If the pipeline has to be ran in a decoupled fashion for multiple stages then always make sure that checkpoints and used indices are available in the NVMe or local disk from where the process has to be run and make sure variable RESUME=true is set before calling commands.sh

## 1. Preparation & Setup

- **NVME Setup**: Use scripts/setup_nvme.sh to create the /mnt/nvme or run the commands manually
- **Copy commands.sh**: cd to directory /mnt/nvme/, copy commands.sh to /mnt/nvme/ and run it from this directory itself
- **Perform the setup**: Comment Sections 2, 5, 6, 7, 7b, 8
- **Environment Configuration**: Export basic variables like `S3_BUCKET`, `NUM_SHARDS`, and `STAGES` are initialized.
- **Dependency Installation**: Ensures `uv` (Python manager) and AWS CLI are installed.
- **Repository Sync**: Pulls the latest code from the target branch.
- **ENABLE_NVME**: Always enable this from the parameters to be able to run and redirect all the IO to NVMe

## 2. Infrastructure & Environment

- **Virtual Environment**: Creates/syncs a Python environment via `uv sync`.
- **[OPTIONAL] Infrastructure Validation**: Confirms EBS/NVMe mounts and permissions via `validate_infra.sh`.
- **Storage Redirection**: When `ENABLE_NVME=true`, coresets, manifests, and checkpoints are written to NVMe (e.g. `/mnt/nvme/output/coresets`, `/mnt/nvme/output/manifests`, `/mnt/nvme/output/checkpoints`). A background S3 sync persists these to S3.

## 3. Pipeline Execution

- **[OPTIIONAL] Background Monitoring**: Tracks CPU, memory, and disk usage during the run.
- **Call commands.sh**: Comment Sections 1, 2, 3, 5, 6
- **Pipeline Launch**: Starts the [shard.sh](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/shard.sh) engine in the background using `nohup` for persistence. We trigger [commands.sh](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/scripts/commands.sh) to run the pipeline inturn commands.sh calls shard.sh.

## 4. Automated Post-Processing & Sync

- **S3 destination**: When NVMe is enabled, outputs are synced to S3. Default path is `s3://${S3_BUCKET}/t3-coreset_outputs/`. Override with `export S3_SYNC_DEST=s3://your-bucket/your-prefix` (trailing slash is added automatically to avoid double slashes). Subpaths: `coresets/`, `manifests/`, `checkpoints/` (e.g. `s3://bucket/t3-coreset_outputs/checkpoints/`).
- **Interval syncing**: Periodic background sync (default every 10 minutes; set `S3_SYNC_INTERVAL` in seconds to change).
- **Final consolidation**: Triggered automatically upon pipeline completion:
  - **Manifest Merging**: Aggregates shards into a single `manifest.json`.
  - **Report Merging**: Consolidated into `ablation_validation_report.md`.
  - **Validation**: Runs [validate_coresets_outputs_v2.py](https://github.com/The-School-of-AI/LLM/blob/p3/feat/stage-wise-coreset-selection_v2/experiments/3_coreset_engineering/coreset_engine_v5/tools/validate_coresets_outputs_v2.py) to generate verification reports.
- **Final Persistence**: Comprehensive final sync of all merged files and reports to S3.

## 5. Resuming from Checkpoints

The system supports restarting from the last saved state if a run is interrupted:

- **Resume Capability**: Enabled via the `RESUME` environment variable.
- **Persistence**: Relies on checkpoint files and the `used_chunks.db` (SQL database) being present.
- **Used Chunks DB**: The SQL database tracks processed data to prevent duplicates. It is synced from S3 to ensure consistency across resumed runs.
- **Workflow**: The engine detects existing checkpoints and skips already processed batches.

## Quick Start

### Parameters to be passed always before running the pipeline

Keep these parameters as-is to run the workload (please use them as-is) -

- export STAGES="1B"
  - export STAGES="1B 3B 8B 70B"
- export BATCH_SIZE=500000
- export ENABLE_NVME=true
- export TOTAL_TOKENS="2130633645405"
- export CHECKPOINT_EVERY_N_BATCHES=25
- export SKIP_EBS_VALIDATION=true
- export NUM_SHARDS=20
- export S3_INPUT_PATH="/mnt/nvme/data/curriculum_pyspark_output"
  - This parameter could be used interchangeably with Local and S3 input paths parameters
  - **Single-source example (books):** `s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/source=books/` (set `TOTAL_TOKENS` to match, e.g. books ≈ 2.8B from T3StatsFromT2.txt)
- export BATCH_PREFETCH_MODE=auto
- export RESUME=false
  - export RESUME=true (only when process needs to be restarted)

To launch a **fresh** production pipeline:

```bash
export S3_BUCKET="your-bucket-name"
./commands.sh
```

To **resume** a pipeline from checkpoints:

```bash
export S3_BUCKET="your-bucket-name"
export RESUME=true
./commands.sh
```

To check progress:

- **Pipeline logs**: `tail -f shard_run.log`
- **S3 sync & Post-processing logs**: `tail -f s3_sync.log`
