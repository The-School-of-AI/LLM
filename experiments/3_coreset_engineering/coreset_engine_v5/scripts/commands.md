# EC2 Coreset Pipeline Commands

This document covers how to run the coreset pipeline
on an EC2 instance, using either the **automated script**
(`commands.sh`) or **manual steps**.

---

## Quick Start (Automated)

The `commands.sh` script automates the full pipeline in 8 steps:

1. System Setup & Prerequisites
2. AWS Authentication Check
3. Repository Setup
4. Dependency Sync (via UV)
5. Infrastructure Validation (`validate_infra.sh`)
6. Start Monitoring (`monitor.sh`)
7. Launch Pipeline (`shard.sh`)
8. Post-Run Validation & Reports

### Prerequisites

- **EMR Job (Step 0):** An AWS Admin must first run the EMR Serverless job:
  [`emr/T3_final_emr_serverless_stats.py`](../emr/T3_final_emr_serverless_stats.py)
  Once the EMR job completes, it generates chunked data files and source-wise stats in CSV format. These stats must be aggregated to get `TOTAL_TOKENS` and passed to `shard.sh` as a parameter.
  - To aggregate, run: [`tools/estimate_total_tokens.py`](../tools/estimate_total_tokens.py)
  - For distribution analysis, use: [`notebooks/distribution_plots_notebook_extended.ipynb`](../notebooks/distribution_plots_notebook_extended.ipynb) (this also creates an aggregate CSV providing `TOTAL_TOKENS`)
- **S3_BUCKET** must be set (required)
- SSH access to an EC2 instance (Ubuntu)

### Run with defaults (c7gd.16xlarge)

```bash
export S3_BUCKET="your-bucket-name"
./commands.sh
```

### Dry run (preview config, no execution)

```bash
export S3_BUCKET="your-bucket-name"
./commands.sh --dry-run
```

### Foreground mode (CI / SSH — tracks exit code)

```bash
export S3_BUCKET="your-bucket-name"
./commands.sh --foreground
```

### Skip repo setup (CI self-hosted runners)

```bash
export S3_BUCKET="your-bucket-name"
./commands.sh --foreground --skip-repo-setup
```

---

## Estimating TOTAL_TOKENS

Before running the pipeline, estimate `TOTAL_TOKENS`
from the post-dedup `stats/` folder on EC2. This folder
contains one CSV file per source with token counts.

```bash
# Option 1: Python tool (aggregates across all CSVs)
python3 tools/estimate_total_tokens.py \
  --input-path "/mnt/nvme/stats/" \
  --input-format csv --quiet

# Option 2: Quick awk one-liner
awk -F',' 'NR>1{s+=$COL}END{print s}' \
  /mnt/nvme/stats/*.csv

# Option 3: Distribution Notebook
# Run notebooks/distribution_plots_notebook_extended.ipynb
# This generates an aggregate combined_source_distribution.csv 
# which includes total_tokens across all sources.

# Then replace the run following command before running commands.sh
export TOTAL_TOKENS=<replace with output of estimate_total_tokens.py(total_tokens)>
```

> [!TIP]
> Run this after the EMR dedup process generates
> `stats/` CSV files. The aggregate `total_tokens`
> across all sources becomes the `TOTAL_TOKENS`
> input for the coreset pipeline.

---

## Overriding Pipeline Parameters

All pipeline variables have defaults and can be
overridden via environment variables:

```bash
export S3_BUCKET="your-bucket-name"
export NUM_SHARDS=4
export STAGES="1B 3B"
export TOTAL_TOKENS=<replace with output of estimate_total_tokens.py(total_tokens)>
export BATCH_SIZE=50000
export RESUME=true
./commands.sh --foreground
```

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `S3_BUCKET` | *(required)* | S3 bucket name |
| `S3_INPUT_PATH` | `s3://${S3_BUCKET}/...` | Input data path |
| `NUM_SHARDS` | `8` | Parallel shards |
| `STAGES` | `1B` | Stages to run |
| `TOTAL_TOKENS` | `4523096944` | Input tokens |
| `BATCH_SIZE` | `80000` | Rows per batch |
| `CHECKPOINT_EVERY_N_BATCHES` | `3` | Ckpt frequency |
| `RESUME` | `false` | Resume from ckpt |
| `BRANCH_NAME` | `p3/feat/..._v2` | Git branch |

---

## Overriding Infrastructure Validation Thresholds

The infrastructure validation step (`validate_infra.sh`)
has defaults tuned for **c7gd.16xlarge** but all
thresholds are configurable.

### Example: Running on c6i.8xlarge (no NVMe)

```bash
export S3_BUCKET="your-bucket-name"
export EXPECTED_INSTANCE_TYPE="c6i.8xlarge"
export MIN_VCPU=32
export MIN_RAM_GB=60
export ENABLE_NVME=false
export MIN_EBS_FREE_GB=200
export MIN_EBS_ROOT_GB=500
./commands.sh --foreground
```

### Example: Running on m5d.4xlarge (with NVMe)

```bash
export S3_BUCKET="your-bucket-name"
export EXPECTED_INSTANCE_TYPE="m5d.4xlarge"
export MIN_VCPU=16
export MIN_RAM_GB=60
export MIN_NVME_FREE_GB=100
./commands.sh --foreground
```

### Full threshold reference

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `EXPECTED_INSTANCE_TYPE` | `c7gd.16xlarge` | EC2 instance type |
| `MIN_VCPU` | `64` | Min vCPU count |
| `MIN_RAM_GB` | `120` | Min RAM (GiB) |
| `ENABLE_NVME` | *(auto)* | NVMe checks |
| `MIN_NVME_FREE_GB` | `400` | Min NVMe free (GB) |
| `MIN_EBS_ROOT_GB` | `1000` | Min EBS root (GB) |
| `MIN_EBS_FREE_GB` | `800` | Min EBS free (GB) |
| `MAX_CPU_STEAL_PCT` | `1` | Max CPU steal % |
| `MAX_NVME_LATENCY_US` | `200` | Max NVMe lat (µs) |
| `MIN_NVME_IOPS` | `100000` | Min NVMe IOPS |
| `MIN_EBS_IOPS` | `16000` | Min EBS IOPS |
| `MAX_EBS_AWAIT_MS` | `3` | Max EBS await (ms) |
| `MIN_OPEN_FILES` | `65536` | Min open files |
| `MAX_SWAPPINESS` | `1` | Max swappiness |
| `MIN_S3_SPEED_MBS` | `500` | Min S3 speed MB/s |

> [!NOTE]
> `ENABLE_NVME` is **auto-detected** when not set.
>
> 1. Instance type has `d` suffix → `true`
> 2. NVMe mount point exists → `true`
> 3. NVMe block device found → `true`
> 4. None of the above → `false`

---

## Post-Run Steps

### Foreground mode

Steps run automatically after the pipeline completes:

- Monitoring is stopped and HTML /report is generated
- Coreset outputs are validated against curriculum
- Summary with file locations is printed

### Background mode (default)

After the pipeline finishes, run post-run steps manually:

```bash
# Stop monitoring
kill $(cat /mnt/nvme/logs/monitor.pid)

# Generate monitoring HTML report
python3 experiments/3_coreset_engineering/coreset_engine_v5/\
  scripts/monitor_report.py

# Validate coreset outputs
python3 experiments/3_coreset_engineering/coreset_engine_v5/\
  tools/validate_coreset_outputs.py \
    --curriculum experiments/3_coreset_engineering/\
      coreset_engine_v5/config/curriculum.yaml \
    --stages 1B --format both


# Merge Sharded Ablation reports in the standard manifest folder
python3 tools/merge_sharded_ablation_reports.py --overwrite

# Merge Sharded selected indices manifests
python3 tools/merge_selected_indices.py --coreset-root output/coresets\
  --stages 1B 3B 8B 70B

```

---

## Manual Setup (Without commands.sh)

### 1. SSH and System Setup

```bash
chmod 400 <pem_file>
ssh -i <pem_file> ubuntu@<public-ip>

sudo apt update
sudo apt install -y python3.12 python3.12-venv git python3-pip unzip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 2. AWS Authentication

```bash
aws configure
# Enter: Access Key, Secret Key, Region (us-east-1), Output (json)

aws s3 ls s3://<bucket>/processed_dataset/curriculum_pyspark_output/
```

### 3. Repository Setup

```bash
git clone https://github.com/The-School-of-AI/LLM.git
cd LLM
git fetch origin
git checkout -b p3/feat/stage-wise-coreset-selection_v2 origin/p3/feat/<Replace with latest branch>
```

### 4. Environment & Dependencies

```bash
cd experiments/3_coreset_engineering/
uv venv .venv
export UV_PROJECT_ENVIRONMENT=.venv
uv sync
```

### 5. Monitoring

```bash
chmod +x monitor.sh
nohup ./monitor.sh &
```

### 6. Run Pipeline

```bash
cd /home/ubuntu/LLM

nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
  --num-shards 8 \
  --stages "1B" \
  --input-path "s3://<bucket>/processed_dataset/curriculum_pyspark_output/" \
  --input-format jsonl \
  --total-tokens 4523096944 \
  --resume \
  > shard_run.log 2>&1 &

# Monitor
tail -f shard_run.log
ps aux | grep shard.sh
```
### 7. Post-run steps (default)

After the pipeline finishes, run post-run steps manually:

```bash
# Stop monitoring
kill $(cat /mnt/nvme/logs/monitor.pid)

# Generate monitoring HTML report
python3 experiments/3_coreset_engineering/coreset_engine_v5/\
  scripts/monitor_report.py

# Validate coreset outputs
python3 experiments/3_coreset_engineering/coreset_engine_v5/\
  tools/validate_coreset_outputs.py \
    --curriculum experiments/3_coreset_engineering/\
      coreset_engine_v5/config/curriculum.yaml \
    --stages 1B --format both


# Merge Sharded Ablation reports in the standard manifest folder
python3 tools/merge_sharded_ablation_reports.py --overwrite

# Merge Sharded selected indices manifests
python3 tools/merge_selected_indices.py --coreset-root output/coresets\
  --stages 1B 3B 8B 70B

---

## Output Sync to S3

```bash
# Sync all outputs
aws s3 sync \
  /home/ubuntu/LLM/experiments/3_coreset_engineering/\
coreset_engine_v5/output/ \
  s3://<bucket>/coreset_outputs/run_2/

# Upload logs
aws s3 cp ./shard_run.log s3://<bucket>/coreset_outputs/run_2/
```

---

## Alternative Data Transfer (Local Machine)

```bash
# rsync
rsync -avz -e "ssh -i <pem-file>" \
  ubuntu@<public-ip>:/home/ubuntu/LLM/experiments/\
  3_coreset_engineering/coreset_engine_v5/output/ \
  ./outputs/

# scp
scp -i <pem-file> -r \
  ubuntu@<public-ip>:/home/ubuntu/LLM/experiments/\
  3_coreset_engineering/coreset_engine_v5/output \
  ./outputs

# S3 to local
aws s3 sync s3://<bucket>/processed_dataset/curriculum_pyspark_output/ ./data/
```
