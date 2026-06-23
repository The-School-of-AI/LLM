# EC2 Coreset Pipeline Commands

This document covers how to run the coreset pipeline on an EC2 instance, using either the **automated scripts** or **manual steps**.

---

## 1. Access & Setup

### How to login to EC2 from local machine

```bash
ssh -i "<key_file>.pem" ubuntu@<public-ip>
```

### Copy scripts to EC2

```bash
scp -i <key_file>.pem /Users/user name/Documents/git/TSAI/ERA4/final-capstone/LLM/experiments/3_coreset_engineering/coreset_engine_v5/scripts/* ubuntu@<public-ip>:/home/ubuntu/
```

---

## 2. Prerequisites & Token Estimation

### Step 0: EMR Job

An AWS Admin must first run the EMR Serverless job: [`emr/T3_final_emr_serverless_stats.py`](../emr/T3_final_emr_serverless_stats.py).
Once the EMR job completes, it generates chunked data files and source-wise stats in CSV format. These stats must be aggregated to get `TOTAL_TOKENS` for `shard.sh`.

### Estimating TOTAL_TOKENS

Before running the pipeline, you need the aggregate token count from the EMR/Dedup stats:

#### Option 1: Python tool (aggregates across all CSVs)

```bash
python3 experiments/3_coreset_engineering/coreset_engine_v5/tools/estimate_total_tokens.py \
    --input-path "/path/to/stats/" --input-format csv --quiet
```

#### Option 2: Quick awk one-liner

```bash
# Sums the token column across all source CSVs
awk -F',' 'NR>1{s+=$COL}END{print s}' /path/to/stats/*.csv
```

#### Analysis & Distribution

- To aggregate and analyze: [`tools/estimate_total_tokens.py`](../tools/estimate_total_tokens.py)
- For distribution analysis (bands/domains): [`notebooks/distribution_plots_notebook_extended.ipynb`](../notebooks/distribution_plots_notebook_extended.ipynb)

---

## 3. Automated Run (Quick Start)

The scripts automate the full pipeline in 8 steps:

1. System Setup (including `sysctl vm.swappiness=0`)
2. AWS Authentication Check
3. Repository Setup
4. Dependency Sync (via UV)
5. Infrastructure Validation
6. Monitoring Setup
7. Pipeline Execution
8. Post-Run Reports

There are two entry points:

- **`commands.sh`**: Standard background execution (manual production).
- **`commands_ci.sh`**: Foreground execution (CI/CD or SSH).

### Recommended Production Run

```bash
# Setup thresholds and environment
sudo sysctl -w vm.swappiness=0

export S3_BUCKET="<bucket_name>"
export S3_INPUT_PATH="s3://<bucket_name>/processed_dataset/curriculum_pyspark_output/"
export NUM_SHARDS=8
export STAGES="1B"
export BATCH_SIZE=80000
export ENABLE_NVME=true
export TOTAL_TOKENS="<total_tokens>"
export CHECKPOINT_EVERY_N_BATCHES=20
export EXPECTED_INSTANCE_TYPE="c7gd.2xlarge"

# Step 1: Setup NVMe (if fresh instance)
sudo bash experiments/3_coreset_engineering/coreset_engine_v5/scripts/setup_nvme.sh

# Step 2: Launch pipeline in background
bash commands.sh
```

---

## 4. Manual Execution

If you prefer to run `shard.sh` directly:

```bash
# Example one-liner for specific source
bash shard.sh \
  --num-shards 10 \
  --input-path "s3://<bucket_name>/processed_dataset/curriculum_pyspark_output/source=proof_pile_2-open_web_math/" \
  --total-tokens 6960563545 \
  --stages "1B" \
  --checkpoint-every-n-batches 50 \
  --batch-size 50000 \
  --used-cache-max-entries 1000000 \
  --used-cache-stats-every 100 \
  --batch-prefetch-mode auto
```

---

## 5. Monitoring & Management (Background Mode)

- **Monitor logs**: `tail -f /home/ubuntu/LLM/shard_run.log`
- **Check process**: `ps aux | grep shard.sh`
- **Stop pipeline**: `pkill -f shard.sh`
- **Stop monitor**: `kill $(cat /mnt/nvme/logs/monitor.pid)`

---

## Manual Setup (Without commands.sh)

If you need to initialize the environment manually:

### 1. System & Git

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv git unzip dstat bc sysstat
sudo sysctl -w vm.swappiness=0
git clone https://github.com/The-School-of-AI/LLM.git && cd LLM
git checkout p3/feat/stage-wise-coreset-selection_v2
```

### 2. AWS Authentication

```bash
aws configure
# Or ensure IAM Role is attached to the instance
aws sts get-caller-identity
```

### 3. Repository Setup & Dependencies

```bash
cd experiments/3_coreset_engineering
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv .venv && source .venv/bin/activate
uv sync
```

### 4. Infrastructure Check

```bash
cd coreset_engine_v5
sudo -E bash scripts/validate_infra.sh
# If NVMe is needed:
sudo bash scripts/setup_nvme.sh
```

### 5. Monitoring

```bash
# Start background monitoring
nohup bash scripts/monitor.sh > /dev/null 2>&1 &
```

### 6. Run Pipeline

```bash
# Example manual execution
bash shard.sh \
  --num-shards 8 \
  --input-path "s3://<bucket_name>/processed_dataset/curriculum_pyspark_output/" \
  --total-tokens 400000000000 \
  --stages "1B" \
  --checkpoint-every-n-batches 20 \
  --batch-size 80000
```

### 7. Post-run steps (default)

After the pipeline finishes, run these manually if you used `commands.sh`:

```bash
# Generate monitoring report
python3 /home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/scripts/monitor_report.py

# Validate coreset outputs
python3 /home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/tools/validate_coreset_outputs.py \
    --curriculum /home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/config/curriculum.yaml \
    --output-dir /path/to/coresets \
    --stages 1B --format both

# Merge Sharded Ablation reports in the standard manifest folder
python3 tools/merge_sharded_ablation_reports.py --overwrite

# Merge Sharded selected indices manifests
python3 tools/merge_selected_indices.py --coreset-root output/coresets \
  --stages 1B 3B 8B 70B

# Sync high-volume outputs to S3
aws s3 sync /path/to/coresets s3://<bucket_name>/coresets_output/run_<id>/ --quiet --no-progress
```

---

## Output Sync to S3

Ensure all outputs and logs are safely stored in S3 before shutting down:

```bash
# Sync pipeline artifacts
aws s3 sync /home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/output/ s3://<bucket_name>/coreset_outputs/run_<id>/

# Sync monitoring logs
aws s3 sync /mnt/nvme/logs/ s3://<bucket_name>/coreset_outputs/run_<id>/
```

---

## Alternative Data Transfer (Local Machine)

### Download logs to your local machine

#### Option 1: Using SCP

```bash
scp -i T3-Coreset.pem -r ubuntu@<public-ip>:/mnt/nvme/logs ./local_logs
```

#### Option 2: Using Rsync (Resumable)

```bash
rsync -avz -e "ssh -i T3-Coreset.pem" ubuntu@<public-ip>:/mnt/nvme/logs/ ./local_logs/
```
