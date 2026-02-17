# EC2 Coreset Pipeline Commands

This document contains the steps and commands required to manually run the coresets pipeline on an EC2 instance.

## 1. Setup and Prerequisites

### Change permissions on your local machine

```bash
chmod 400 <pem_file>
```

### Login to EC2

```bash
ssh -i <pem_file> ubuntu@<public-ip>
```

### Check installations and version

```bash
git --version
python3 --version # use python 3.12
```

### Run ubuntu specific commands

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv git python3-pip unzip

# Install uv (Astral)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

---

## 2. AWS CLI and S3 Access

### Check S3 listing access

If access issue then reach out to AWS team and share the error with them.

```bash
aws s3 ls s3://<container-name>/processed_dataset/curriculum_pyspark_output/source=flan/
```

### Install AWS CLI

Run if previous command did not work due to `awscli` not present in the OS.

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip
```

### AWS Authentication (Required)

Before running any S3 commands, you must authenticate with your AWS credentials.

```bash
aws configure
```

You will be prompted to enter:

- **AWS Access Key ID**
- **AWS Secret Access Key**
- **Default region name** (e.g., `us-east-1`)
- **Default output format** (e.g., `json`)

---

## 3. Repository Setup

### Git Configuration

Required to run only when new EC2 machine. **Note**: You can clone the repository without this step; it is only required to identify yourself when making `git commit`.

```bash
git config --global user.name "[YOUR_USERNAME]"
git config --global user.email "[EMAIL_ADDRESS]"
```

### Clone repository

First check if repo already exists or not.

```bash
ls -lrt LLM/
```

If the repository does not exist, run:

```bash
git clone https://github.com/The-School-of-AI/LLM.git
cd LLM
```

### Checkout a specific remote branch

```bash
git fetch origin
git checkout -b p3/feat/stage-wise-coreset-selection_v2 origin/p3/feat/stage-wise-coreset-selection_v2
```

### Pull latest changes

```bash
git pull origin p3/feat/stage-wise-coreset-selection_v2
```

---

## 4. Environment Setup

### Create virtual environment and sync dependencies

```bash
# Move to the experiment folder
cd experiments/3_coreset_engineering/

# Create isolated venv
uv venv .venv
export UV_PROJECT_ENVIRONMENT=.venv

# Sync all dependencies from the root uv.lock
uv sync
```

---

## 5. Running the Pipeline

You can use `tmux` to run the process in the background.

### Start a new tmux session

```bash
tmux new -s coreset
cd experiments/3_coreset_engineering/
export UV_PROJECT_ENVIRONMENT=.venv
python --version
```

### Run the shard command

```bash
nohup bash experiments/3_coreset_engineering/coreset_engine_v5/shard.sh \
  --num-shards 8 \
  --stages "1B" \
  --input-path "s3://<container-name>/processed_dataset/curriculum_pyspark_output/" \
  --input-format jsonl \
  --total-tokens 4523096944 \
  --resume \
  > shard_run.log 2>&1 &
```

### Monitor the process

```bash
ps aux | grep shard.sh
tail -f shard_run.log
```

### Detach from Tmux session

Press `Control + B`, then `D`.

### Kill the session

```bash
tmux kill-session -t coreset
```

---

## 6. Post-Processing and Sync

### Sync output to S3

Run only after confirming with AWS team.

```bash
aws s3 sync /home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/output/ s3://<container-name>/coreset_outputs/run_2/
```

### Uploading single files to S3

```bash
aws s3 cp ./shard_run.log s3://<container-name>/coreset_outputs/run_2/
aws s3 cp experiments/3_coreset_engineering/coreset_engine_v5/coreset_selection.log s3://<container-name>/coreset_outputs/run_2/
aws s3 cp experiments/3_coreset_engineering/coreset_engine_v5/coreset_errors.log s3://<container-name>/coreset_outputs/run_2/
```

---

## 7. Automated Execution Script

The setup and pipeline steps are fully automated in the `commands.sh` script.

> [!IMPORTANT]
> **Configuration**: Before running, open `commands.sh` and update the variables in the **"Configuration"** section (S3 Bucket, Branch Name, etc.).

### How to run the automated script on EC2

1. **Make the script executable**:

   ```bash
   chmod +x experiments/3_coreset_engineering/coreset_engine_v5/commands.sh
   ```

2. **Run the script**:

   ```bash
   ./experiments/3_coreset_engineering/coreset_engine_v5/commands.sh
   ```

The script will automatically detect if it is inside the repository, install `uv`, sync dependencies, and launch the pipeline in the background.

---

---

## 8. Alternative Data Transfer (Local Machine)

If you need to copy outputs directly to your local machine (though S3 sync is preferred):

### Using rsync

```bash
# Public IP can change per the EC2 spot instance
rsync -avz -e "ssh -i <pem-file>" ubuntu@<public-ip>:/home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/output/ ./coreset_engine_v5/outputs/
```

### Using scp

```bash
scp -i <pem-file> -r ubuntu@<public-ip>:/home/ubuntu/LLM/experiments/3_coreset_engineering/coreset_engine_v5/output ./outputs
```

### Downloading data from S3 to EC2/Local

```bash
aws s3 sync s3://<container-name>/processed_dataset/curriculum_pyspark_output/ ./data/
```
