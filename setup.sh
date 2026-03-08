# Terminal 1: 

# 1. Create NVME mount with all 8 devices 
sudo dnf install -y mdadm

# check the lsblk output to find the new NVMe device (e.g., /dev/nvme1n1)
lsblk

sudo mdadm --create --verbose /dev/md0 \
  --level=0 --raid-devices=8 \
  /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 \
  /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 \
  /dev/nvme7n1 /dev/nvme8n1

sudo mkfs.xfs -f /dev/md0

sudo mkdir -p /mnt/local-nvme

sudo mount -o noatime,nodiratime,logbufs=8 /dev/md0 /mnt/local-nvme
sudo chown -R "$(whoami):$(whoami)" /mnt/local-nvme

# 2. Sync Github repo to NVME mount
cd /mnt/local-nvme
git clone -b feature/experiments-p4d-gsa-memory-fixes https://github.com/The-School-of-AI/LLM.git



## Terminal 2:
sudo dnf install -y cuda-toolkit-12
sudo dnf install -y python3-pip
pip install uv

## ensure the github code is available on the NVME mount
cd /mnt/local-nvme/LLM
# ensure that the pyproject.toml file or uv.lock file is present in the root of the repo
uv sync 
uv pip install -U megatron-core transformer-engine
uv pip install grouped-gemm

### For all the runs/new terminals, activate the venv before running run.sh
source /mnt/local-nvme/LLM/.venv/bin/activate


# For tests
## copy the tokenizer to all folders
cd /mnt/local-nvme/LLM
for d in experiments/tests/*/code/src/tokenizer/; do
  cp -i tokenizer.json "$d"    # -i = interactive prompt
done

## sync data from s3
# aws s3 sync s3://t-endgame-experiment-logs-2/Test_2_20-step_save_init_model/init /mnt/local-nvme/LLM/experiments/tests/Test_2_20-step_save_init_model/results/init

aws s3 sync s3://t-endgame-experiment-logs-2/shards/wikitext_shards/ /mnt/local-nvme/LLM/experiments/tests/Test_14_gsa_only_liger_kernels_1000steps-OngoingRun3/data/wikitext_shards/
