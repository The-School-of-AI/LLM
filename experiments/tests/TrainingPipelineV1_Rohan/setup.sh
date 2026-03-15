#!/bin/bash
# =============================================================================
# ONE-SHOT P4DE SETUP — from fresh instance to training in ~5 minutes
#
# Usage:
#   scp -i <PEM> setup.sh <user>@<IP>:/tmp/
#   ssh -i <PEM> <user>@<IP> "bash /tmp/setup.sh"
#
# Prerequisites:
#   - p4de.24xlarge with 8x A100-80GB
#   - AWS credentials for s3data profile already in ~/.aws/ (or set below)
#   - Internet access for S3 + pip/uv downloads
# =============================================================================
set -euo pipefail

echo "============================================="
echo "P4DE Setup — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

# ---------------------------------------------------------------------------
# CONFIG — edit these if needed
# ---------------------------------------------------------------------------
CODE_S3="s3://t-endgame-experiment-logs-2"   # public bucket with code/lockfiles
DATA_S3="s3://t1-dataacquisition-datasets-2/shards_reordered/band_B0"
DATA_PROFILE="s3data"                         # AWS profile for data bucket
NVME_MOUNT="/mnt/local-nvme"
WORK_DIR="$NVME_MOUNT/Test24_1BCandidate"
DATA_DIR="$NVME_MOUNT/data/d1_shards"
PARALLEL_DL=64                                # parallel S3 download workers

# ---------------------------------------------------------------------------
# Detect OS
# ---------------------------------------------------------------------------
if command -v dnf &>/dev/null; then
    OS="amzn"
    PKG="dnf"
    CUDA_HOME="/usr/local/cuda"
elif command -v apt-get &>/dev/null; then
    OS="ubuntu"
    PKG="apt"
    CUDA_HOME="/usr"
else
    echo "ERROR: Unknown OS (no dnf or apt-get)"
    exit 1
fi
echo "[$(date '+%H:%M:%S')] OS=$OS, PKG=$PKG"

# ---------------------------------------------------------------------------
# Step 1: NVMe RAID-0 (skip if already mounted)
# ---------------------------------------------------------------------------
if mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
    echo "[$(date '+%H:%M:%S')] NVMe already mounted at $NVME_MOUNT"
elif [[ -d /opt/dlami/nvme ]]; then
    echo "[$(date '+%H:%M:%S')] DLAMI detected — symlinking NVMe"
    sudo ln -sf /opt/dlami/nvme "$NVME_MOUNT"
else
    echo "[$(date '+%H:%M:%S')] Setting up NVMe RAID-0..."
    if [[ "$OS" == "amzn" ]]; then
        sudo dnf install -y mdadm -q
    else
        sudo apt-get install -y -qq mdadm
    fi

    # Find NVMe instance store devices (exclude root volume)
    NVME_DEVS=$(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && $1~/nvme[1-9]/{print "/dev/"$1}' | sort)
    NUM_DEVS=$(echo "$NVME_DEVS" | wc -w)
    echo "  Found $NUM_DEVS NVMe devices: $NVME_DEVS"

    if [[ $NUM_DEVS -gt 0 ]]; then
        sudo mdadm --create --verbose /dev/md0 --level=0 --raid-devices=$NUM_DEVS $NVME_DEVS
        sudo mkfs.xfs -f /dev/md0
        sudo mkdir -p "$NVME_MOUNT"
        sudo mount -o noatime,nodiratime,logbufs=8 /dev/md0 "$NVME_MOUNT"
        sudo chown -R "$(whoami):$(whoami)" "$NVME_MOUNT"
        echo "[$(date '+%H:%M:%S')] NVMe RAID-0 ready: $(df -h $NVME_MOUNT | tail -1 | awk '{print $2}')"
    else
        echo "ERROR: No NVMe instance store devices found"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: System packages (CUDA toolkit + pip)
# ---------------------------------------------------------------------------
if ! command -v nvcc &>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Installing CUDA toolkit + pip..."
    if [[ "$OS" == "amzn" ]]; then
        sudo dnf install -y cuda-toolkit-12 python3-pip -q
        CUDA_HOME="/usr/local/cuda"
    else
        sudo apt-get update -qq
        sudo apt-get install -y -qq python3.12-venv python3-pip nvidia-cuda-toolkit
        CUDA_HOME="/usr"
    fi
    echo "[$(date '+%H:%M:%S')] nvcc installed: $(nvcc --version | grep release)"
else
    echo "[$(date '+%H:%M:%S')] nvcc already installed: $(nvcc --version | grep release)"
fi
export CUDA_HOME

# ---------------------------------------------------------------------------
# Step 3: Install uv
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[$(date '+%H:%M:%S')] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[$(date '+%H:%M:%S')] uv=$(uv --version)"

# ---------------------------------------------------------------------------
# Step 4: Upload code (if not present)
# ---------------------------------------------------------------------------
if [[ ! -f "$WORK_DIR/run.sh" ]]; then
    echo "[$(date '+%H:%M:%S')] Code not found at $WORK_DIR"
    echo "  Please upload Test24_1BCandidate to $NVME_MOUNT/"
    echo "  From local: scp -i <PEM> /tmp/Test24.tar.gz <user>@<IP>:$NVME_MOUNT/"
    echo "  Then: cd $NVME_MOUNT && tar xzf Test24.tar.gz"
    echo ""
    echo "  OR if code is on S3, uncomment the S3 download below."
    # Uncomment if code is on S3:
    # aws s3 cp s3://your-bucket/Test24_1BCandidate.tar.gz /tmp/
    # cd $NVME_MOUNT && tar xzf /tmp/Test24_1BCandidate.tar.gz
    exit 1
fi
echo "[$(date '+%H:%M:%S')] Code found at $WORK_DIR"

# ---------------------------------------------------------------------------
# Step 5: Python environment (uv sync)
# ---------------------------------------------------------------------------
if [[ ! -f "$WORK_DIR/.venv/bin/python3" ]]; then
    echo "[$(date '+%H:%M:%S')] Setting up Python venv..."
    cd "$WORK_DIR"

    # Download lockfiles if not present
    if [[ ! -f pyproject.toml ]]; then
        curl -sO https://t-endgame-experiment-logs-2.s3.us-east-1.amazonaws.com/pyproject.toml
        curl -sO https://t-endgame-experiment-logs-2.s3.us-east-1.amazonaws.com/uv.lock
    fi

    # Try full sync, fall back to skipping grouped-gemm
    uv sync 2>/dev/null || uv sync --no-install-package grouped-gemm
    echo "[$(date '+%H:%M:%S')] Python env ready"
else
    echo "[$(date '+%H:%M:%S')] Python venv already exists"
fi

# Verify
cd "$WORK_DIR"
source .venv/bin/activate
echo "[$(date '+%H:%M:%S')] Verifying packages..."
python3 -c "
import torch; print(f'  torch={torch.__version__}')
import triton; print(f'  triton={triton.__version__}')
import deepspeed; print(f'  deepspeed={deepspeed.__version__}')
import fla; print(f'  fla={fla.__version__}')
print(f'  CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')
"

# ---------------------------------------------------------------------------
# Step 6: Download D1 shards (parallel, ~90s for 611GB)
# ---------------------------------------------------------------------------
EXISTING_BINS=$(find "$DATA_DIR" -name 'tokens.bin' 2>/dev/null | wc -l)
if [[ $EXISTING_BINS -ge 4894 ]]; then
    echo "[$(date '+%H:%M:%S')] D1 shards already downloaded: $EXISTING_BINS"
else
    echo "[$(date '+%H:%M:%S')] Downloading D1 shards ($PARALLEL_DL parallel workers)..."
    mkdir -p "$DATA_DIR"

    aws s3 ls "$DATA_S3/" --profile "$DATA_PROFILE" \
        | awk '{print $NF}' | sed 's|/$||' > /tmp/d1_shard_list.txt
    TOTAL=$(wc -l < /tmp/d1_shard_list.txt)
    echo "  Found $TOTAL shards to download"

    download_shard() {
        local shard="$1"
        local dest="$DATA_DIR/$shard"
        [[ -f "$dest/tokens.bin" ]] && return 0
        mkdir -p "$dest"
        aws s3 cp "$DATA_S3/${shard}/tokens.bin" "$dest/tokens.bin" \
            --profile "$DATA_PROFILE" --quiet 2>/dev/null
    }
    export -f download_shard
    export DATA_DIR DATA_S3 DATA_PROFILE

    cat /tmp/d1_shard_list.txt | xargs -P "$PARALLEL_DL" -I{} bash -c 'download_shard "$@"' _ {}

    FINAL=$(find "$DATA_DIR" -name 'tokens.bin' | wc -l)
    echo "[$(date '+%H:%M:%S')] Downloaded: $FINAL/$TOTAL shards"
fi

# ---------------------------------------------------------------------------
# Step 7: Generate .idx files
# ---------------------------------------------------------------------------
EXISTING_IDX=$(find "$DATA_DIR" -name 'tokens.idx' 2>/dev/null | wc -l)
if [[ $EXISTING_IDX -ge 4894 ]]; then
    echo "[$(date '+%H:%M:%S')] .idx files already generated: $EXISTING_IDX"
else
    echo "[$(date '+%H:%M:%S')] Generating .idx files..."
    python3 << 'PYEOF'
import numpy as np
from pathlib import Path
local_dir = Path('/mnt/local-nvme/data/d1_shards')
BYTES_PER_BLOCK = 4096 * 4
IDX_HEADER = b'\x00' * 8
count = 0
for sd in sorted(local_dir.iterdir()):
    if not sd.is_dir(): continue
    bp, ip = sd / 'tokens.bin', sd / 'tokens.idx'
    if ip.exists() or not bp.exists(): continue
    n = bp.stat().st_size // BYTES_PER_BLOCK
    offsets = np.arange(n + 1, dtype=np.uint64) * BYTES_PER_BLOCK
    with open(ip, 'wb') as f:
        f.write(IDX_HEADER)
        f.write(offsets.tobytes())
    count += 1
    if count % 1000 == 0: print(f'  Generated {count} .idx files...')
print(f'  Generated {count} .idx files total')
PYEOF
fi

# ---------------------------------------------------------------------------
# Step 8: Set up bashrc
# ---------------------------------------------------------------------------
if ! grep -q 'DEEPSPEED_BIN' ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << BASHEOF

# Training environment (added by setup.sh)
export PATH=\$HOME/.local/bin:\$PATH
export CUDA_HOME=$CUDA_HOME
export PYTHON_BIN=$WORK_DIR/.venv/bin/python3
export DEEPSPEED_BIN=$WORK_DIR/.venv/bin/deepspeed
source $WORK_DIR/.venv/bin/activate
BASHEOF
    echo "[$(date '+%H:%M:%S')] ~/.bashrc updated"
fi

# ---------------------------------------------------------------------------
# Final verification
# ---------------------------------------------------------------------------
BINS=$(find "$DATA_DIR" -name 'tokens.bin' 2>/dev/null | wc -l)
IDXS=$(find "$DATA_DIR" -name 'tokens.idx' 2>/dev/null | wc -l)
SIZE=$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')

echo ""
echo "============================================="
echo "SETUP COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="
echo "  Code:    $WORK_DIR"
echo "  Data:    $DATA_DIR ($BINS bins, $IDXS idxs, $SIZE)"
echo "  Python:  $(python3 --version)"
echo "  GPUs:    $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "To start training:"
echo "  cd $WORK_DIR && bash run.sh"
echo ""
echo "To run in background:"
echo "  cd $WORK_DIR && nohup bash run.sh > /tmp/train.log 2>&1 &"
echo "  tail -f /tmp/train.log"
echo "============================================="
