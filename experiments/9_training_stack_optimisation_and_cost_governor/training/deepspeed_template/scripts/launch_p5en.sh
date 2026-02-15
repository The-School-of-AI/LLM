#!/bin/bash
# ============================================================================
# P5en.48xlarge Launch Script
#
# This script runs at instance boot and:
# 1. Sets up RAID-0 across 8× NVMe SSDs → /data
# 2. Configures NCCL and CUDA environment variables
# 3. Launches DeepSpeed training
#
# Usage:
#   chmod +x scripts/launch_p5en.sh
#   sudo ./scripts/launch_p5en.sh [--config config_70b.yaml]
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────
CONFIG_FILE="${1:---config config.yaml}"
NUM_GPUS=8
DATA_MOUNT="/data"
DATA_DIR="/data/dolmo"

# NVMe devices on P5en.48xlarge (adjust if different)
NVME_DEVICES=(/dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 \
              /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1)

RAID_DEVICE="/dev/md0"

# ── Helper ─────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Step 1: RAID-0 Setup ──────────────────────────────────────────────────
setup_raid0() {
    log "Step 1: Setting up RAID-0 across NVMe SSDs..."

    # Find available NVMe devices
    local available=()
    for dev in "${NVME_DEVICES[@]}"; do
        if [ -b "$dev" ]; then
            available+=("$dev")
        fi
    done

    if [ ${#available[@]} -eq 0 ]; then
        log "ERROR: No NVMe devices found!"
        exit 1
    fi

    log "  Found ${#available[@]} NVMe device(s): ${available[*]}"

    # Check if RAID already assembled
    if [ -b "$RAID_DEVICE" ]; then
        log "  RAID device $RAID_DEVICE already exists, checking mount..."
        if mountpoint -q "$DATA_MOUNT" 2>/dev/null; then
            log "  Already mounted at $DATA_MOUNT — skipping RAID setup."
            return
        fi
    fi

    # Stop any existing RAID
    mdadm --stop "$RAID_DEVICE" 2>/dev/null || true

    # Zero superblocks
    for dev in "${available[@]}"; do
        mdadm --zero-superblock "$dev" 2>/dev/null || true
    done

    # Create RAID-0
    mdadm --create "$RAID_DEVICE" \
        --level=0 \
        --raid-devices=${#available[@]} \
        "${available[@]}" \
        --force --run

    log "  Created RAID-0 at $RAID_DEVICE"

    # Create filesystem
    mkfs.ext4 -F -E lazy_itable_init=0 "$RAID_DEVICE"
    log "  Created ext4 filesystem"

    # Mount
    mkdir -p "$DATA_MOUNT"
    mount -o noatime,nodiratime "$RAID_DEVICE" "$DATA_MOUNT"
    log "  Mounted at $DATA_MOUNT"

    # Set permissions
    chmod 777 "$DATA_MOUNT"

    # Create data directory
    mkdir -p "$DATA_DIR"
    chmod 777 "$DATA_DIR"

    # Report size
    local total_size
    total_size=$(df -h "$DATA_MOUNT" | tail -1 | awk '{print $2}')
    log "  Total RAID-0 size: $total_size"
    log "  Data directory: $DATA_DIR"
}

# ── Step 2: Environment Variables ─────────────────────────────────────────
setup_env() {
    log "Step 2: Setting environment variables..."

    # NCCL optimizations for P5en.48xlarge with EFA
    export NCCL_DEBUG=WARN
    export NCCL_PROTO=simple
    export NCCL_SOCKET_IFNAME=eth0
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export NCCL_TREE_THRESHOLD=0

    # CUDA settings
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    # Memory optimization
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    log "  NCCL, CUDA, and memory environment configured."
}

# ── Step 3: Launch Training ───────────────────────────────────────────────
launch_training() {
    log "Step 3: Launching DeepSpeed training..."
    log "  GPUs: $NUM_GPUS"
    log "  Config: $CONFIG_FILE"

    deepspeed \
        --num_gpus="$NUM_GPUS" \
        main.py \
        $CONFIG_FILE
}

# ── Main ──────────────────────────────────────────────────────────────────
main() {
    log "============================================"
    log "P5en.48xlarge Training Launch Script"
    log "============================================"

    setup_raid0
    setup_env
    launch_training

    log "============================================"
    log "Training complete."
    log "============================================"
}

main "$@"
