#!/usr/bin/env bash
# =============================================================================
# P5en.48xlarge Instance Launch Script
#
# Runs at instance boot to:
#   1. RAID-0 the 8× NVMe SSDs → /data
#   2. Set environment variables for NCCL / CUDA
#   3. Launch training with DeepSpeed
# =============================================================================
set -euo pipefail

# ---- RAID-0 Setup -----------------------------------------------------------
MOUNT_POINT="/data"
MD_DEVICE="/dev/md0"

# Detect NVMe devices (skip the root volume, typically /dev/nvme0n1)
NVME_DEVICES=($(lsblk -dn -o NAME,TYPE | awk '$2=="disk" && /nvme[1-9]/' | awk '{print "/dev/"$1}'))

echo "[RAID-0] Found ${#NVME_DEVICES[@]} NVMe devices: ${NVME_DEVICES[*]}"

if mountpoint -q "${MOUNT_POINT}" 2>/dev/null; then
    echo "[RAID-0] ${MOUNT_POINT} already mounted — skipping RAID setup"
else
    echo "[RAID-0] Creating RAID-0 array..."

    # Stop any existing array (ignore errors)
    mdadm --stop "${MD_DEVICE}" 2>/dev/null || true

    # Zero superblocks
    for dev in "${NVME_DEVICES[@]}"; do
        mdadm --zero-superblock "${dev}" 2>/dev/null || true
    done

    # Create RAID-0
    mdadm --create "${MD_DEVICE}" \
        --level=0 \
        --raid-devices=${#NVME_DEVICES[@]} \
        "${NVME_DEVICES[@]}" \
        --force --run

    # Format with XFS (best for large sequential I/O on NVMe)
    mkfs.xfs -f "${MD_DEVICE}"

    # Mount
    mkdir -p "${MOUNT_POINT}"
    mount "${MD_DEVICE}" "${MOUNT_POINT}"

    echo "[RAID-0] Mounted ${MD_DEVICE} at ${MOUNT_POINT}"
    df -h "${MOUNT_POINT}"
fi

# Create data directory
mkdir -p "${MOUNT_POINT}/dolmo"

# ---- NCCL / CUDA Environment ------------------------------------------------
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# EFA (Elastic Fabric Adapter) for multi-node
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa

# CUDA
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "[ENV] NCCL and CUDA environment configured"

# ---- Launch Training ---------------------------------------------------------
CONFIG="${1:-config.yaml}"
NUM_GPUS="${2:-8}"

echo "[LAUNCH] Starting DeepSpeed training"
echo "  Config: ${CONFIG}"
echo "  GPUs: ${NUM_GPUS}"

cd "$(dirname "$0")/.."

deepspeed --num_gpus="${NUM_GPUS}" main.py --config "${CONFIG}"
