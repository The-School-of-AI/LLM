#!/bin/bash
# ==============================================================================
# setup_nvme.sh — Safe "One-Click" NVMe Mounting (single or merged)
# ==============================================================================
# Auto-detects local ephemeral NVMe storage. With 2 devices (e.g. r7gd.16xlarge:
# ephemeral0, ephemeral1), merges them into one RAID 0 volume for maximum
# capacity and throughput. With 1 device, uses it as-is. Mounts at /mnt/nvme.
#
# Usage:
#   sudo ./setup_nvme.sh
# ==============================================================================

set -euo pipefail

MOUNT_POINT="/mnt/nvme"
RAID_DEVICE="/dev/md0"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

if [[ $EUID -ne 0 ]]; then
   log_err "This script must be run as root (use sudo)."
fi

# 0. If already mounted at MOUNT_POINT, just fix permissions and exit
if grep -q "$MOUNT_POINT" /proc/mounts; then
    log_info "Already mounted at $MOUNT_POINT. Permissions check..."
    chown -R ubuntu:ubuntu "$MOUNT_POINT" 2>/dev/null || true
    log_info "NVMe is ready at $MOUNT_POINT"
    exit 0
fi

# 1. Auto-detect ephemeral NVMe devices (exclude nvme0n1, usually EBS root)
log_info "Detecting ephemeral NVMe devices..."
DEVICES=()
while IFS= read -r dev; do
    [[ -z "$dev" ]] && continue
    if grep -q "/dev/$dev" /proc/mounts; then
        log_warn "/dev/$dev is already mounted; skipping."
        continue
    fi
    DEVICES+=( "/dev/$dev" )
done < <(lsblk -dn -o NAME | grep -E "^nvme[1-9]n1$" || true)

if [[ ${#DEVICES[@]} -eq 0 ]]; then
    log_err "No available ephemeral NVMe storage found (e.g. nvme1n1, nvme2n1). Ensure you are using a 'd' instance (r7gd, c7gd, m5d, etc.)."
fi

log_info "Found ${#DEVICES[@]} device(s): ${DEVICES[*]}"

# 2. One device: use directly. Two or more: merge with RAID 0
if [[ ${#DEVICES[@]} -eq 1 ]]; then
    DEVICE="${DEVICES[0]}"
    USE_RAID=false
else
    USE_RAID=true
    # Ensure mdadm is available
    if ! command -v mdadm &>/dev/null; then
        log_info "Installing mdadm for RAID..."
        apt-get update -qq && apt-get install -y mdadm
    fi
    # If RAID array already exists and is not mounted, use it
    if [[ -b "$RAID_DEVICE" ]] && ! grep -q "$RAID_DEVICE" /proc/mounts; then
        log_info "Existing RAID array $RAID_DEVICE found (not mounted). Will format and mount."
        DEVICE="$RAID_DEVICE"
        USE_RAID=false
    elif [[ -b "$RAID_DEVICE" ]] && grep -q "$RAID_DEVICE" /proc/mounts; then
        log_err "$RAID_DEVICE is already mounted. Unmount it first if you want to reconfigure."
    else
        log_info "Creating RAID 0 array $RAID_DEVICE from ${DEVICES[*]}..."
        mdadm --create "$RAID_DEVICE" --level=0 --raid-devices=${#DEVICES[@]} "${DEVICES[@]}" --force
        DEVICE="$RAID_DEVICE"
        USE_RAID=false
    fi
fi

# 3. Safety check: existing filesystem?
FS_TYPE=$(lsblk -no FSTYPE "$DEVICE" | tr -d ' ' || true)
if [[ -n "$FS_TYPE" ]]; then
    log_warn "$DEVICE already has a filesystem ($FS_TYPE)."
    log_info "Attempting to mount existing filesystem..."
else
    log_info "Formatting $DEVICE as ext4..."
    mkfs.ext4 -E lazy_itable_init=0,lazy_journal_init=0 "$DEVICE"
fi

# 4. Mount
log_info "Mounting $DEVICE to $MOUNT_POINT..."
mkdir -p "$MOUNT_POINT"
mount -o discard,defaults "$DEVICE" "$MOUNT_POINT"

# 5. Permissions
log_info "Setting ownership to ubuntu:ubuntu..."
chown -R ubuntu:ubuntu "$MOUNT_POINT"

# 6. Verify
TOTAL_GB=$(df -BG "$MOUNT_POINT" | awk 'NR==2{print $2}' | tr -d 'G')
log_info "Success! $DEVICE mounted at $MOUNT_POINT ($TOTAL_GB GB available)."

if [[ "$DEVICE" == "$RAID_DEVICE" ]]; then
    log_info "Save RAID config so the array is reassembled on reboot:"
    echo "  mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf"
    log_info "Optional: add to /etc/fstab for mount on boot:"
    echo "  $(blkid -s UUID -o value $DEVICE)  $MOUNT_POINT  ext4  defaults,nofail,discard  0  2"
else
    log_info "Optional: add to /etc/fstab for mount on boot:"
    echo "  UUID=$(blkid -s UUID -o value $DEVICE)  $MOUNT_POINT  ext4  defaults,nofail,discard  0  2"
fi
