#!/bin/bash
# ==============================================================================
# setup_nvme.sh — Safe "One-Click" NVMe Mounting
# ==============================================================================
# Auto-detects local ephemeral NVMe storage, checks for existing filesystems,
# and mounts to /mnt/nvme.
#
# Usage:
#   sudo ./setup_nvme.sh
# ==============================================================================

set -euo pipefail

MOUNT_POINT="/mnt/nvme"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

if [[ $EUID -ne 0 ]]; then
   log_err "This script must be run as root (use sudo)."
fi

# 1. Auto-detect NVMe device
# Usually nvme1n1 or nvme2n1 (nvme0n1 is typically the EBS root)
log_info "Detecting ephemeral NVMe devices..."
DEVICE=""

# Find devices that have NO partitions and are NOT the root device
# We look for "Disk" devices that are not nvme0n1 (usually EBS)
POTENTIAL_DEVICES=$(lsblk -dn -o NAME | grep "nvme[1-9]n1" || true)

for dev in $POTENTIAL_DEVICES; do
    # Check if it's already mounted
    if grep -q "/dev/$dev" /proc/mounts; then
        log_warn "/dev/$dev is already mounted. Checking mount point..."
        if grep -q "$MOUNT_POINT" /proc/mounts; then
            log_info "Already mounted at $MOUNT_POINT. Permissions check..."
            chown -R ubuntu:ubuntu "$MOUNT_POINT"
            log_info "NVMe is ready at $MOUNT_POINT"
            exit 0
        fi
        continue
    fi
    DEVICE="/dev/$dev"
    break
done

if [[ -z "$DEVICE" ]]; then
    log_err "No available ephemeral NVMe storage found (e.g., nvme1n1). Ensure you are using a 'd' instance (c7gd, m5d, etc.)."
fi

log_info "Target device identified: $DEVICE"

# 2. Safety Check: Is there already a filesystem?
log_info "Checking for existing filesystem on $DEVICE..."
FS_TYPE=$(lsblk -no FSTYPE "$DEVICE" | tr -d ' ')

if [[ -n "$FS_TYPE" ]]; then
    log_warn "$DEVICE already has a filesystem ($FS_TYPE)."
    log_info "Attempting to mount existing filesystem..."
else
    log_info "No filesystem detected. Formatting $DEVICE as ext4..."
    mkfs.ext4 -E lazy_itable_init=0,lazy_journal_init=0 "$DEVICE"
fi

# 3. Mount
log_info "Mounting $DEVICE to $MOUNT_POINT..."
mkdir -p "$MOUNT_POINT"
mount -o discard,defaults "$DEVICE" "$MOUNT_POINT"

# 4. Permissions
log_info "Setting ownership to ubuntu:ubuntu..."
chown -R ubuntu:ubuntu "$MOUNT_POINT"

# 5. Verify
TOTAL_GB=$(df -BG "$MOUNT_POINT" | awk 'NR==2{print $2}' | tr -d 'G')
log_info "Success! $DEVICE mounted at $MOUNT_POINT ($TOTAL_GB GB available)."
log_info "Commands for manual persistence (optional, add to /etc/fstab):"
echo "  UUID=$(blkid -s UUID -o value $DEVICE)  $MOUNT_POINT  ext4  defaults,nofail,discard  0  2"
