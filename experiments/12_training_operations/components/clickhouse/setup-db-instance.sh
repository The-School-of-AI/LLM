#!/usr/bin/env bash
# =============================================================================
# P12 ClickHouse DB Instance Setup
#
# Run this on a freshly launched DB instance (via SSM Session Manager).
# It handles everything: EBS format/mount, git clone, auth setup, docker start.
#
# Prerequisites:
#   - Instance launched with user data that installed docker, docker-compose, git, awscli
#   - EBS data volume attached at /dev/xvdf (or NVMe equivalent)
#   - Instance has p12-clickhouse-db-profile IAM role
#
# Usage:
#   # Interactive (prompts for passwords, CIDRs):
#   bash setup-db-instance.sh
#
#   # Non-interactive:
#   P12_WRITER_PASSWORD=... P12_READER_PASSWORD=... \
#   TRAINING_SUBNET_CIDR=10.0.1.0/24 DASHBOARD_SUBNET_CIDR=10.0.2.0/24 \
#   P12_REGION=us-east-1 bash setup-db-instance.sh
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/p12-db-setup.log) 2>&1
echo "P12 DB instance setup started at $(date -u)"

# ---- 1. Format and mount the EBS volume ----
echo "[1/5] Setting up EBS volume..."

DEVICE="/dev/xvdf"
if [ ! -b "$DEVICE" ]; then
  # Nitro instances use nvme naming — find the unformatted/unmounted device
  DEVICE=$(lsblk -o NAME,SIZE -dn | awk 'NR>1{print "/dev/"$1}' | while read d; do
    if ! mount | grep -q "$d"; then echo "$d"; break; fi
  done)
fi
echo "Data device: $DEVICE"

if ! sudo blkid "$DEVICE" 2>/dev/null | grep -q ext4; then
  sudo mkfs.ext4 -L p12-clickhouse "$DEVICE"
fi

sudo mkdir -p /data/clickhouse
if ! mountpoint -q /data/clickhouse; then
  sudo mount "$DEVICE" /data/clickhouse
fi

sudo mkdir -p /data/clickhouse/data /data/clickhouse/logs
sudo chown -R 101:101 /data/clickhouse

# Persist in fstab (idempotent)
UUID=$(sudo blkid -s UUID -o value "$DEVICE")
if ! grep -q "$UUID" /etc/fstab; then
  echo "UUID=$UUID /data/clickhouse ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
fi

df -h /data/clickhouse
echo "✓ EBS volume mounted"

# ---- 2. Pull the clickhouse/ directory from git ----
echo "[2/5] Pulling clickhouse config from git..."

REPO_URL="https://github.com/<org>/<repo>.git"
REPO_TMP="/tmp/p12-repo"

if [ ! -d "$REPO_TMP" ]; then
  git clone "$REPO_URL" "$REPO_TMP"
fi

rm -rf ~/clickhouse
cp -r "$REPO_TMP/experiments/12_training_operations/components/clickhouse" ~/clickhouse
echo "✓ clickhouse/ directory ready at ~/clickhouse"

# ---- 3. Run setup-auth.sh ----
echo "[3/5] Running auth setup..."

cd ~/clickhouse
export DB_PRIVATE_IP="${DB_PRIVATE_IP:-$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)}"
export P12_REGION="${P12_REGION:-us-east-1}"

bash setup-auth.sh

# ---- 4. Start ClickHouse ----
echo "[4/5] Starting ClickHouse..."

sudo docker compose up -d

echo "Waiting for ClickHouse to be healthy..."
for i in $(seq 1 30); do
  if sudo docker exec p12-clickhouse clickhouse-client --query "SELECT 1" &>/dev/null; then
    echo "✓ ClickHouse is healthy"
    break
  fi
  sleep 2
done

# ---- 5. Verify ----
echo "[5/5] Verifying..."

sudo docker exec p12-clickhouse clickhouse-client \
  --query "SHOW TABLES FROM training_observability"

echo ""
echo "============================================================"
echo "  DB instance ready!"
echo "  Private IP: $DB_PRIVATE_IP"
echo "  ClickHouse HTTPS: https://${DB_PRIVATE_IP}:8443"
echo "============================================================"

# Cleanup
rm -rf "$REPO_TMP"

echo "P12 DB instance setup completed at $(date -u)"
