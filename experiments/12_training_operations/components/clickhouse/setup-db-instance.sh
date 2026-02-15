#!/usr/bin/env bash
# =============================================================================
# P12 ClickHouse DB Instance Setup (idempotent — safe to re-run)
#
# Run this on a freshly launched DB instance (via SSM Session Manager).
# It handles everything: EBS format/mount, git clone, auth setup, docker start.
#
# Re-running is safe:
#   - EBS won't be reformatted if already ext4
#   - Git repo is pulled (not re-cloned)
#   - Generated artifacts (TLS certs, users.xml, .env) are preserved across updates
#   - setup-auth.sh skips credential setup on re-run (pass passwords to force)
#   - docker compose up -d is a no-op for unchanged containers
#
# Prerequisites:
#   - Instance launched with user data that installed docker, docker-compose, git, awscli
#   - EBS data volume attached at /dev/xvdf (or NVMe equivalent)
#   - Instance has p12-clickhouse-db-profile IAM role
#
# Usage:
#   # First run — interactive (prompts for passwords, CIDRs):
#   bash setup-db-instance.sh
#
#   # First run — non-interactive:
#   P12_WRITER_PASSWORD=... P12_READER_PASSWORD=... \
#   TRAINING_SUBNET_CIDR=10.0.1.0/24 DASHBOARD_SUBNET_CIDR=10.0.2.0/24 \
#   P12_REGION=us-east-1 bash setup-db-instance.sh
#
#   # Re-run (code update, no credential changes):
#   bash setup-db-instance.sh
# =============================================================================

set -euo pipefail
exec > >(tee -a /var/log/p12-db-setup.log) 2>&1
echo ""
echo "========== P12 DB instance setup started at $(date -u) =========="

# ---- 1. Format and mount the EBS volume ----
echo "[1/5] Setting up EBS volume..."

DEVICE="/dev/xvdf"
if [ ! -b "$DEVICE" ]; then
  # Nitro instances expose EBS as NVMe — find the data volume
  ROOT_DISK=$(lsblk -ndo PKNAME "$(findmnt -no SOURCE /)")
  DEVICE=$(lsblk -dnpo NAME,TYPE | awk '$2 == "disk"' | awk '{print $1}' | grep -v "$ROOT_DISK" | head -1)
fi

if [ -z "${DEVICE:-}" ] || [ ! -b "$DEVICE" ]; then
  echo "ERROR: No EBS data volume found. Is the volume attached?"
  exit 1
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
REPO_SRC="$REPO_TMP/experiments/12_training_operations/components/clickhouse"

if [ -d "$REPO_TMP/.git" ]; then
  if ! git -C "$REPO_TMP" pull --ff-only 2>/dev/null; then
    echo "git pull failed (non-fast-forward?) — re-cloning"
    rm -rf "$REPO_TMP"
    git clone "$REPO_URL" "$REPO_TMP"
  else
    echo "✓ Repo updated (git pull)"
  fi
else
  rm -rf "$REPO_TMP"
  git clone "$REPO_URL" "$REPO_TMP"
  echo "✓ Repo cloned"
fi

# Generated artifacts that must survive code updates
GENERATED_FILES=(
  "tls/ca"
  "tls/server.crt"
  "tls/server.key"
  "users.d/p12-users.xml"
  "training-instance.env"
  "dashboard.env"
)

if [ -d ~/clickhouse ]; then
  # Preserve generated artifacts, then replace repo code
  BACKUP="/tmp/p12-generated-backup"
  rm -rf "$BACKUP" && mkdir -p "$BACKUP"

  for f in "${GENERATED_FILES[@]}"; do
    if [ -e ~/clickhouse/"$f" ]; then
      mkdir -p "$BACKUP/$(dirname "$f")"
      cp -r ~/clickhouse/"$f" "$BACKUP/$f"
    fi
  done

  rm -rf ~/clickhouse
  cp -r "$REPO_SRC" ~/clickhouse

  # Restore generated artifacts over fresh code
  for f in "${GENERATED_FILES[@]}"; do
    if [ -e "$BACKUP/$f" ]; then
      TARGET=~/clickhouse/"$f"
      rm -rf "$TARGET"
      mkdir -p "$(dirname "$TARGET")"
      cp -r "$BACKUP/$f" "$TARGET"
    fi
  done
  rm -rf "$BACKUP"
  echo "✓ clickhouse/ updated (generated artifacts preserved)"
else
  cp -r "$REPO_SRC" ~/clickhouse
  echo "✓ clickhouse/ directory ready at ~/clickhouse"
fi

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
HEALTHY=false
for i in $(seq 1 30); do
  if sudo docker exec p12-clickhouse clickhouse-client --query "SELECT 1" &>/dev/null; then
    echo "✓ ClickHouse is healthy"
    HEALTHY=true
    break
  fi
  sleep 2
done

if [ "$HEALTHY" != "true" ]; then
  echo "ERROR: ClickHouse did not become healthy within 60s"
  sudo docker logs p12-clickhouse --tail 20
  exit 1
fi

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

echo "P12 DB instance setup completed at $(date -u)"
