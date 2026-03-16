#!/usr/bin/env bash
# =============================================================================
# setup.sh — Training instance bootstrap (non-AMI artifacts)
#
# Purpose:
# - Fetch config from S3 (CA cert, Vector config)
# - Read ClickHouse creds from AWS Secrets Manager
# - Write env files for Vector + training, and restart Vector if present
#
# Assumptions:
# - AMI already includes: awscli, jq, curl, bc; Vector installed; optional systemd unit
# - Instance profile has permissions: s3:GetObject (for specified prefixes),
#   sts:AssumeRole if cross-account is used elsewhere, and cloudwatch:PutMetricData if needed
# - Secret `t12/clickhouse` exists and contains JSON: {"endpoint": "https://<host>:8443", "writer-password": "..."}
# =============================================================================
set -euo pipefail

# --- Configurable inputs (override via environment or edit defaults) ---
T12_CONFIG_BUCKET="${T12_CONFIG_BUCKET:-p12-training-ops-base-CHANGE-ME}"
AWS_REGION="${AWS_REGION:-us-east-1}"
SECRET_ID="${SECRET_ID:-t12/clickhouse}"
VECTOR_SERVICE_NAME="${VECTOR_SERVICE_NAME:-t12-vector.service}"

# --- Prepare directories ---
sudo mkdir -p /etc/t12 /var/lib/vector /tmp/training_logs
sudo chown "${USER}:${USER}" /tmp/training_logs || true

# --- Pull CA cert + Vector config from S3 (private bucket recommended) ---
echo "[setup.sh] Downloading CA cert and Vector config from S3 bucket: ${T12_CONFIG_BUCKET} (${AWS_REGION})"
aws s3 cp "s3://${T12_CONFIG_BUCKET}/certs/ca_clickhouse.crt" /tmp/ca_clickhouse.crt --region "${AWS_REGION}"
aws s3 cp "s3://${T12_CONFIG_BUCKET}/vector/vector.toml" /tmp/vector.toml --region "${AWS_REGION}"

sudo mv /tmp/ca_clickhouse.crt /etc/t12/ca.crt
sudo mv /tmp/vector.toml /etc/t12/vector.toml
sudo chmod 644 /etc/t12/ca.crt /etc/t12/vector.toml

# --- Read credentials from Secrets Manager and write env files ---
echo "[setup.sh] Reading credentials from Secrets Manager: ${SECRET_ID} (${AWS_REGION})"
SECRET_JSON=$(aws secretsmanager get-secret-value \
  --secret-id "${SECRET_ID}" \
  --region "${AWS_REGION}" \
  --query 'SecretString' --output text)

CH_PASSWORD=$(echo "${SECRET_JSON}" | jq -r '."writer-password"')
CH_ENDPOINT=$(echo "${SECRET_JSON}" | jq -r '.endpoint')

cat <<EOF >/tmp/vector.env
CLICKHOUSE_HTTPS_ENDPOINT=${CH_ENDPOINT}
CLICKHOUSE_USER=p12_writer
CLICKHOUSE_PASSWORD=${CH_PASSWORD}
CLICKHOUSE_CA_CERT=/etc/t12/ca.crt
EOF

sudo mv /tmp/vector.env /etc/t12/vector.env
sudo chmod 600 /etc/t12/vector.env

# Copy for the training process (ubuntu user if present, else current user)
TARGET_HOME="/home/ubuntu"
if [ ! -d "$TARGET_HOME" ]; then TARGET_HOME="$HOME"; fi
sudo cp /etc/t12/vector.env "$TARGET_HOME/.t12.env"
sudo chown $(id -u):$(id -g) "$TARGET_HOME/.t12.env" 2>/dev/null || true
sudo chmod 600 "$TARGET_HOME/.t12.env"

echo "[setup.sh] Wrote /etc/t12/vector.env and $TARGET_HOME/.t12.env"

# --- Restart Vector service if present ---
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q "${VECTOR_SERVICE_NAME}"; then
  echo "[setup.sh] Restarting ${VECTOR_SERVICE_NAME}"
  sudo systemctl restart "${VECTOR_SERVICE_NAME}" || true
else
  echo "[setup.sh] Systemd unit ${VECTOR_SERVICE_NAME} not found; skipping restart."
fi

echo "[setup.sh] Completed successfully"
