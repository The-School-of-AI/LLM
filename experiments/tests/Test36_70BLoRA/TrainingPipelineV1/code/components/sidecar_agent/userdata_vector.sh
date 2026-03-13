#!/usr/bin/env bash
# =============================================================================
# T12 Vector Sidecar — EC2 User Data Bootstrap
#
# Installs Vector on Ubuntu 18.04+, pulls CA cert and config from S3,
# retrieves ClickHouse credentials from Secrets Manager via cross-account
# assume-role, and starts Vector as a systemd service.
#
# Prerequisites — complete these before running (see Post_Clickhouse_Install.md):
#   Sec 3: Create the t12/clickhouse secret in the infra account (Account B)
#   Sec 4: Run setup-infra-account.sh (Account B) and setup-training-account.sh (Account A)
#   S3:    Upload CA cert to s3://<bucket>/certs/ca_clickhouse.crt (public-read)
#   SG:    Outbound HTTPS (443) to S3/Secrets Manager/STS/CloudWatch + TCP 8443 to ClickHouse
#
# Execution modes (all idempotent):
#   First boot:  aws ec2 run-instances --user-data file://userdata_vector.sh ...
#   Manual/rerun: sudo bash userdata_vector.sh
#
# Logs: cat /var/log/t12-userdata.log
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/t12-userdata.log) 2>&1
echo "T12 Vector bootstrap started at $(date -u)"

# ---- 1. System packages ----
echo "[1/9] Installing dependencies..."
export DEBIAN_FRONTEND=noninteractive

# Wait for dpkg lock to clear (common on first boot with unattended-upgrades)
wait_for_dpkg_lock() {
  local max_attempts=30
  for i in $(seq 1 $max_attempts); do
    if ! lsof /var/lib/dpkg/lock-frontend >/dev/null 2>&1; then
      return 0
    fi
    echo "Waiting for dpkg lock... ($i/$max_attempts)"
    sleep 5
  done
  echo "ERROR: dpkg lock still held after $max_attempts attempts"
  exit 1
}

wait_for_dpkg_lock
apt-get update -qq

wait_for_dpkg_lock
apt-get install -y -qq awscli jq curl bc

# ---- Configuration (EDIT THESE) ----
T12_CONFIG_BUCKET="p12-training-ops-base-869633161654" # replace with your S3 bucket name
AWS_REGION="${AWS_REGION:-us-east-1}"
PREFIX="t12-TrainingOperations-239"                                  # REPLACE with your unique prefix for resource naming
SECRETS_ROLE_ARN="arn:aws:iam::205991465724:role/t12-secrets-reader" # cross-account role for Secrets Manager access

# ---- 2. Install Vector ----
echo "[2/9] Installing Vector..."
if ! command -v vector &>/dev/null; then
  # Set HOME for the installer if not defined (EC2 user data runs without it)
  export HOME="${HOME:-/root}"
  curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local
fi
/usr/local/bin/vector --version

# ---- 3. Create directories ----
echo "[3/9] Creating directories..."
mkdir -p /etc/t12
mkdir -p /tmp/training_logs
mkdir -p /var/lib/vector
chown ubuntu:ubuntu /tmp/training_logs
chown ubuntu:ubuntu /var/lib/vector

# ---- 4. Pull CA cert + Vector config from S3 (public URLs, no IAM needed) ----
echo "[4/9] Pulling config from S3..."
curl -fsSL "https://${T12_CONFIG_BUCKET}.s3.amazonaws.com/certs/ca_clickhouse.crt" -o /etc/t12/ca.crt
curl -fsSL "https://raw.githubusercontent.com/The-School-of-AI/LLM/refs/heads/P12/feat/training-ops-base/experiments/12_training_operations/components/sidecar_agent/vector.toml" -o /etc/t12/vector.toml
chmod 644 /etc/t12/ca.crt

# ---- 5. Assume cross-account role for Secrets Manager access ----
echo "[5/9] Assuming cross-account role for Secrets Manager..."
CREDS_JSON=$(aws sts assume-role \
  --role-arn "$SECRETS_ROLE_ARN" \
  --role-session-name "t12-vector-$(hostname -s)" \
  --duration-seconds 900 \
  --output json)

export AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | jq -r '.Credentials.SessionToken')

# ---- 6. Pull credentials from Secrets Manager (using cross-account creds) ----
echo "[6/9] Retrieving credentials from Secrets Manager..."
SECRET_JSON=$(aws secretsmanager get-secret-value \
  --secret-id "t12/clickhouse" \
  --region "$AWS_REGION" \
  --query 'SecretString' --output text)

CH_PASSWORD=$(echo "$SECRET_JSON" | jq -r '."writer-password"')
CH_ENDPOINT=$(echo "$SECRET_JSON" | jq -r '.endpoint')

# Drop assumed credentials — back to instance profile for CloudWatch etc.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

# Write environment file (read by systemd + training process)
cat >/etc/t12/vector.env <<EOF
CLICKHOUSE_HTTPS_ENDPOINT=${CH_ENDPOINT}
CLICKHOUSE_USER=p12_writer
CLICKHOUSE_PASSWORD=${CH_PASSWORD}
CLICKHOUSE_CA_CERT=/etc/t12/ca.crt
EOF
chmod 600 /etc/t12/vector.env

# Copy for the training process (ubuntu user)
cp /etc/t12/vector.env /home/ubuntu/.t12.env
chown ubuntu:ubuntu /home/ubuntu/.t12.env
chmod 600 /home/ubuntu/.t12.env

# ---- 7. Create Vector systemd service ----
echo "[7/9] Creating Vector systemd service..."
cat >/etc/systemd/system/t12-vector.service <<'UNIT'
[Unit]
Description=T12 Vector Sidecar (ClickHouse log shipper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
EnvironmentFile=/etc/t12/vector.env
ExecStart=/usr/local/bin/vector --config /etc/t12/vector.toml
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=t12-vector

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/vector /tmp/training_logs

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable t12-vector
systemctl restart t12-vector

# ---- 8. Install training-side health check ----
echo "[8/9] Installing health check..."
cat >/usr/local/bin/t12-training-healthcheck.sh <<'HEALTHCHECK'
#!/usr/bin/env bash
set -uo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
NAMESPACE="T12/Training"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")

if [ -f /etc/t12/vector.env ]; then
  set -a; source /etc/t12/vector.env; set +a
fi

push_metric() {
  local name="$1" value="$2" unit="${3:-None}"
  aws cloudwatch put-metric-data \
    --namespace "$NAMESPACE" \
    --metric-name "$name" \
    --value "$value" \
    --unit "$unit" \
    --dimensions "InstanceId=$INSTANCE_ID" \
    --region "$REGION" 2>/dev/null
}

# 1. Vector alive
if pgrep -x vector &>/dev/null; then
  push_metric "VectorAlive" 1
else
  push_metric "VectorAlive" 0
fi

# 2. Vector systemd status
if systemctl is-active --quiet t12-vector; then
  push_metric "VectorServiceActive" 1
else
  push_metric "VectorServiceActive" 0
fi

# 3. ClickHouse reachable (one-way TLS)
CH_ENDPOINT="${CLICKHOUSE_HTTPS_ENDPOINT:-}"
if [ -n "$CH_ENDPOINT" ]; then
  HTTP_CODE=$(curl -sk -o /dev/null -w '%{http_code}' \
    --cacert "${CLICKHOUSE_CA_CERT:-/etc/t12/ca.crt}" \
    --max-time 5 \
    --header "X-ClickHouse-User: ${CLICKHOUSE_USER}" \
    --header "X-ClickHouse-Key: ${CLICKHOUSE_PASSWORD}" \
    "${CH_ENDPOINT}/?query=SELECT+1" \
    2>/dev/null || echo "000")
  if [ "$HTTP_CODE" = "200" ]; then
    push_metric "ClickHouseReachable" 1
  else
    push_metric "ClickHouseReachable" 0
  fi
else
  push_metric "ClickHouseReachable" 0
fi

# 4. JSONL freshness
NEWEST_LOG=$(find /tmp/training_logs -name "*.jsonl" -type f -printf '%T@\n' 2>/dev/null | sort -rn | head -1)
if [ -n "$NEWEST_LOG" ]; then
  NOW=$(date +%s)
  AGE=$(echo "$NOW - ${NEWEST_LOG%.*}" | bc 2>/dev/null || echo "99999")
  push_metric "JsonlFreshnessSeconds" "$AGE" "Seconds"
else
  push_metric "JsonlFreshnessSeconds" 99999 "Seconds"
fi

# 5. Vector health API
VECTOR_HEALTH=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8686/health 2>/dev/null || echo "000")
if [ "$VECTOR_HEALTH" = "200" ]; then
  push_metric "VectorApiHealthy" 1
else
  push_metric "VectorApiHealthy" 0
fi
HEALTHCHECK

chmod +x /usr/local/bin/t12-training-healthcheck.sh

# Cron: run every minute
echo "* * * * * root /usr/local/bin/t12-training-healthcheck.sh >> /var/log/t12-training-healthcheck.log 2>&1" \
  >/etc/cron.d/t12-training-healthcheck

# ---- 9. Verify ----
echo "[9/9] Verifying..."
sleep 3
echo "Vector status: $(systemctl is-active t12-vector)"

echo "T12 Vector bootstrap completed at $(date -u)"
