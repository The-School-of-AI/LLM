#!/usr/bin/env bash
# =============================================================================
# P12 Vector Sidecar — EC2 User Data Bootstrap
#
# Works on any Ubuntu AMI (18.04+). Installs Vector, pulls CA cert and config
# from S3, retrieves credentials from SSM Parameter Store, and starts Vector
# as a systemd service.
#
# Prerequisites:
#   - EC2 instance must have the p12-training-instance-profile IAM role attached
#   - S3 bucket and SSM parameters must be populated (via setup-auth.sh)
#
# Customize these 3 variables before use:
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/p12-userdata.log) 2>&1
echo "P12 Vector bootstrap started at $(date -u)"

# ---- Configuration (EDIT THESE) ----
P12_CONFIG_BUCKET="p12-training-ops-base-869633161654"  # replace with your S3 bucket name
AWS_REGION="${AWS_REGION:-us-east-1}"

# ---- 1. System packages ----
echo "[1/8] Installing dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq awscli jq curl bc


# ---- Configuration (EDIT THESE) ----
P12_CONFIG_BUCKET="p12-training-ops-base-869633161654"  # replace with your S3 bucket name
AWS_REGION="${AWS_REGION:-us-east-1}"
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming

# ---- 2. Install Vector ----
echo "[2/8] Installing Vector..."
if ! command -v vector &>/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local
fi
/usr/local/bin/vector --version

# ---- 3. Create directories ----
echo "[3/8] Creating directories..."
mkdir -p /etc/p12
mkdir -p /tmp/training_logs
mkdir -p /var/lib/vector
chown ubuntu:ubuntu /tmp/training_logs
chown ubuntu:ubuntu /var/lib/vector

# ---- 4. Pull CA cert + Vector config from S3 (public URLs, no IAM needed) ----
echo "[4/8] Pulling config from S3..."
curl -fsSL "https://${P12_CONFIG_BUCKET}.s3.amazonaws.com/certs/ca_clickhouse.crt" -o /etc/p12/ca.crt
curl -fsSL "https://raw.githubusercontent.com/The-School-of-AI/LLM/refs/heads/P12/feat/training-ops-base/experiments/12_training_operations/components/sidecar_agent/vector.toml" -o /etc/p12/vector.toml
chmod 644 /etc/p12/ca.crt

# ---- 5. Pull credentials from SSM Parameter Store ----
echo "[5/8] Retrieving credentials from SSM..."
CH_PASSWORD=$(aws ssm get-parameter \
  --name "/$PREFIX/clickhouse/writer-password" \
  --with-decryption \
  --region "$P12_REGION" \
  --query 'Parameter.Value' --output text)

CH_ENDPOINT=$(aws ssm get-parameter \
  \"Name\": \"/$PREFIX/clickhouse/endpoint\",
  --region "$P12_REGION" \
  --query 'Parameter.Value' --output text)

# Write environment file (read by systemd + training process)
cat > /etc/p12/vector.env <<EOF
CLICKHOUSE_HTTPS_ENDPOINT=${CH_ENDPOINT}
CLICKHOUSE_USER=p12_writer
CLICKHOUSE_PASSWORD=${CH_PASSWORD}
CLICKHOUSE_CA_CERT=/etc/p12/ca.crt
EOF
chmod 600 /etc/p12/vector.env

# Copy for the training process (ubuntu user)
cp /etc/p12/vector.env /home/ubuntu/.p12.env
chown ubuntu:ubuntu /home/ubuntu/.p12.env
chmod 600 /home/ubuntu/.p12.env

# ---- 6. Create Vector systemd service ----
echo "[6/8] Creating Vector systemd service..."
cat > /etc/systemd/system/p12-vector.service <<'UNIT'
[Unit]
Description=P12 Vector Sidecar (ClickHouse log shipper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
EnvironmentFile=/etc/p12/vector.env
ExecStart=/usr/local/bin/vector --config /etc/p12/vector.toml --data-dir /var/lib/vector
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=p12-vector

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/vector /tmp/training_logs

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable p12-vector
systemctl start p12-vector

# ---- 7. Install training-side health check ----
echo "[7/8] Installing health check..."
cat > /usr/local/bin/p12-training-healthcheck.sh <<'HEALTHCHECK'
#!/usr/bin/env bash
set -uo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
NAMESPACE="P12/Training"
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")

if [ -f /etc/p12/vector.env ]; then
  set -a; source /etc/p12/vector.env; set +a
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
if systemctl is-active --quiet p12-vector; then
  push_metric "VectorServiceActive" 1
else
  push_metric "VectorServiceActive" 0
fi

# 3. ClickHouse reachable (one-way TLS)
CH_ENDPOINT="${CLICKHOUSE_HTTPS_ENDPOINT:-}"
if [ -n "$CH_ENDPOINT" ]; then
  HTTP_CODE=$(curl -sk -o /dev/null -w '%{http_code}' \
    --cacert "${CLICKHOUSE_CA_CERT:-/etc/p12/ca.crt}" \
    --max-time 5 \
    "${CH_ENDPOINT}/?user=${CLICKHOUSE_USER}&password=${CLICKHOUSE_PASSWORD}&query=SELECT+1" \
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

chmod +x /usr/local/bin/p12-training-healthcheck.sh

# Cron: run every minute
echo "* * * * * root /usr/local/bin/p12-training-healthcheck.sh >> /var/log/p12-training-healthcheck.log 2>&1" \
  > /etc/cron.d/p12-training-healthcheck

# ---- 8. Verify ----
echo "[8/8] Verifying..."
sleep 3
echo "Vector status: $(systemctl is-active p12-vector)"

echo "P12 Vector bootstrap completed at $(date -u)"