#!/usr/bin/env bash
# =============================================================================
# P12 Vector Sidecar — EC2 User Data Bootstrap
#
# Works on any Ubuntu AMI (18.04+). Installs Vector, pulls CA cert and config
# from S3, retrieves ClickHouse credentials from SSM Parameter Store via
# cross-account assume-role, and starts Vector as a systemd service.
#
# Architecture:
#   Training instances run in Account A. SSM parameters (ClickHouse creds)
#   live in Account B (infra account). The script assumes a cross-account
#   role to read SSM, then drops those credentials so CloudWatch health
#   metrics are pushed to Account A using the instance profile.
#
# Prerequisites (all must be satisfied BEFORE running):
#
#   1. AMI:     Ubuntu 18.04+ (tested on 22.04 LTS)
#
#   2. IAM (Training Account):
#               Attach instance profile with training-instance-iam-policy.json
#               Grants: sts:AssumeRole to SSM account + cloudwatch:PutMetricData
#
#   3. IAM (SSM/Infra Account):
#               Create t12-ssm-reader role from ssm-reader-cross-account-role.json
#               Grants: ssm:GetParameter + kms:Decrypt for T12 parameters
#               Trust:  the training account's instance profile role
#
#   4. SSM:     Parameters pre-populated in the SSM/infra account:
#                 /$PREFIX/clickhouse/writer-password  (SecureString)
#                 /$PREFIX/clickhouse/endpoint          (String)
#               See Post_Clickhouse_Install.md Section 3.
#
#   5. S3:      CA cert uploaded to s3://<bucket>/certs/ca_clickhouse.crt
#               Bucket is public-read — no IAM needed.
#
#   6. SG:      Security group allows outbound:
#                 - HTTPS (443): S3, SSM, STS, CloudWatch, github.com
#                 - TCP 8443:    ClickHouse endpoint
#
# Customize the 4 variables in the "Configuration" section below.
#
# Execution modes:
#   1. As EC2 userdata (first boot):
#      aws ec2 run-instances --user-data file://userdata_vector.sh ...
#
#   2. Manually on a running instance:
#      sudo bash userdata_vector.sh
#
#   3. Re-run (update config/creds on instance where this already ran):
#      sudo bash userdata_vector.sh
#      (All steps are idempotent. Vector will be restarted with fresh config.)
#
# Logs:  cat /var/log/t12-userdata.log
#
# Deployment (one-time setup):
#
#   Step 1 — SSM/Infra Account (Account B): create cross-account reader role
#
#     aws iam create-role \
#       --role-name t12-ssm-reader \
#       --assume-role-policy-document '{
#         "Version": "2012-10-17",
#         "Statement": [{
#           "Effect": "Allow",
#           "Principal": {
#             "AWS": "arn:aws:iam::<TRAINING_ACCOUNT_ID>:role/t12-traininginstance-239-role"
#           },
#           "Action": "sts:AssumeRole"
#         }]
#       }'
#
#     aws iam put-role-policy \
#       --role-name t12-ssm-reader \
#       --policy-name t12-ssm-read-access \
#       --policy-document file://ssm-reader-cross-account-role.json
#       # (use the PermissionsPolicy object from that file)
#
#   Step 2 — Training Account (Account A): add AssumeRole + CloudWatch to instance role
#   aws iam create-role \
#  --role-name t12-traininginstance-239-role \
#  --assume-role-policy-document '{
#    "Version": "2012-10-17",
#    "Statement": [{
#      "Effect": "Allow",
#      "Principal": {
#        "Service": "ec2.amazonaws.com"
#      },
#      "Action": "sts:AssumeRole"
#    }]
#  }'
#
#     aws iam put-role-policy \
#       --role-name t12-traininginstance-239-role \
#       --policy-name t12-cross-account-ssm \
#       --policy-document file://training-instance-iam-policy.json
#
#   Step 3 — Edit the 4 configuration variables below.
#
#   Step 4 — Run: pass as userdata for new instances, or sudo bash on existing.
#
# =============================================================================

set -euo pipefail
exec > >(tee /var/log/t12-userdata.log) 2>&1
echo "T12 Vector bootstrap started at $(date -u)"

# ---- 1. System packages ----
echo "[1/9] Installing dependencies..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq awscli jq curl bc


# ---- Configuration (EDIT THESE) ----
P12_CONFIG_BUCKET="p12-training-ops-base-869633161654"  # replace with your S3 bucket name
AWS_REGION="${AWS_REGION:-us-east-1}"
PREFIX="T12-TrainingOperations-239" # REPLACE with your unique prefix for resource naming
SSM_ROLE_ARN="arn:aws:iam::<SSM_ACCOUNT_ID>:role/t12-ssm-reader"  # cross-account role for SSM access

# ---- 2. Install Vector ----
echo "[2/9] Installing Vector..."
if ! command -v vector &>/dev/null; then
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
curl -fsSL "https://${P12_CONFIG_BUCKET}.s3.amazonaws.com/certs/ca_clickhouse.crt" -o /etc/t12/ca.crt
curl -fsSL "https://raw.githubusercontent.com/The-School-of-AI/LLM/refs/heads/P12/feat/training-ops-base/experiments/12_training_operations/components/sidecar_agent/vector.toml" -o /etc/t12/vector.toml
chmod 644 /etc/t12/ca.crt

# ---- 5. Assume cross-account role for SSM access ----
echo "[5/9] Assuming cross-account role for SSM..."
CREDS_JSON=$(aws sts assume-role \
  --role-arn "$SSM_ROLE_ARN" \
  --role-session-name "t12-vector-$(hostname -s)" \
  --duration-seconds 900 \
  --output json)

export AWS_ACCESS_KEY_ID=$(echo "$CREDS_JSON" | jq -r '.Credentials.AccessKeyId')
export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS_JSON" | jq -r '.Credentials.SecretAccessKey')
export AWS_SESSION_TOKEN=$(echo "$CREDS_JSON" | jq -r '.Credentials.SessionToken')

# ---- 6. Pull credentials from SSM Parameter Store (using cross-account creds) ----
echo "[6/9] Retrieving credentials from SSM..."
CH_PASSWORD=$(aws ssm get-parameter \
  --name "/$PREFIX/clickhouse/writer-password" \
  --with-decryption \
  --region "$AWS_REGION" \
  --query 'Parameter.Value' --output text)

CH_ENDPOINT=$(aws ssm get-parameter \
  --name "/$PREFIX/clickhouse/endpoint" \
  --region "$AWS_REGION" \
  --query 'Parameter.Value' --output text)

# Drop assumed credentials — back to instance profile for CloudWatch etc.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

# Write environment file (read by systemd + training process)
cat > /etc/t12/vector.env <<EOF
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
cat > /etc/systemd/system/t12-vector.service <<'UNIT'
[Unit]
Description=P12 Vector Sidecar (ClickHouse log shipper)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
EnvironmentFile=/etc/t12/vector.env
ExecStart=/usr/local/bin/vector --config /etc/t12/vector.toml --data-dir /var/lib/vector
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
cat > /usr/local/bin/t12-training-healthcheck.sh <<'HEALTHCHECK'
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
  > /etc/cron.d/t12-training-healthcheck

# ---- 9. Verify ----
echo "[9/9] Verifying..."
sleep 3
echo "Vector status: $(systemctl is-active t12-vector)"

echo "P12 Vector bootstrap completed at $(date -u)"