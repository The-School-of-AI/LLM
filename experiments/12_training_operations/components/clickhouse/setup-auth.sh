#!/usr/bin/env bash
# =============================================================================
# P12 ClickHouse Auth Setup
#
# Generates:
#   1. TLS certificates (proper CA → server cert, one-way TLS)
#   2. ClickHouse users XML with password hashes + CIDR restrictions
#   3. Uploads CA cert + vector.toml to S3 config bucket
#   4. Stores passwords + endpoint in SSM Parameter Store
#
# Usage:
#   bash setup-auth.sh
#
# You can pre-set values via env vars to skip prompts:
#   P12_WRITER_PASSWORD=... P12_READER_PASSWORD=... \
#   TRAINING_SUBNET_CIDR=10.0.1.0/24 DASHBOARD_SUBNET_CIDR=10.0.2.0/24 \
#   DB_PRIVATE_IP=10.0.1.5 P12_REGION=us-east-1 \
#   bash setup-auth.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  P12 ClickHouse Auth Setup"
echo "============================================================"
echo ""

# ---- 1. Collect inputs ----

if [ -z "${P12_WRITER_PASSWORD:-}" ]; then
  read -sp "Enter password for p12_writer (Vector/training): " P12_WRITER_PASSWORD
  echo ""
fi

if [ -z "${P12_READER_PASSWORD:-}" ]; then
  read -sp "Enter password for p12_reader (dashboard/queries): " P12_READER_PASSWORD
  echo ""
fi

if [ ${#P12_WRITER_PASSWORD} -lt 12 ]; then
  echo "ERROR: p12_writer password must be at least 12 characters."
  exit 1
fi

if [ ${#P12_READER_PASSWORD} -lt 12 ]; then
  echo "ERROR: p12_reader password must be at least 12 characters."
  exit 1
fi

if [ -z "${TRAINING_SUBNET_CIDR:-}" ]; then
  read -p "Enter training subnet CIDR (e.g. 10.0.1.0/24): " TRAINING_SUBNET_CIDR
fi

if [ -z "${DASHBOARD_SUBNET_CIDR:-}" ]; then
  read -p "Enter dashboard subnet CIDR (e.g. 10.0.2.0/24): " DASHBOARD_SUBNET_CIDR
fi

if [ -z "${DB_PRIVATE_IP:-}" ]; then
  read -p "Enter DB instance private IP (e.g. 10.0.1.5): " DB_PRIVATE_IP
fi

P12_REGION="${P12_REGION:-us-east-1}"

# ---- 2. Generate SHA256 password hashes ----

WRITER_HASH=$(printf '%s' "$P12_WRITER_PASSWORD" | sha256sum | cut -d' ' -f1)
READER_HASH=$(printf '%s' "$P12_READER_PASSWORD" | sha256sum | cut -d' ' -f1)

echo "✓ Password hashes generated"

# ---- 3. Generate users XML from template (with CIDR restrictions) ----

TEMPLATE="$SCRIPT_DIR/users.d/p12-users.xml.template"
OUTPUT="$SCRIPT_DIR/users.d/p12-users.xml"

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: Template not found: $TEMPLATE"
  exit 1
fi

sed -e "s/__WRITER_HASH__/$WRITER_HASH/g" \
    -e "s/__READER_HASH__/$READER_HASH/g" \
    -e "s|__TRAINING_SUBNET_CIDR__|$TRAINING_SUBNET_CIDR|g" \
    -e "s|__DASHBOARD_SUBNET_CIDR__|$DASHBOARD_SUBNET_CIDR|g" \
    "$TEMPLATE" > "$OUTPUT"

echo "✓ Users config written to $OUTPUT"

# ---- 4. Generate TLS certificates (proper CA → server cert) ----

TLS_DIR="$SCRIPT_DIR/tls"

# 4a. CA (idempotent — won't regenerate if ca.key exists)
bash "$TLS_DIR/generate-ca.sh"

# 4b. Server cert signed by CA
if [ -f "$TLS_DIR/server.crt" ] && [ -f "$TLS_DIR/server.key" ]; then
  echo "✓ Server certificate already exists (skipping)"
else
  bash "$TLS_DIR/generate-server-cert.sh" "$DB_PRIVATE_IP"
fi

# ---- 5. Create S3 config bucket (idempotent) and upload certs + config ----

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="p12-training-ops-base-${ACCOUNT_ID}"

# Create bucket if it doesn't exist (one-time operation, shared by all systems)
if ! aws s3 ls "s3://${BUCKET}" &>/dev/null; then
  aws s3 mb "s3://${BUCKET}" --region "$P12_REGION"

  # Allow public reads — ca.crt and vector.toml are non-sensitive config
  aws s3api put-public-access-block \
    --bucket "$BUCKET" \
    --public-access-block-configuration \
      "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

  aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Sid\": \"PublicReadConfig\",
      \"Effect\": \"Allow\",
      \"Principal\": \"*\",
      \"Action\": \"s3:GetObject\",
      \"Resource\": \"arn:aws:s3:::${BUCKET}/*\"
    }]
  }"
  echo "✓ Created public S3 config bucket: s3://${BUCKET}"
else
  echo "✓ S3 config bucket already exists: s3://${BUCKET}"
fi

# Upload CA cert (clients verify the server with this)
aws s3 cp "$TLS_DIR/ca/ca.crt" "s3://${BUCKET}/certs/ca.crt"
echo "✓ Uploaded ca.crt to s3://${BUCKET}/certs/ca.crt"

# Upload vector.toml
if [ -f "$SCRIPT_DIR/../sidecar_agent/vector.toml" ]; then
  aws s3 cp "$SCRIPT_DIR/../sidecar_agent/vector.toml" "s3://${BUCKET}/vector/vector.toml"
  echo "✓ Uploaded vector.toml to s3://${BUCKET}/vector/vector.toml"
fi

echo "✓ Public URLs:"
echo "  https://${BUCKET}.s3.amazonaws.com/certs/ca.crt"
echo "  https://${BUCKET}.s3.amazonaws.com/vector/vector.toml"

# ---- 6. Store credentials in SSM Parameter Store ----

aws ssm put-parameter \
  --name "/p12/training/clickhouse-password" \
  --value "$P12_WRITER_PASSWORD" \
  --type SecureString \
  --overwrite \
  --region "$P12_REGION" >/dev/null

aws ssm put-parameter \
  --name "/p12/dashboard/clickhouse-password" \
  --value "$P12_READER_PASSWORD" \
  --type SecureString \
  --overwrite \
  --region "$P12_REGION" >/dev/null

aws ssm put-parameter \
  --name "/p12/training/clickhouse-endpoint" \
  --value "https://${DB_PRIVATE_IP}:8443" \
  --type String \
  --overwrite \
  --region "$P12_REGION" >/dev/null

echo "✓ Credentials stored in SSM Parameter Store"

# ---- 7. Write local .env files (for reference/manual use) ----

ENV_FILE="$SCRIPT_DIR/training-instance.env"
cat > "$ENV_FILE" <<EOF
# P12 Training Instance Environment Variables
# These are also in SSM Parameter Store — this file is for reference only.
CLICKHOUSE_HTTPS_ENDPOINT=https://${DB_PRIVATE_IP}:8443
CLICKHOUSE_USER=p12_writer
CLICKHOUSE_PASSWORD=$P12_WRITER_PASSWORD
CLICKHOUSE_CA_CERT=/etc/p12/ca.crt
EOF

DASH_ENV_FILE="$SCRIPT_DIR/dashboard.env"
cat > "$DASH_ENV_FILE" <<EOF
# P12 Dashboard Environment Variables (read-only access)
CLICKHOUSE_HTTPS_ENDPOINT=https://${DB_PRIVATE_IP}:8443
CLICKHOUSE_USER=p12_reader
CLICKHOUSE_PASSWORD=$P12_READER_PASSWORD
CLICKHOUSE_CA_CERT=/etc/p12/ca.crt
EOF

echo "✓ Local .env files written (for reference)"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  S3 bucket: s3://${BUCKET}"
echo "  SSM params: /p12/training/*, /p12/dashboard/*"
echo ""
echo "  IMPORTANT: Do NOT commit .env files, p12-users.xml, or TLS certs to git."
echo ""
