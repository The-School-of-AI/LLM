#!/usr/bin/env bash
# =============================================================================
# P12 ClickHouse Auth Setup
#
# Generates:
#   1. TLS certificates (self-signed) for HTTPS
#   2. ClickHouse users XML with password hashes
#   3. .env file with credentials for the training instance
#
# Usage:
#   bash setup-auth.sh
#
# You will be prompted for passwords, or you can set them via env vars:
#   P12_WRITER_PASSWORD=... P12_READER_PASSWORD=... bash setup-auth.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  P12 ClickHouse Auth Setup"
echo "============================================================"
echo ""

# ---- 1. Passwords ----

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

# ---- 2. Generate SHA256 hashes ----

WRITER_HASH=$(printf '%s' "$P12_WRITER_PASSWORD" | sha256sum | cut -d' ' -f1)
READER_HASH=$(printf '%s' "$P12_READER_PASSWORD" | sha256sum | cut -d' ' -f1)

echo ""
echo "✓ Password hashes generated"

# ---- 3. Generate users XML from template ----

TEMPLATE="$SCRIPT_DIR/users.d/p12-users.xml.template"
OUTPUT="$SCRIPT_DIR/users.d/p12-users.xml"

if [ ! -f "$TEMPLATE" ]; then
  echo "ERROR: Template not found: $TEMPLATE"
  exit 1
fi

sed -e "s/__WRITER_HASH__/$WRITER_HASH/g" \
    -e "s/__READER_HASH__/$READER_HASH/g" \
    "$TEMPLATE" > "$OUTPUT"

echo "✓ Users config written to $OUTPUT"

# ---- 4. Generate TLS certificates ----

TLS_DIR="$SCRIPT_DIR/tls"
if [ -f "$TLS_DIR/server.crt" ] && [ -f "$TLS_DIR/server.key" ]; then
  echo "✓ TLS certificates already exist (skipping generation)"
else
  bash "$TLS_DIR/generate-certs.sh"
fi

# ---- 5. Write .env file for the training instance ----

ENV_FILE="$SCRIPT_DIR/training-instance.env"
cat > "$ENV_FILE" <<EOF
# P12 Training Instance Environment Variables
# Copy this file to the training instance and source it before starting Vector/training.
#
# Usage:
#   export \$(cat training-instance.env | grep -v '^#' | xargs)

# ClickHouse connection (HTTPS)
CLICKHOUSE_HTTPS_ENDPOINT=https://<DB_INSTANCE_IP>:8443
CLICKHOUSE_USER=p12_writer
CLICKHOUSE_PASSWORD=$P12_WRITER_PASSWORD
CLICKHOUSE_CA_CERT=/etc/p12/ca.crt
EOF

echo "✓ Training instance env file written to $ENV_FILE"

# ---- 6. Write .env file for the dashboard ----

DASH_ENV_FILE="$SCRIPT_DIR/dashboard.env"
cat > "$DASH_ENV_FILE" <<EOF
# P12 Dashboard Environment Variables
# For read-only access to ClickHouse.

CLICKHOUSE_HTTPS_ENDPOINT=https://<DB_INSTANCE_IP>:8443
CLICKHOUSE_USER=p12_reader
CLICKHOUSE_PASSWORD=$P12_READER_PASSWORD
CLICKHOUSE_CA_CERT=/etc/p12/ca.crt
EOF

echo "✓ Dashboard env file written to $DASH_ENV_FILE"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start ClickHouse:"
echo "     cd $SCRIPT_DIR && sudo docker compose up -d"
echo ""
echo "  2. Copy to training instance:"
echo "     scp $ENV_FILE training-instance:~/.p12.env"
echo "     scp $TLS_DIR/ca.crt training-instance:/etc/p12/ca.crt"
echo ""
echo "  3. On the training instance, source the env before running:"
echo "     export \$(cat ~/.p12.env | grep -v '^#' | xargs)"
echo ""
echo "  4. Copy to dashboard instance:"
echo "     scp $DASH_ENV_FILE dashboard-instance:~/.p12-dashboard.env"
echo "     scp $TLS_DIR/ca.crt dashboard-instance:/etc/p12/ca.crt"
echo ""
echo "  IMPORTANT: Do NOT commit .env files, p12-users.xml, or TLS certs to git."
echo "             These are already in .gitignore."
echo ""
