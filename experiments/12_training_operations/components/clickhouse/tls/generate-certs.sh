#!/usr/bin/env bash
# Generate self-signed TLS certificates for ClickHouse HTTPS.
# Run once on the DB instance. Certs are mounted into the container.
#
# Usage:  bash generate-certs.sh
# Output: server.crt, server.key, ca.crt (self-signed CA = server cert)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Generating self-signed TLS certificate for ClickHouse..."

openssl req -x509 -newkey rsa:4096 \
  -keyout "$SCRIPT_DIR/server.key" \
  -out "$SCRIPT_DIR/server.crt" \
  -sha256 -days 3650 -nodes \
  -subj "/C=US/ST=Training/L=P12/O=P12Ops/CN=clickhouse" \
  -addext "subjectAltName=DNS:clickhouse,DNS:localhost,IP:127.0.0.1"

# Copy server cert as CA cert (self-signed, so they're the same)
cp "$SCRIPT_DIR/server.crt" "$SCRIPT_DIR/ca.crt"

# ClickHouse needs to read the key
chmod 644 "$SCRIPT_DIR/server.key"

echo "✓ Certificates generated in $SCRIPT_DIR/"
echo "  server.crt  — server certificate (also used as CA cert)"
echo "  server.key  — private key"
echo "  ca.crt      — CA certificate (copy to training instance)"
echo ""
echo "Copy ca.crt to the training instance so Vector can verify the connection:"
echo "  scp $SCRIPT_DIR/ca.crt training-instance:/etc/p12/ca.crt"
