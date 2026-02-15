#!/usr/bin/env bash
# Generate self-signed TLS certificates for ClickHouse HTTPS.
# Run once on the DB instance. Certs are mounted into the container.
#
# Usage:  bash generate-certs.sh
# Output: server.crt, server.key, ca.crt (self-signed CA = server cert)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CERT_CN="${CLICKHOUSE_CERT_CN:-clickhouse}"
CA_CN="${CLICKHOUSE_CA_CN:-${CERT_CN}-ca}"
SERVER_CN="${CLICKHOUSE_SERVER_CN:-${CERT_CN}}"
CERT_SANS="${CLICKHOUSE_CERT_SANS:-DNS:clickhouse,DNS:localhost,IP:127.0.0.1}"

echo "Generating self-signed TLS certificate for ClickHouse..."

CA_KEY="$SCRIPT_DIR/ca.key"
CA_CRT="$SCRIPT_DIR/ca.crt"
SERVER_KEY="$SCRIPT_DIR/server.key"
SERVER_CSR="$SCRIPT_DIR/server.csr"
SERVER_CRT="$SCRIPT_DIR/server.crt"

openssl req -x509 -new -nodes -newkey rsa:4096 \
  -keyout "$CA_KEY" \
  -out "$CA_CRT" \
  -sha256 -days 3650 \
  -subj "/C=US/ST=Training/L=P12/O=P12Ops/CN=${CA_CN}" \
  -addext "basicConstraints=critical,CA:TRUE" \
  -addext "keyUsage=critical,keyCertSign,cRLSign"

openssl req -new -nodes -newkey rsa:4096 \
  -keyout "$SERVER_KEY" \
  -out "$SERVER_CSR" \
  -sha256 \
  -subj "/C=US/ST=Training/L=P12/O=P12Ops/CN=${SERVER_CN}" \
  -addext "subjectAltName=${CERT_SANS}"

EXT_FILE="$(mktemp)"
cat >"$EXT_FILE" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=${CERT_SANS}
EOF

openssl x509 -req \
  -in "$SERVER_CSR" \
  -CA "$CA_CRT" \
  -CAkey "$CA_KEY" \
  -CAcreateserial \
  -out "$SERVER_CRT" \
  -days 3650 \
  -sha256 \
  -extfile "$EXT_FILE"

rm -f "$EXT_FILE" "$SERVER_CSR" "$SCRIPT_DIR/ca.srl"

chmod 600 "$CA_KEY"
chmod 644 "$CA_CRT" "$SERVER_CRT"
chmod 644 "$SERVER_KEY"

echo "✓ Certificates generated in $SCRIPT_DIR/"
echo "  server.crt  — server certificate (also used as CA cert)"
echo "  server.key  — private key"
echo "  ca.crt      — CA certificate (copy to training instance)"
echo ""
echo "Copy ca.crt to the training instance so Vector can verify the connection:"
echo "  scp $SCRIPT_DIR/ca.crt training-instance:/etc/p12/ca.crt"
