#!/usr/bin/env bash
# Generate a server certificate for ClickHouse, signed by the P12 CA.
# Requires: generate-ca.sh has been run first.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CA_DIR="$SCRIPT_DIR/ca"
SERVER_DIR="$SCRIPT_DIR"

if [ ! -f "$CA_DIR/ca.key" ]; then
  echo "ERROR: CA not found. Run generate-ca.sh first."
  exit 1
fi

# Accept DB instance private IP as argument or prompt
DB_PRIVATE_IP="${1:-}"
if [ -z "$DB_PRIVATE_IP" ]; then
  read -p "Enter DB instance private IP (e.g. 10.0.1.5): " DB_PRIVATE_IP
fi

echo "Generating ClickHouse server certificate..."

# 1. Server private key
openssl genrsa -out "$SERVER_DIR/server.key" 4096

# 2. CSR
openssl req -new \
  -key "$SERVER_DIR/server.key" \
  -out "$SERVER_DIR/server.csr" \
  -subj "/C=US/ST=Training/L=P12/O=P12Ops/OU=ClickHouse/CN=clickhouse-server"

# 3. Extensions file (SANs — include the DB private IP so TLS hostname verification works)
cat > "$SERVER_DIR/server_ext.cnf" <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:clickhouse,DNS:localhost,IP:127.0.0.1,IP:${DB_PRIVATE_IP}
EOF

# 4. Sign with CA
openssl x509 -req \
  -in "$SERVER_DIR/server.csr" \
  -CA "$CA_DIR/ca.crt" \
  -CAkey "$CA_DIR/ca.key" \
  -CAcreateserial \
  -out "$SERVER_DIR/server.crt" \
  -days 825 \
  -sha256 \
  -extfile "$SERVER_DIR/server_ext.cnf"

# 5. Copy CA cert to the expected location (ClickHouse config references tls/ca.crt)
cp "$CA_DIR/ca.crt" "$SERVER_DIR/ca.crt"

# 6. Permissions
chmod 644 "$SERVER_DIR/server.key"

# 7. Cleanup temp files
rm -f "$SERVER_DIR/server.csr" "$SERVER_DIR/server_ext.cnf"

echo "✓ Server certificate generated:"
echo "  $SERVER_DIR/server.crt  — signed by P12 CA"
echo "  $SERVER_DIR/server.key  — server private key"
echo "  $SERVER_DIR/ca.crt      — CA cert (distribute to clients)"
