#!/usr/bin/env bash
# Generate a private Certificate Authority for the P12 observability stack.
# Run ONCE. Guard the CA key carefully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CA_DIR="$SCRIPT_DIR/ca"
mkdir -p "$CA_DIR"

if [ -f "$CA_DIR/ca.key" ]; then
  echo "CA already exists at $CA_DIR/ca.key — skipping."
  echo "To regenerate, delete $CA_DIR first."
  exit 0
fi

echo "Generating P12 Certificate Authority..."

# 1. CA private key
openssl genrsa -out "$CA_DIR/ca.key" 4096

# 2. CA certificate (10-year validity)
openssl req -x509 -new -nodes \
  -key "$CA_DIR/ca.key" \
  -sha256 -days 3650 \
  -out "$CA_DIR/ca.crt" \
  -subj "/C=US/ST=Training/L=P12/O=P12Ops/OU=CA/CN=P12 Training CA"

# 3. Lock down the CA key
chmod 400 "$CA_DIR/ca.key"

echo "✓ CA generated:"
echo "  $CA_DIR/ca.key  — GUARD THIS. Never distribute."
echo "  $CA_DIR/ca.crt  — Distribute to all clients and the server."
