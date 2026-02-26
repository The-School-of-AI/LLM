#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TEST_ROOT/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
CODE_DIR="$TEST_ROOT/code"
CFG="$TEST_ROOT/configs/test14_gsa_only_liger_kernels_1000steps.yaml"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-1}"

# --- Observability bootstrap (local, no S3/Secrets) ---
LOCAL_ASSETS_DIR="$TEST_ROOT/local"
CA_SRC="${CA_SRC:-$LOCAL_ASSETS_DIR/ca_clickhouse.crt}"
ENV_SRC="${ENV_SRC:-$LOCAL_ASSETS_DIR/vector.env}"
VECTOR_TOML_SRC="${VECTOR_TOML_SRC:-$REPO_ROOT/experiments/12_training_operations/components/sidecar_agent/vector.toml}"

# Ensure curl available (for Vector installer)
if ! command -v curl >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq curl
fi

# Install Vector if missing
if ! command -v vector >/dev/null 2>&1; then
  export HOME="${HOME:-/root}"
  curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local
fi

# Prepare directories
sudo mkdir -p /etc/t12 /var/lib/vector /tmp/training_logs
sudo chown "$(whoami)":"$(id -gn)" /tmp/training_logs || true

# Require local CA and env files for this dry run
if [[ ! -f "$CA_SRC" ]]; then
  echo "[ERROR] Missing CA cert: $CA_SRC" >&2; exit 1
fi
if [[ ! -f "$ENV_SRC" ]]; then
  echo "[ERROR] Missing env file: $ENV_SRC" >&2; exit 1
fi

# Copy artifacts into place
sudo install -m 0644 "$CA_SRC" /etc/t12/ca.crt
sudo install -m 0644 "$VECTOR_TOML_SRC" /etc/t12/vector.toml
sudo install -m 0600 "$ENV_SRC" /etc/t12/vector.env
cp /etc/t12/vector.env "$HOME/.t12.env" && chmod 600 "$HOME/.t12.env"

# Create/refresh systemd unit
sudo tee /etc/systemd/system/t12-vector.service >/dev/null <<'UNIT'
[Unit]
Description=T12 Vector Sidecar
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

[Install]
WantedBy=multi-user.target
UNIT

if command -v systemctl >/dev/null 2>&1; then
  sudo systemctl daemon-reload
  sudo systemctl enable --now t12-vector || sudo systemctl restart t12-vector
else
  echo "[WARN] systemd not found; starting Vector in background (no preflight)"
  nohup /usr/local/bin/vector --config /etc/t12/vector.toml --data-dir /var/lib/vector \
    >/var/log/vector.log 2>&1 &
  export SKIP_VECTOR_CHECK=1
fi

# Export runtime env for training
set -a; [ -f "$HOME/.t12.env" ] && source "$HOME/.t12.env"; set +a
export VECTOR_SERVICE_NAME="t12-vector.service"
export SKIP_VECTOR_CHECK="${SKIP_VECTOR_CHECK:-0}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model for Test 14..."
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init model: $INIT_CKPT"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Test 14 (GSA-only, Liger RoPE/MLP/fused CE, no DeltaNet, 1000 steps)..."
(
  cd "$CODE_DIR"
  "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
) 2>&1 | tee "$RESULTS_DIR/run/train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 14 completed"
echo "  Init model: $INIT_CKPT"
echo "  Train log:  $RESULTS_DIR/run/train.log"
echo "  Metrics:    $RESULTS_DIR/run/metrics.jsonl"
