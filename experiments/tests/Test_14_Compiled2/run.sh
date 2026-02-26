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

# Resolve DeepSpeed binary/module from the Python environment being used
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] PYTHON_BIN not found/executable: $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v "$DEEPSPEED_BIN" >/dev/null 2>&1; then
  _DS_FROM_PY="$($PYTHON_BIN -c 'import shutil; print(shutil.which("deepspeed") or "")' 2>/dev/null || true)"
  if [[ -n "${_DS_FROM_PY:-}" ]]; then
    DEEPSPEED_BIN="$_DS_FROM_PY"
  elif "$PYTHON_BIN" -c 'import deepspeed' >/dev/null 2>&1; then
    DEEPSPEED_BIN=("$PYTHON_BIN" -m deepspeed)
  else
    echo "[ERROR] DeepSpeed not found. Neither '$DEEPSPEED_BIN' is on PATH, nor is 'deepspeed' importable from $PYTHON_BIN." >&2
    echo "        Install it in the same environment as $PYTHON_BIN (e.g. pip install -r $REPO_ROOT/requirements.txt)." >&2
    exit 1
  fi
fi

# --- Logging & diagnostics ---
exec > >(tee -a "$TEST_ROOT/run_bootstrap.log") 2>&1
echo "[run.sh] Started at $(date -u)"
set -x

die() { echo "[ERROR] $*" >&2; exit 1; }
check_file() { local p="$1"; [[ -f "$p" ]] || die "Expected file not found: $p"; }
http_code() { curl -s -o /dev/null -w "%{http_code}" "$1"; }

# --- Optional vendoring: bring 12_training_operations into experiments/ ---
TRAINING_OPS_DEST="$REPO_ROOT/experiments/12_training_operations"
if [[ ! -d "$TRAINING_OPS_DEST/components" || -z "$(ls -A "$TRAINING_OPS_DEST" 2>/dev/null)" ]]; then
  if [[ -n "${TRAINING_OPS_SRC:-}" && -d "$TRAINING_OPS_SRC" ]]; then
    echo "[INFO] Vendoring TrainingOps from: $TRAINING_OPS_SRC -> $TRAINING_OPS_DEST"
    sudo mkdir -p "$TRAINING_OPS_DEST"
    if command -v rsync >/dev/null 2>&1; then
      sudo rsync -a --delete "$TRAINING_OPS_SRC"/ "$TRAINING_OPS_DEST"/
    else
      sudo cp -a "$TRAINING_OPS_SRC"/. "$TRAINING_OPS_DEST"/
    fi
  else
    echo "[WARN] experiments/12_training_operations missing or empty."
    echo "      Set TRAINING_OPS_SRC=/path/to/staging/12_training_operations and re-run,"
    echo "      or provide $LOCAL_ASSETS_DIR/vector.toml and ENV_SRC for local-only mode."
  fi
fi

# Ensure sudo can run (will prompt once if needed)
if ! sudo -n true 2>/dev/null; then
  echo "[INFO] Caching sudo credentials (you may be prompted)";
  sudo -v || die "sudo is required to install Vector and write /etc/t12";
fi

# --- Observability bootstrap (local, no S3/Secrets) ---
LOCAL_ASSETS_DIR="$TEST_ROOT/local"
CA_SRC="${CA_SRC:-$LOCAL_ASSETS_DIR/ca_clickhouse.crt}"
ENV_SRC="${ENV_SRC:-$LOCAL_ASSETS_DIR/vector.env}"
# Prefer a local vector.toml if present; else use repo path; else error
_VEC_LOCAL="$LOCAL_ASSETS_DIR/vector.toml"
_VEC_REPO="$REPO_ROOT/experiments/12_training_operations/components/sidecar_agent/vector.toml"
if [[ -n "${VECTOR_TOML_SRC:-}" ]]; then
  VECTOR_TOML_SRC="$VECTOR_TOML_SRC"
elif [[ -f "$_VEC_LOCAL" ]]; then
  VECTOR_TOML_SRC="$_VEC_LOCAL"
elif [[ -f "$_VEC_REPO" ]]; then
  VECTOR_TOML_SRC="$_VEC_REPO"
else
  VECTOR_TOML_SRC=""
fi
echo "[INFO] Using sources:"
echo "       CA_SRC=$CA_SRC"
echo "       ENV_SRC=$ENV_SRC"
echo "       VECTOR_TOML_SRC=${VECTOR_TOML_SRC:-<not found>}"

# Ensure curl available (for Vector installer)
if ! command -v curl >/dev/null 2>&1; then
  sudo apt-get update -qq && sudo apt-get install -y -qq curl
fi

# Install Vector if missing
if ! command -v vector >/dev/null 2>&1; then
  export HOME="${HOME:-/root}"
  curl --proto '=https' --tlsv1.2 -sSfL https://sh.vector.dev | bash -s -- -y --prefix /usr/local
fi
echo "[INFO] Vector binary: $(command -v vector || echo 'not found')"
if command -v vector >/dev/null 2>&1; then /usr/local/bin/vector --version || vector --version || true; fi

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
if [[ -z "$VECTOR_TOML_SRC" ]]; then
  die "Vector config not found. Provide $LOCAL_ASSETS_DIR/vector.toml or set VECTOR_TOML_SRC to a valid path."
fi
sudo install -m 0644 "$VECTOR_TOML_SRC" /etc/t12/vector.toml

# Normalize env var names expected by /etc/t12/vector.toml
set -a
source "$ENV_SRC"
set +a
CLICKHOUSE_ENDPOINT="${CLICKHOUSE_ENDPOINT:-${CLICKHOUSE_HTTPS_ENDPOINT:-}}"
CLICKHOUSE_CA_CERT="${CLICKHOUSE_CA_CERT:-}"
if [[ -z "${CLICKHOUSE_CA_CERT:-}" || ! -f "$CLICKHOUSE_CA_CERT" ]]; then
  CLICKHOUSE_CA_CERT="/etc/t12/ca.crt"
fi
if [[ -z "${CLICKHOUSE_ENDPOINT:-}" ]]; then
  die "Vector env is missing CLICKHOUSE_ENDPOINT (or CLICKHOUSE_HTTPS_ENDPOINT) in: $ENV_SRC"
fi
if [[ -z "${CLICKHOUSE_USER:-}" || -z "${CLICKHOUSE_PASSWORD:-}" ]]; then
  die "Vector env is missing CLICKHOUSE_USER/CLICKHOUSE_PASSWORD in: $ENV_SRC"
fi
if [[ ! -f "$CLICKHOUSE_CA_CERT" ]]; then
  die "Vector env CA cert not found: $CLICKHOUSE_CA_CERT"
fi

sudo tee /etc/t12/vector.env >/dev/null <<EOF
CLICKHOUSE_ENDPOINT=${CLICKHOUSE_ENDPOINT}
CLICKHOUSE_USER=${CLICKHOUSE_USER}
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
CLICKHOUSE_CA_CERT=${CLICKHOUSE_CA_CERT}
EOF
sudo chmod 600 /etc/t12/vector.env

# Copy to the invoking user's home with proper ownership and perms
sudo cp /etc/t12/vector.env "$HOME/.t12.env"
sudo chown "$(id -u)":"$(id -gn)" "$HOME/.t12.env"
sudo chmod 600 "$HOME/.t12.env"

# Verify copies
check_file /etc/t12/ca.crt
check_file /etc/t12/vector.toml
check_file /etc/t12/vector.env
check_file "$HOME/.t12.env"

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

# Probe Vector health (best-effort)
sleep 2
HC=$(http_code http://127.0.0.1:8686/health || true)
echo "[INFO] Vector health HTTP code: ${HC:-none}"

# Export runtime env for training
set -a; [ -f "$HOME/.t12.env" ] && source "$HOME/.t12.env"; set +a
export VECTOR_SERVICE_NAME="t12-vector.service"
export SKIP_VECTOR_CHECK="${SKIP_VECTOR_CHECK:-0}"

# Final preflight before training
check_file /etc/t12/vector.env
check_file "$HOME/.t12.env"

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
  "${DEEPSPEED_BIN[@]:-$DEEPSPEED_BIN}" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
) 2>&1 | tee "$RESULTS_DIR/run/train.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Test 14 completed"
echo "  Init model: $INIT_CKPT"
echo "  Train log:  $RESULTS_DIR/run/train.log"
echo "  Metrics:    $RESULTS_DIR/run/metrics.jsonl"
