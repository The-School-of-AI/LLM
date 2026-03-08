#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CFG:-$TEST_ROOT/configs/config.yaml}"
RESULTS_DIR="$TEST_ROOT/_data/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"
TOKENIZER_DIR="${TOKENIZER_DIR:-$TEST_ROOT/_data/tokenizer}"

NUM_GPUS="${NUM_GPUS:-8}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-1}"
RANK_ISSUE_REGEX="${RANK_ISSUE_REGEX:-NCCL|ProcessGroupNCCL|RuntimeError|Traceback|CUDA error|ERROR|WARN}"
NCCL_PER_PROCESS_LOGS="${NCCL_PER_PROCESS_LOGS:-0}"
LOGROTATE_TEMPLATE="${LOGROTATE_TEMPLATE:-$TEST_ROOT/../infra/logging/logrotate.training.conf}"
AUTO_INSTALL_LOGROTATE="${AUTO_INSTALL_LOGROTATE:-0}"
LOGROTATE_TARGET="${LOGROTATE_TARGET:-/etc/logrotate.d/llm-training}"
VERIFY_LOGROTATE="${VERIFY_LOGROTATE:-0}"
WATCHDOG_ENABLED="${WATCHDOG_ENABLED:-1}"

if [[ ! -f "$CFG" ]]; then
  if [[ -f "$TEST_ROOT/configs/config-mini.yaml" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config not found at $CFG; falling back to config-mini.yaml"
    CFG="$TEST_ROOT/configs/config-mini.yaml"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Config file not found: $CFG"
    exit 1
  fi
fi

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

# ---------------------------------------------------------------------------
# Parse loader_type, shard_dir, eval_shard_dir from YAML config
# ---------------------------------------------------------------------------
_yaml_val() {
  uv run python -c "
import sys, yaml
with open('$CFG') as f:
    cfg = yaml.safe_load(f)
key = '$1'
val = cfg.get('data', {}).get(key)
print('' if val is None else str(val))
"
}

_yaml_top_val() {
  uv run python -c "
import sys, yaml
with open('$CFG') as f:
    cfg = yaml.safe_load(f) or {}
key = '$1'
val = cfg.get(key)
print('' if val is None else str(val))
"
}

_yaml_observability_backend() {
  uv run python -c "
import yaml
with open('$CFG') as f:
    cfg = yaml.safe_load(f) or {}
obs = cfg.get('observability') or {}
print(str(obs.get('backend') or ''))
"
}

LOADER_TYPE="$(_yaml_val loader_type)"
_raw_shard_dir="$(_yaml_val shard_dir)"
_raw_eval_shard_dir="$(_yaml_val eval_shard_dir)"
_dataset_name="$(_yaml_val dataset_name)"
_dataset_config="$(_yaml_val dataset_config)"
RUN_ID_CFG="$(_yaml_top_val run_id)"
OBS_BACKEND="$(_yaml_observability_backend)"

# Resolve relative paths against config file directory (matches main.py _resolve_path)
CFG_DIR="$(cd "$(dirname "$CFG")" && pwd)"
_abs_path() {
  local p="$1"
  [[ -z "$p" ]] && echo "" && return
  [[ "$p" = /* ]] && echo "$p" || echo "$CFG_DIR/$p"
}
SHARD_DIR="$(_abs_path "$_raw_shard_dir")"
EVAL_SHARD_DIR="$(_abs_path "$_raw_eval_shard_dir")"

# ---------------------------------------------------------------------------
# Auto-create shards if loader_type=bin_idx and shard dirs are missing/empty
# ---------------------------------------------------------------------------
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  if [[ -n "$SHARD_DIR" && ( ! -d "$SHARD_DIR" || -z "$(ls -A "$SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shard dir missing/empty: $SHARD_DIR — creating train shards..."
    mkdir -p "$SHARD_DIR"
    (
      uv run "$TEST_ROOT/scripts/create_shards.py" \
        --dataset "$_dataset_name" \
        --dataset-config "$_dataset_config" \
        --split train \
        --output-dir "$SHARD_DIR" \
        --tokenizer "$TOKENIZER_DIR" \
        --tokens-per-shard 4096000 \
        --band B1 \
        --domain general \
        --stage 1
    )
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Train shards written to: $SHARD_DIR"
  fi

  if [[ -n "$EVAL_SHARD_DIR" && ( ! -d "$EVAL_SHARD_DIR" || -z "$(ls -A "$EVAL_SHARD_DIR" 2>/dev/null)" ) ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval shard dir missing/empty: $EVAL_SHARD_DIR — creating eval shards..."
    mkdir -p "$EVAL_SHARD_DIR"
    (
      uv run "$TEST_ROOT/scripts/create_shards.py" \
        --dataset "$_dataset_name" \
        --dataset-config "$_dataset_config" \
        --split validation \
        --output-dir "$EVAL_SHARD_DIR" \
        --tokenizer "$TOKENIZER_DIR" \
        --tokens-per-shard 4096000 \
        --band B1 \
        --domain general \
        --stage 1
    )
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Eval shards written to: $EVAL_SHARD_DIR"
  fi
fi

if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Saving deterministic init model..."
  uv run "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$CFG" \
    --output "$INIT_CKPT" \
    --meta "$INIT_META"
else
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Reusing existing init model: $INIT_CKPT"
fi

RUN_LOG_DIR="$RESULTS_DIR/run"
COMBINED_LOG="$RUN_LOG_DIR/training_combined.log"
ISSUES_LOG="$RUN_LOG_DIR/rank_issues.log"
RANK_TMP_DIR="$RUN_LOG_DIR/.rank_tmp"
RENDERED_LOGROTATE_PATH="$RUN_LOG_DIR/logrotate.training.rendered.conf"
WATCHDOG_STDOUT_LOG="$RUN_LOG_DIR/watchdog.log"
WATCHDOG_TRAINING_PID_FILE="$RUN_LOG_DIR/training_launcher.pid"
WATCHDOG_PID=""

timestamp_stream() {
  if command -v ts >/dev/null 2>&1; then
    ts '%Y-%m-%dT%H:%M:%SZ'
  else
    python3 -u -c '
import datetime
import sys

for line in sys.stdin:
    now = datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sys.stdout.write(f"{now} {line}")
    sys.stdout.flush()
'
  fi
}

setup_logrotate_config() {
  local escaped_run_log_dir
  local config_to_check

  if [[ ! -f "$LOGROTATE_TEMPLATE" ]]; then
    echo "  Logrotate template missing: $LOGROTATE_TEMPLATE"
    return 0
  fi

  escaped_run_log_dir="$(printf '%s' "$RUN_LOG_DIR" | sed 's/[\\/&]/\\&/g')"
  if ! sed "s/__RUN_LOG_DIR__/$escaped_run_log_dir/g" "$LOGROTATE_TEMPLATE" > "$RENDERED_LOGROTATE_PATH"; then
    echo "  Failed to render logrotate config from template: $LOGROTATE_TEMPLATE"
    return 0
  fi
  echo "  Rendered logrotate config: $RENDERED_LOGROTATE_PATH"

  if [[ "$AUTO_INSTALL_LOGROTATE" == "1" ]]; then
    if [[ $EUID -eq 0 ]]; then
      install -m 0644 "$RENDERED_LOGROTATE_PATH" "$LOGROTATE_TARGET" || {
        echo "  Failed to install logrotate config to: $LOGROTATE_TARGET"
        return 0
      }
      echo "  Installed logrotate config: $LOGROTATE_TARGET"
    elif command -v sudo >/dev/null 2>&1; then
      sudo install -m 0644 "$RENDERED_LOGROTATE_PATH" "$LOGROTATE_TARGET" || {
        echo "  Failed to install logrotate config with sudo: $LOGROTATE_TARGET"
        return 0
      }
      echo "  Installed logrotate config: $LOGROTATE_TARGET"
    else
      echo "  Cannot auto-install logrotate config (no root/sudo)."
      echo "  Install manually: install -m 0644 \"$RENDERED_LOGROTATE_PATH\" \"$LOGROTATE_TARGET\""
    fi
  fi

  if [[ "$VERIFY_LOGROTATE" == "1" ]]; then
    if command -v logrotate >/dev/null 2>&1; then
      config_to_check="$RENDERED_LOGROTATE_PATH"
      if [[ "$AUTO_INSTALL_LOGROTATE" == "1" ]]; then
        config_to_check="$LOGROTATE_TARGET"
      fi
      if logrotate -d "$config_to_check" >/dev/null 2>&1; then
        echo "  Logrotate config validated: $config_to_check"
      else
        echo "  Logrotate validation failed: $config_to_check"
      fi
    else
      echo "  logrotate not found; skipped validation."
    fi
  fi
}

start_watchdog() {
  if [[ "$WATCHDOG_ENABLED" != "1" ]]; then
    return 0
  fi

  export WATCHDOG_TRAINING_PID_FILE="$WATCHDOG_TRAINING_PID_FILE"
  export WATCHDOG_TRACKING_LOG_PATH="${WATCHDOG_TRACKING_LOG_PATH:-/tmp/watchdog_metrics.jsonl}"
  export WATCHDOG_ALERT_LOG_PATH="${WATCHDOG_ALERT_LOG_PATH:-/tmp/watchdog_alerts.jsonl}"
  export WATCHDOG_CONTROL_FILE_PATH="${WATCHDOG_CONTROL_FILE_PATH:-/tmp/training_control.flag}"
  export WATCHDOG_METRICS_URL="${WATCHDOG_METRICS_URL:-http://localhost:8000}"
  export WATCHDOG_POLL_INTERVAL_S="${WATCHDOG_POLL_INTERVAL_S:-5}"
  export WATCHDOG_ENABLE_ALERTER="${WATCHDOG_ENABLE_ALERTER:-1}"

  if [[ -n "$RUN_ID_CFG" ]]; then
    export WATCHDOG_RUN_ID="$RUN_ID_CFG"
  fi

  uv run python -u -m llm.logger.p12.watchdog.watchdog >> "$WATCHDOG_STDOUT_LOG" 2>&1 &
  WATCHDOG_PID=$!
}

stop_watchdog() {
  if [[ -z "${WATCHDOG_PID:-}" ]]; then
    return 0
  fi
  if kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
  fi
}

mkdir -p "$RANK_TMP_DIR"
find "$RANK_TMP_DIR" -type f -delete
: > "$COMBINED_LOG"
: > "$ISSUES_LOG"
rm -f "$WATCHDOG_TRAINING_PID_FILE"
trap stop_watchdog EXIT

if [[ "$NCCL_PER_PROCESS_LOGS" == "1" ]]; then
  export NCCL_DEBUG_FILE="$RUN_LOG_DIR/nccl.%h.%p.log"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..." | tee -a "$COMBINED_LOG"
echo "  Combined log:  $COMBINED_LOG" | tee -a "$COMBINED_LOG"
echo "  Rank issues:   $ISSUES_LOG" | tee -a "$COMBINED_LOG"
echo "  NCCL_DEBUG:    $NCCL_DEBUG" | tee -a "$COMBINED_LOG"
if [[ "${NCCL_DEBUG_FILE:-}" != "" ]]; then
  echo "  NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE" | tee -a "$COMBINED_LOG"
fi
echo "  NUM_GPUS:      $NUM_GPUS" | tee -a "$COMBINED_LOG"
setup_logrotate_config | tee -a "$COMBINED_LOG"
if [[ "$WATCHDOG_ENABLED" == "1" ]]; then
  if [[ "$OBS_BACKEND" != "p12" ]]; then
    echo "  ⚠ observability.backend='$OBS_BACKEND' (expected 'p12' for full watchdog metrics)" | tee -a "$COMBINED_LOG"
  fi
  start_watchdog
  sleep 1
  if ! kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    echo "  ❌ Watchdog failed to start. Check $WATCHDOG_STDOUT_LOG" | tee -a "$COMBINED_LOG"
    exit 1
  fi
  echo "  Watchdog PID:  $WATCHDOG_PID" | tee -a "$COMBINED_LOG"
  echo "  Watchdog log:  $WATCHDOG_STDOUT_LOG" | tee -a "$COMBINED_LOG"
fi

set +e
(
  echo "$BASHPID" > "$WATCHDOG_TRAINING_PID_FILE"
  exec uv run "$DEEPSPEED_BIN" \
    --num_gpus="$NUM_GPUS" \
    --enable_each_rank_log "$RANK_TMP_DIR/rank" \
    main.py \
    --config "$CFG"
) 2>&1 | timestamp_stream | tee -a "$COMBINED_LOG"
TRAIN_EXIT="${PIPESTATUS[0]}"
set -e
stop_watchdog

_extract_rank_from_path() {
  local path="$1"
  local base rank
  base="$(basename "$path")"

  rank="$(echo "$base" | sed -nE 's/.*([Rr]ank|local_rank)[^0-9]*([0-9]+).*/\2/p')"
  if [[ -z "$rank" ]]; then
    rank="$(echo "$base" | sed -nE 's/.*[^0-9]([0-9]+)(\.[^.]+)?$/\1/p')"
  fi
  [[ -n "$rank" ]] && echo "$rank" || echo "unknown"
}

_append_issues_from_file() {
  local file_path="$1"
  local rank="$2"
  grep -Ei "$RANK_ISSUE_REGEX" "$file_path" | sed "s/^/[rank=$rank] /" >> "$ISSUES_LOG" || true
}

rank_log_count=0
while IFS= read -r rank_log; do
  rank_log_count=$((rank_log_count + 1))
  rank_value="$(_extract_rank_from_path "$rank_log")"
  _append_issues_from_file "$rank_log" "$rank_value"
done < <(find "$RANK_TMP_DIR" -type f | sort)

if [[ "$rank_log_count" -eq 0 ]]; then
  grep -Ei "$RANK_ISSUE_REGEX" "$COMBINED_LOG" > "$ISSUES_LOG" || true
fi

find "$RANK_TMP_DIR" -mindepth 1 -delete
rmdir "$RANK_TMP_DIR" 2>/dev/null || true

if [[ "$TRAIN_EXIT" -ne 0 ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] training failed with exit code $TRAIN_EXIT" | tee -a "$COMBINED_LOG"
  echo "  Combined log:  $COMBINED_LOG"
  echo "  Rank issues:   $ISSUES_LOG"
  exit "$TRAIN_EXIT"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] completed"
echo "  Init model:      $INIT_CKPT"
echo "  Combined log:    $COMBINED_LOG"
echo "  Rank issues:     $ISSUES_LOG"
echo "  Metrics:         $RESULTS_DIR/run/metrics.jsonl"
echo "  Profile report:  $RESULTS_DIR/run/profile_report.txt  (if profile_steps were set)"
echo "  Profile JSONL:   $RESULTS_DIR/run/profile.jsonl       (if profile_steps were set)"
if [[ "$LOADER_TYPE" == "bin_idx" ]]; then
  echo "  Train shards:    $SHARD_DIR"
  [[ -n "$EVAL_SHARD_DIR" ]] && echo "  Eval shards:     $EVAL_SHARD_DIR"
fi
