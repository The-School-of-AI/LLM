#!/usr/bin/env bash
# Stable single-node setup for LightningLM training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LIGHTNINGLM_ROOT="${LIGHTNINGLM_ROOT:-$ROOT_DIR}"
LIGHTNINGLM_VENV="${LIGHTNINGLM_VENV:-$LIGHTNINGLM_ROOT/.venv}"
LIGHTNINGLM_DATA_DIR="${LIGHTNINGLM_DATA_DIR:-$LIGHTNINGLM_ROOT/data/curriculum_test_shards}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================="
echo "LightningLM stable setup — $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================="

if [[ ! -x "$LIGHTNINGLM_VENV/bin/python3" ]]; then
  echo "Creating venv: $LIGHTNINGLM_VENV"
  "$PYTHON_BIN" -m venv "$LIGHTNINGLM_VENV"
fi

source "$LIGHTNINGLM_VENV/bin/activate"
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "$LIGHTNINGLM_ROOT/requirements/runtime.txt"

export PYTHONPATH="$LIGHTNINGLM_ROOT:${PYTHONPATH:-}"
python3 "$LIGHTNINGLM_ROOT/scripts/doctor.py"

SHARD_COUNT=$(find "$LIGHTNINGLM_DATA_DIR" -name "tokens.bin" -type f 2>/dev/null | wc -l | tr -d ' ')
if [[ "$SHARD_COUNT" == "0" ]]; then
  echo "No shards found at $LIGHTNINGLM_DATA_DIR"
  echo "Create smoke shards with:"
  echo "  python3 scripts/create_curriculum_test_shards.py --output-dir $LIGHTNINGLM_DATA_DIR --manifest-dir manifests"
else
  echo "Data present: $LIGHTNINGLM_DATA_DIR ($SHARD_COUNT shards)"
fi

mkdir -p "$LIGHTNINGLM_ROOT/results/init" "$LIGHTNINGLM_ROOT/results/run"

echo ""
echo "READY"
echo "  Root: $LIGHTNINGLM_ROOT"
echo "  Venv: $LIGHTNINGLM_VENV"
echo "  Data: $LIGHTNINGLM_DATA_DIR"
echo ""
echo "Run a stage with:"
echo "  source $LIGHTNINGLM_VENV/bin/activate"
echo "  NUM_GPUS=8 bash scripts/run_2b_stage.sh"
