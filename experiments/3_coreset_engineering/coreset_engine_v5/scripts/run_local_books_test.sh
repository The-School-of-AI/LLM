#!/usr/bin/env bash
# =============================================================================
# Run the full coreset pipeline locally on the books source.
# By default downloads real T2 books from S3 and runs selection on it.
# Fallback: synthetic sample (generate_books_sample.py) if download fails or
# USE_T2_BOOKS=0.
#
# T2 books source (downloaded to data/local_test/books/t2_books/):
#   s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/source=books/
#
# Prerequisites:
#   - Python env: cd <repo_root> && uv sync --all-packages
#   - For T2 download: AWS CLI configured (aws s3 sync)
#   - Run from repo root:  bash experiments/3_coreset_engineering/coreset_engine_v5/scripts/run_local_books_test.sh
#   - Or from engine dir:  bash scripts/run_local_books_test.sh
#
# Optional env:
#   USE_T2_BOOKS=0     Use synthetic sample only (no S3 download).
#   SKIP_CLEAN=1       Do not remove previous output/checkpoints.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_DIR="$(cd "$ENGINE_DIR/.." && pwd)"   # experiments/3_coreset_engineering (has pyproject.toml)
# Repo root (LLM): where uv.lock and workspace pyproject.toml live
REPO_ROOT="$(cd "$ENGINE_DIR/../../.." && pwd 2>/dev/null)" || REPO_ROOT="$EXPERIMENT_DIR"

# Input/output under engine dir for a self-contained test
DATA_DIR="${ENGINE_DIR}/data/local_test/books"
# Actual T2 books source (used when USE_T2_BOOKS=1)
BOOKS_T2_S3="s3://t2-datacurriculum-353/processed_dataset/curriculum_pyspark_output/source=books"
T2_BOOKS_DIR="${DATA_DIR}/t2_books"
# Synthetic fallback
SAMPLE_JSONL="${DATA_DIR}/sample.jsonl"
CONFIG="${ENGINE_DIR}/config/pipeline.yaml"
CURRICULUM="${ENGINE_DIR}/config/curriculum_t3_aligned.yaml"
OUTPUT_CORESETS="${ENGINE_DIR}/output/coresets"
OUTPUT_CHECKPOINTS="${ENGINE_DIR}/output/checkpoints"
NUM_CHUNKS="${NUM_CHUNKS:-800}"
# Use real T2 books from S3 (download and use). Set to 0 to use synthetic sample only.
USE_T2_BOOKS="${USE_T2_BOOKS:-1}"

echo "=== Local books pipeline test ==="
echo "  Engine dir:  $ENGINE_DIR"
echo "  Data dir:   $DATA_DIR"
echo "  Config:     $CONFIG"
echo "  Curriculum: $CURRICULUM"
echo ""

# Prerequisite: dependencies (scipy, pyarrow, etc.)
# This repo is a uv workspace: use "uv sync --all-packages" at REPO_ROOT so member deps (scipy, pyyaml, etc.) are installed
VENV_DIR=""
for d in "$REPO_ROOT" "$EXPERIMENT_DIR"; do
  if [[ -x "${d}/.venv/bin/python3" ]]; then
    VENV_DIR="${d}/.venv"
    break
  fi
done
if [[ -z "$VENV_DIR" ]]; then
  echo "Creating venv and installing dependencies (uv sync --all-packages from repo root)..."
  (cd "$REPO_ROOT" && uv sync --all-packages) || true
  for d in "$REPO_ROOT" "$EXPERIMENT_DIR"; do
    if [[ -x "${d}/.venv/bin/python3" ]]; then
      VENV_DIR="${d}/.venv"
      break
    fi
  done
fi
if [[ -n "$VENV_DIR" ]]; then
  echo "  Using venv: $VENV_DIR"
  export PATH="${VENV_DIR}/bin:${PATH}"
  export PYTHON_BIN="${VENV_DIR}/bin/python3"
  PYTHON="${VENV_DIR}/bin/python3"
else
  PYTHON=python3
fi
# Coreset pipeline needs scipy (selection) and yaml (config/curriculum)
if ! "$PYTHON" -c "import scipy" 2>/dev/null; then
  echo "ERROR: scipy not found. Install workspace deps from repo root:"
  echo "  cd $REPO_ROOT && uv sync --all-packages"
  echo "Then run this script again."
  exit 1
fi
if ! "$PYTHON" -c "import yaml" 2>/dev/null; then
  echo "ERROR: PyYAML (yaml) not found in $VENV_DIR. Install workspace deps:"
  echo "  cd $REPO_ROOT && uv sync --all-packages"
  echo "Then run this script again."
  exit 1
fi

# 1. Prepare input: prefer real T2 books from S3 (download if needed), else synthetic sample
INPUT_PATH=""
INPUT_FORMAT=""
TOTAL_TOKENS=""

if [[ "${USE_T2_BOOKS}" == "1" ]]; then
  echo "[1/4] Using T2 books source: $BOOKS_T2_S3"
  mkdir -p "$T2_BOOKS_DIR"
  # Download if we don't have any parquet yet
  PARQUET_COUNT=$(find "$T2_BOOKS_DIR" -maxdepth 3 -name "*.parquet" 2>/dev/null | wc -l | tr -d ' ')
  if [[ -z "$PARQUET_COUNT" || "$PARQUET_COUNT" -eq 0 ]]; then
    echo "  Downloading from S3 (aws s3 sync)..."
    if aws s3 sync "${BOOKS_T2_S3}/" "${T2_BOOKS_DIR}/" 2>/dev/null; then
      echo "  Download complete."
    else
      echo "  WARNING: aws s3 sync failed (check AWS credentials). Falling back to synthetic sample."
      USE_T2_BOOKS=0
    fi
  else
    echo "  Using existing T2 books under: $T2_BOOKS_DIR"
  fi

  if [[ "${USE_T2_BOOKS}" == "1" ]]; then
    # Sum token_count or token_count_estimate from all parquet under T2_BOOKS_DIR
    TOTAL_TOKENS=$("$PYTHON" -c "
import pyarrow.parquet as pq
from pathlib import Path
total = 0
for p in Path('$T2_BOOKS_DIR').rglob('*.parquet'):
    try:
        t = pq.read_table(p)
        for col in ('token_count', 'token_count_estimate'):
            if col in t.column_names:
                total += t.column(col).sum().as_py()
                break
    except Exception:
        pass
print(total)
" 2>/dev/null) || TOTAL_TOKENS=""
    if [[ -z "$TOTAL_TOKENS" || "$TOTAL_TOKENS" -le 0 ]]; then
      echo "  WARNING: Could not compute total tokens from parquet. Using 50000000."
      TOTAL_TOKENS=50000000
    fi
    INPUT_PATH="$T2_BOOKS_DIR"
    INPUT_FORMAT="parquet"
  fi
fi

if [[ -z "$INPUT_PATH" ]]; then
  # Synthetic sample fallback
  if [[ ! -f "$SAMPLE_JSONL" ]]; then
    echo "[1/4] Generating synthetic books sample ($NUM_CHUNKS chunks)..."
    mkdir -p "$DATA_DIR"
    "$PYTHON" "${ENGINE_DIR}/tools/generate_books_sample.py" \
      --out "$SAMPLE_JSONL" \
      --num-chunks "$NUM_CHUNKS"
  else
    echo "[1/4] Using existing synthetic sample: $SAMPLE_JSONL"
  fi
  INPUT_PATH="$SAMPLE_JSONL"
  INPUT_FORMAT="jsonl"
  TOTAL_TOKENS="$(( NUM_CHUNKS * 512 ))"
fi

echo "  Input:       $INPUT_PATH"
echo "  Format:      $INPUT_FORMAT"
echo "  Total tokens (estimate): $TOTAL_TOKENS"
echo ""

# 2. Clean previous test outputs (optional: set SKIP_CLEAN=1 to keep)
if [[ "${SKIP_CLEAN:-0}" != "1" ]]; then
  echo "[2/4] Cleaning previous test outputs..."
  rm -rf "$OUTPUT_CORESETS" "$OUTPUT_CHECKPOINTS" 2>/dev/null || true
fi
mkdir -p "$OUTPUT_CORESETS" "$OUTPUT_CHECKPOINTS"
echo ""

# 3. Run pipeline: 1 shard, 1B stage only, small batch size
echo "[3/4] Running coreset pipeline (1 shard, 1B stage)..."
cd "$ENGINE_DIR"

bash "${ENGINE_DIR}/shard.sh" \
  --input-path "$INPUT_PATH" \
  --input-format "$INPUT_FORMAT" \
  --num-shards 1 \
  --stages "1B" \
  --config "$CONFIG" \
  --curriculum "$CURRICULUM" \
  --checkpoint-base "$OUTPUT_CHECKPOINTS" \
  --total-tokens "$TOTAL_TOKENS" \
  --batch-size 200 \
  --checkpoint-every-n-batches 2

echo ""

# 4. Validate outputs
echo "[4/4] Validating outputs..."
if [[ -f "${ENGINE_DIR}/tools/validate_coreset_outputs.py" ]]; then
  "$PYTHON" "${ENGINE_DIR}/tools/validate_coreset_outputs.py" \
    --curriculum "$CURRICULUM" \
    --output-dir "$OUTPUT_CORESETS" \
    --stages "1B" \
    --format checklist 2>/dev/null || true
else
  echo "  (validate_coreset_outputs.py not run)"
fi

echo ""
echo "=== Done ==="
echo "  Coreset output:  $OUTPUT_CORESETS/1B/"
echo "  Checkpoints:     $OUTPUT_CHECKPOINTS/"
echo "  Inspect indices: head -5 \"$OUTPUT_CORESETS/1B/\"selected_indices*.parquet 2>/dev/null || cat .../selected_indices_part_*.jsonl | head -3"
echo "  Merge parts (copy-paste; works from any dir):"
echo "    $PYTHON ${ENGINE_DIR}/tools/merge_selected_indices.py --coreset-root ${OUTPUT_CORESETS} --stage 1B"
