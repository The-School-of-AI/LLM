#!/usr/bin/env bash
# Run the SFT data pipeline on the sample input for observation-mode validation.
# Team 18 — No weight updates; validation only.
# Run from: sft_data/observation_mode (or from repo root with adjusted paths)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${SCRIPT_DIR}/../scripts"
cd "$SCRIPT_DIR"

echo "=== Pipeline validation (observation mode) ==="
echo "Working dir: $SCRIPT_DIR"
echo ""

# 1. Standardize
echo "[1/4] Standardize format (alpaca -> conversations)..."
python3 "${SCRIPTS_DIR}/standardize_conversation_format.py" \
  "${SCRIPT_DIR}/sample_input_alpaca.jsonl" \
  "${SCRIPT_DIR}/standardized.jsonl" \
  --format alpaca
echo "  -> standardized.jsonl created"
echo ""

# 2. Apply chat template
echo "[2/4] Apply chat template (chatml)..."
python3 "${SCRIPTS_DIR}/apply_chat_template.py" \
  "${SCRIPT_DIR}/standardized.jsonl" \
  "${SCRIPT_DIR}/templated.jsonl" \
  --template chatml
echo "  -> templated.jsonl created"
echo ""

# 3. Train/val split (use standardized as input for split)
echo "[3/4] Train/val split..."
python3 "${SCRIPTS_DIR}/train_val_split.py" \
  "${SCRIPT_DIR}/standardized.jsonl" \
  --train-out "${SCRIPT_DIR}/train.jsonl" \
  --val-out "${SCRIPT_DIR}/val.jsonl" \
  --val-ratio 0.2 \
  --seed 42
echo "  -> train.jsonl, val.jsonl created"
echo ""

# 4. Verify loss masking (no tokenizer required for basic run)
echo "[4/4] Verify loss masking..."
python3 "${SCRIPTS_DIR}/verify_loss_masking.py" \
  "${SCRIPT_DIR}/train.jsonl" \
  --sample 5
echo ""

echo "=== Pipeline validation complete ==="
echo "Optional: re-run verify_loss_masking with --tokenizer path/to/model for full label check."
echo "Record results in NEXT_STEPS_COMPLETION_REPORT.md §4."
