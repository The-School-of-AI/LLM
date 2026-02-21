#!/usr/bin/env bash
# ============================================================
# Team 16 Early Warning — Full Checkpoint Evaluation Pipeline
# ============================================================
# Usage:
#   ./scripts/run_checkpoint_eval.sh <checkpoint_path> <checkpoint_name> [quant_mode] [backend]
#
# Examples:
#   ./scripts/run_checkpoint_eval.sh ./models/step_500.gguf step_500 int4 llama_cpp
#   ./scripts/run_checkpoint_eval.sh meta-llama/Llama-2-7b step_0_baseline int8 bitsandbytes
#
# After evaluation, optionally submit to central collector:
#   COLLECTOR_URL=http://192.168.1.100:5001 ./scripts/run_checkpoint_eval.sh ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${1:-}"
CHECKPOINT_NAME="${2:-}"
QUANT="${3:-int4}"
BACKEND="${4:-auto}"
COLLECTOR_URL="${COLLECTOR_URL:-}"

if [[ -z "$CHECKPOINT" || -z "$CHECKPOINT_NAME" ]]; then
    echo "Usage: $0 <checkpoint_path> <checkpoint_name> [quant=int4] [backend=auto]"
    exit 1
fi

cd "$ROOT"

echo "========================================================"
echo "  Team 16 Early Warning — Checkpoint Evaluation"
echo "  Checkpoint : $CHECKPOINT_NAME"
echo "  Quant      : $QUANT"
echo "  Backend    : $BACKEND"
echo "========================================================"

# Step 1: Run quantized evaluation
echo ""
echo "[1/4] Running eval suite ..."
python evals/quantized/run_eval.py \
    --checkpoint "$CHECKPOINT" \
    --checkpoint-name "$CHECKPOINT_NAME" \
    --quant "$QUANT" \
    --backend "$BACKEND" \
    --verbose

# Step 2: Aggregate results
echo ""
echo "[2/4] Aggregating results ..."
python collector/collect_results.py \
    --results-dir results/raw \
    --out results/aggregated_results.json

# Step 3: Track trends
echo ""
echo "[3/4] Tracking trends and detecting anomalies ..."
python scripts/track_trends.py --plot-format both

# Step 4: Generate early warning report
echo ""
echo "[4/4] Generating early warning report ..."
python scripts/generate_report.py

# Optional: submit to central collector
if [[ -n "$COLLECTOR_URL" ]]; then
    echo ""
    echo "[+] Submitting results to central collector: $COLLECTOR_URL"
    # Find the most recent result file
    LATEST_RESULT=$(ls -t results/raw/*.json 2>/dev/null | head -1)
    if [[ -n "$LATEST_RESULT" ]]; then
        python scripts/submit_result.py --file "$LATEST_RESULT" --server "$COLLECTOR_URL"
    fi
fi

echo ""
echo "========================================================"
echo "  DONE"
echo "  Results  : results/raw/"
echo "  Report   : results/reports/latest_early_warning.md"
echo "  Dashboard: results/plots/trend_dashboard.html"
echo "========================================================"
