#!/usr/bin/env bash
set -euo pipefail

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../" && pwd)"
CODE_DIR="$TEST_ROOT/code"
RESULTS_DIR="$TEST_ROOT/results"
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
INIT_META="$RESULTS_DIR/init/model_init_meta.json"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

export PYTHONPATH="${TEST_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHDYNAMO_DISABLE=1
export T19_STEP_CUDA_SYNC=1
export T19_STEP_GC_COLLECT=1
export T19_STEP_EMPTY_CACHE=1
export T19_STEP_IPC_COLLECT=0
export T19_ZERO3_RELEASE_EVERY=1
export T19_ZERO3_FORCE_CLEAR_CONTAINERS=0
export T19_CLEAR_ROUTER_CACHE_EVERY=1
export T19_TRACK_CUDA_MEMORY=1
export T19_REV_CKPT_USE_REENTRANT=0

echo "============================================================"
echo "8B MoE Batch Size Sweep (BS8 -> BS16 -> BS32), SL4096, 10 steps each"
echo "============================================================"

# Save init model with 8B config (use BS8 config, just for init)
INIT_CFG="$TEST_ROOT/configs/test_8b_moe_4096_bs8_10steps.yaml"
echo "[13:10:32] Saving 8B init model..."
mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"
"$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py"     --config "$INIT_CFG"     --output "$INIT_CKPT"     --meta "$INIT_META"
echo "[13:10:32] 8B init model saved."

for BS in 8 16 32; do
    CFG="$TEST_ROOT/configs/test_8b_moe_4096_bs${BS}_10steps.yaml"
    echo ""
    echo "============================================================"
    echo "[13:10:32] Starting 8B MoE BS${BS} (10 steps)..."
    echo "============================================================"

    cd "$CODE_DIR"
    if "$DEEPSPEED_BIN" --num_gpus=8 main.py --config "$CFG" 2>&1; then
        echo "[13:10:32] BS${BS} completed successfully."
    else
        RC=$?
        echo "[13:10:32] BS${BS} FAILED (exit code $RC) — likely OOM. Stopping sweep."
        exit $RC
    fi
done

echo ""
echo "============================================================"
echo "[13:10:32] Sweep complete!"
echo "============================================================"
