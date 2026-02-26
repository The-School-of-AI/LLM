#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# run_diagnostic.sh — Isolation matrix for cudaErrorMisalignedAddress
#
# Usage:
#   bash run_diagnostic.sh no_compile        # Test 1: compile OFF, 1 GPU
#   bash run_diagnostic.sh compile_on        # Test 2: compile ON,  1 GPU
#   bash run_diagnostic.sh compile_on_8gpu   # Test 3: compile ON,  8 GPU
#
# All tests run with CUDA_LAUNCH_BLOCKING=1 so errors are reported synchronously
# at the ACTUAL failing kernel, not at some later unrelated API call.
#
# Decision tree:
#   Test 1 FAILS → problem is FLA or fused CE kernel (not compile)
#   Test 1 PASS, Test 2 FAILS → compile + single GPU problem
#   Test 1 PASS, Test 2 PASS, Test 3 FAILS → compile + multi-GPU/DeepSpeed problem
#   All PASS → original error was non-deterministic or fixed by earlier patches
###############################################################################

TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$TEST_ROOT/code"
RESULTS_DIR="$TEST_ROOT/results"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

# ── Debug environment: makes CUDA errors synchronous ──
export CUDA_LAUNCH_BLOCKING=1
export TORCH_SHOW_CPP_STACKTRACES=1
export NCCL_ASYNC_ERROR_HANDLING=1

MODE="${1:-no_compile}"

mkdir -p "$RESULTS_DIR/init" "$RESULTS_DIR/run"

# ── Generate init model if needed ──
INIT_CKPT="$RESULTS_DIR/init/model_init.pt"
FORCE_REWRITE_INIT="${FORCE_REWRITE_INIT:-1}"

# Use the no_compile config for init (doesn't matter which, just need model structure)
INIT_CFG="$TEST_ROOT/configs/diag_no_compile.yaml"
if [[ ! -f "$INIT_CKPT" || "$FORCE_REWRITE_INIT" == "1" ]]; then
  echo "================================================================"
  echo "[$(date '+%H:%M:%S')] Saving deterministic init model..."
  echo "================================================================"
  "$PYTHON_BIN" "$TEST_ROOT/scripts/save_init_model.py" \
    --config "$INIT_CFG" \
    --output "$INIT_CKPT" \
    --meta "$RESULTS_DIR/init/model_init_meta.json"
fi

case "$MODE" in
  # ──────────────────────────────────────────────────────────────────
  # TEST 1: No compile, single GPU
  # Isolates: is the error from FLA / fused CE Triton kernels?
  # ──────────────────────────────────────────────────────────────────
  no_compile)
    CFG="$TEST_ROOT/configs/diag_no_compile.yaml"
    LOG="$RESULTS_DIR/run/diag_no_compile.log"
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] TEST 1: compile=OFF, 1 GPU, CUDA_LAUNCH_BLOCKING=1"
    echo "  Config: $CFG"
    echo "  Log:    $LOG"
    echo "================================================================"
    (
      cd "$CODE_DIR"
      CUDA_VISIBLE_DEVICES=0 \
        "$DEEPSPEED_BIN" --num_gpus=1 main.py --config "$CFG"
    ) 2>&1 | tee "$LOG"
    echo ""
    echo "[$(date '+%H:%M:%S')] TEST 1 COMPLETE. Check $LOG for errors."
    ;;

  # ──────────────────────────────────────────────────────────────────
  # TEST 2: Compile ON, single GPU
  # Isolates: is torch.compile itself the problem?
  # ──────────────────────────────────────────────────────────────────
  compile_on)
    CFG="$TEST_ROOT/configs/diag_compile_on.yaml"
    LOG="$RESULTS_DIR/run/diag_compile_on_1gpu.log"
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] TEST 2: compile=ON, 1 GPU, CUDA_LAUNCH_BLOCKING=1"
    echo "  Config: $CFG"
    echo "  Log:    $LOG"
    echo "================================================================"
    (
      cd "$CODE_DIR"
      CUDA_VISIBLE_DEVICES=0 \
        "$DEEPSPEED_BIN" --num_gpus=1 main.py --config "$CFG"
    ) 2>&1 | tee "$LOG"
    echo ""
    echo "[$(date '+%H:%M:%S')] TEST 2 COMPLETE. Check $LOG for errors."
    ;;

  # ──────────────────────────────────────────────────────────────────
  # TEST 3: Compile ON, 8 GPU
  # Isolates: is it compile + multi-GPU interaction?
  # ──────────────────────────────────────────────────────────────────
  compile_on_8gpu)
    CFG="$TEST_ROOT/configs/diag_compile_on.yaml"
    LOG="$RESULTS_DIR/run/diag_compile_on_8gpu.log"
    NUM_GPUS="${NUM_GPUS:-8}"
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] TEST 3: compile=ON, ${NUM_GPUS} GPU, CUDA_LAUNCH_BLOCKING=1"
    echo "  Config: $CFG"
    echo "  Log:    $LOG"
    echo "================================================================"
    (
      cd "$CODE_DIR"
      "$DEEPSPEED_BIN" --num_gpus="$NUM_GPUS" main.py --config "$CFG"
    ) 2>&1 | tee "$LOG"
    echo ""
    echo "[$(date '+%H:%M:%S')] TEST 3 COMPLETE. Check $LOG for errors."
    ;;

  *)
    echo "Usage: bash run_diagnostic.sh {no_compile|compile_on|compile_on_8gpu}"
    echo ""
    echo "  no_compile       - Test 1: compile OFF, 1 GPU  (baseline)"
    echo "  compile_on       - Test 2: compile ON,  1 GPU  (compile isolation)"
    echo "  compile_on_8gpu  - Test 3: compile ON,  8 GPU  (full test)"
    exit 1
    ;;
esac
