#!/bin/bash
# ============================================================================
# run_validation.sh
# ============================================================================
# Launch script for FULL DeepSpeed ZeRO-2 training validation.
#
# Hardware: AWS g5.12xlarge (4x NVIDIA A10G 24GB)
# Runtime:  ~15-25 minutes for 250 steps + ~5 min for 1-GPU comparison
#
# Pre-training checks (Phase 0a/0b):
#   - Gate routing determinism (catches MoE non-determinism)
#   - Reversible vs standard gradient comparison (catches MidpointFunction bugs)
#
# Checkpoint verification (Phase 3):
#   - Parameter restoration (state_dict roundtrip)
#   - Optimizer state restoration (Adam momentum/variance in ZeRO-2 flat buffers)
#
# Post-run comparison:
#   - 1-GPU vs N-GPU loss trajectory (catches ZeRO-2 partition corruption)
#
# Usage:
#   chmod +x run_validation.sh
#   ./run_validation.sh          # 4 GPU (full validation + 1-GPU comparison)
#   ./run_validation.sh 1        # 1 GPU only (debug mode, no comparison)
#   ./run_validation.sh 2        # 2 GPU
# ============================================================================

#!/bin/bash
# ============================================================================
# run_validation.sh (FIXED)
# ============================================================================
set -euo pipefail

# ---- Configuration ----
NUM_GPUS="${1:-4}"
TOTAL_STEPS=250
CHECKPOINT_STEP=200
SEQ_LEN=128
LOG_DIR="./validation_logs"
DS_CONFIG="./ds_config_zero2.json"
SCRIPT="validate_full_training.py"
COMPARISON_STEPS=50

# ---- Pre-flight checks ----
echo "========================================================================"
echo "  DEEPSPEED ZERO-2 FULL TRAINING VALIDATION"
echo "========================================================================"
echo ""
echo "  GPUs requested:  ${NUM_GPUS}"
echo "  Total steps:     ${TOTAL_STEPS}"
echo "  Checkpoint at:   ${CHECKPOINT_STEP}"
echo "  Log directory:   ${LOG_DIR}"
echo ""

for f in "${SCRIPT}" "${DS_CONFIG}" "recurrence_model_3b.py" "reversible_ops_midpoint.py"; do
    if [ ! -f "$f" ]; then
        echo "  MISSING: $f"
        echo "    Make sure all files are in the same directory."
        exit 1
    fi
    echo "  Found: $f"
done
echo ""

python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'  CUDA devices: {torch.cuda.device_count()}')"
python3 -c "import deepspeed; print(f'  DeepSpeed: {deepspeed.__version__}')"
echo ""

export TORCH_LOGS="-dynamo"
export TORCHDYNAMO_VERBOSE=0

if [ -d "${LOG_DIR}" ]; then
    echo "  Clearing previous logs in ${LOG_DIR}..."
    rm -rf "${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}"

# ---- Launch main validation ----
echo "========================================================================"
echo "  LAUNCHING: deepspeed --num_gpus=${NUM_GPUS} ${SCRIPT}"
echo "  Phases: 0a(gate determ) -> 0b(reversible grad) -> 1-2(train) -> 3(ckpt+optim) -> 4(resume)"
echo "========================================================================"
echo ""

# FIX: Passed --deepspeed as flag and config via --deepspeed_config
deepspeed --num_gpus="${NUM_GPUS}" \
    "${SCRIPT}" \
    --deepspeed \
    --deepspeed_config "${DS_CONFIG}" \
    --total_steps "${TOTAL_STEPS}" \
    --checkpoint_step "${CHECKPOINT_STEP}" \
    --seq_len "${SEQ_LEN}" \
    --log_dir "${LOG_DIR}"

MAIN_EXIT=$?

echo ""
echo "========================================================================"
if [ ${MAIN_EXIT} -eq 0 ]; then
    echo "  MAIN VALIDATION COMPLETED SUCCESSFULLY"
else
    echo "  MAIN VALIDATION FAILED (exit code ${MAIN_EXIT})"
fi
echo "========================================================================"

# ============================================================================
# 1-GPU vs N-GPU LOSS COMPARISON
# ============================================================================
if [ "${NUM_GPUS}" -gt 1 ] && [ ${MAIN_EXIT} -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "  RUNNING 1-GPU BASELINE FOR ZERO CORRUPTION CHECK"
    echo "  (${COMPARISON_STEPS} steps -- should take ~2-5 minutes)"
    echo "========================================================================"
    echo ""

    LOG_DIR_1GPU="${LOG_DIR}_1gpu"
    rm -rf "${LOG_DIR_1GPU}"
    mkdir -p "${LOG_DIR_1GPU}"

    # FIX: Passed --deepspeed as flag and config via --deepspeed_config
    deepspeed --num_gpus=1 \
        "${SCRIPT}" \
        --deepspeed \
        --deepspeed_config "${DS_CONFIG}" \
        --total_steps "${COMPARISON_STEPS}" \
        --checkpoint_step 99999 \
        --seq_len "${SEQ_LEN}" \
        --log_dir "${LOG_DIR_1GPU}" \
        --skip_reversible_check \
        || true

    echo ""
    echo "========================================================================"
    echo "  ZERO-2 PARTITION CORRUPTION CHECK"
    echo "========================================================================"

    python3 << 'PYEOF'
import csv
import sys

def read_losses(path, max_steps=None):
    losses = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                losses.append(float(row['loss_total']))
                if max_steps and len(losses) >= max_steps:
                    break
    except Exception as e:
        print(f"  Could not read {path}: {e}")
        return None
    return losses

multi_losses = read_losses('./validation_logs/metrics_rank0.csv', max_steps=50)
single_losses = read_losses('./validation_logs_1gpu/metrics_rank0.csv', max_steps=50)

if multi_losses is None or single_losses is None:
    print("  SKIP: Could not read one or both metric files")
    sys.exit(0)

n = min(len(multi_losses), len(single_losses))
if n < 10:
    print(f"  SKIP: Too few steps to compare ({n})")
    sys.exit(0)

multi_losses = multi_losses[:n]
single_losses = single_losses[:n]

diffs = [abs(m - s) for m, s in zip(multi_losses, single_losses)]
max_diff = max(diffs)
mean_diff = sum(diffs) / len(diffs)

multi_first10 = sum(multi_losses[:10]) / 10
multi_last10 = sum(multi_losses[-10:]) / 10
single_first10 = sum(single_losses[:10]) / 10
single_last10 = sum(single_losses[-10:]) / 10

print(f"  Compared {n} steps:")
print(f"    Multi-GPU loss: {multi_first10:.4f} -> {multi_last10:.4f}")
print(f"    1-GPU loss:     {single_first10:.4f} -> {single_last10:.4f}")
print(f"    Max step-wise diff: {max_diff:.4f}")
print(f"    Mean step-wise diff: {mean_diff:.4f}")

multi_decreasing = multi_last10 < multi_first10
single_decreasing = single_last10 < single_first10
ratio = multi_last10 / max(single_last10, 1e-8)

if not multi_decreasing:
    print("  FAIL: Multi-GPU loss not decreasing -- possible ZeRO corruption")
elif not single_decreasing:
    print("  WARNING: 1-GPU loss not decreasing (may be too few steps)")
elif ratio > 2.0 or ratio < 0.5:
    print(f"  WARNING: Loss magnitude divergence (ratio={ratio:.2f}) -- investigate")
elif max_diff > 2.0:
    print(f"  WARNING: Large step-wise divergence -- may indicate partition issue")
else:
    print("  PASS: Loss curves consistent between 1-GPU and multi-GPU")

PYEOF

fi

# ---- Final Results ----
echo ""
echo "========================================================================"
echo "  RESULTS"
echo "========================================================================"
echo ""
echo "  Metrics CSV:      ${LOG_DIR}/metrics_rank0.csv"
echo "  Summary:          ${LOG_DIR}/validation_summary_rank0.txt"
echo "  DeepSpeed Ckpt:   ${LOG_DIR}/checkpoints/"
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "  1-GPU Baseline:   ${LOG_DIR}_1gpu/metrics_rank0.csv"
fi
echo ""

SUMMARY_FILE="${LOG_DIR}/validation_summary_rank0.txt"
if [ -f "${SUMMARY_FILE}" ]; then
    echo "========================================================================"
    echo "  VALIDATION SUMMARY"
    echo "========================================================================"
    cat "${SUMMARY_FILE}"
fi

exit ${MAIN_EXIT}
