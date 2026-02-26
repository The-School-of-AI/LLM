#!/bin/bash
################################################################################
# Profiling Orchestrator for Recurrence Model 1B Training
################################################################################
#
# Runs Nsight Systems, Nsight Compute, and/or PyTorch Profiler on training
# scripts in the profiling/ folder. Each profiling type outputs to its own
# directory: output_nsys/, output_ncu/, output_pytorch/.
#
# Usage:
#   # Run ALL three profilers on train_profile.py
#   ./run_profiling.sh --mode all
#
#   # Run only Nsight Systems
#   ./run_profiling.sh --mode nsys
#
#   # Run only PyTorch Profiler
#   ./run_profiling.sh --mode pytorch
#
#   # Run only Nsight Compute
#   ./run_profiling.sh --mode ncu
#
#   # Run on a specific script (instead of default train_profile.py)
#   ./run_profiling.sh --mode all --script my_training.py
#
#   # Run on ALL python scripts in profiling/ folder
#   ./run_profiling.sh --mode all --all-scripts
#
#   # Override training hyperparameters
#   ./run_profiling.sh --mode all --batch-size 4 --seq-length 1024
#
################################################################################

set -euo pipefail

# =============================================================================
# Configuration Defaults
# =============================================================================
BATCH_SIZE=8
SEQ_LENGTH=2048
MAX_STEPS=20
WARMUP_STEPS=10
NUM_PROFILE_STEPS=5
GRAD_ACCUM=1
LR="1e-4"

# Script resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SCRIPT="train_profile.py"

# Profiling mode: nsys, ncu, pytorch, or all
MODE="nsys,pytorch"
TARGET_SCRIPT=""
ALL_SCRIPTS=false

# =============================================================================
# Parse Arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --script)
            TARGET_SCRIPT="$2"; shift 2 ;;
        --all-scripts)
            ALL_SCRIPTS=true; shift ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --seq-length)
            SEQ_LENGTH="$2"; shift 2 ;;
        --max-steps)
            MAX_STEPS="$2"; shift 2 ;;
        --warmup-steps)
            WARMUP_STEPS="$2"; shift 2 ;;
        --num-profile-steps)
            NUM_PROFILE_STEPS="$2"; shift 2 ;;
        --grad-accum)
            GRAD_ACCUM="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE         Profiling mode: nsys, ncu, pytorch, all (default: nsys,pytorch)"
            echo "  --script FILE       Target script in profiling/ dir (default: train_profile.py)"
            echo "  --all-scripts       Run on all *.py scripts in profiling/ dir"
            echo "  --batch-size N      Batch size (default: $BATCH_SIZE)"
            echo "  --seq-length N      Sequence length (default: $SEQ_LENGTH)"
            echo "  --max-steps N       Max training steps (default: $MAX_STEPS)"
            echo "  --warmup-steps N    Warmup steps before profiling (default: $WARMUP_STEPS)"
            echo "  --num-profile-steps N  Steps to profile after warmup (default: $NUM_PROFILE_STEPS)"
            echo "  --grad-accum N      Gradient accumulation (default: $GRAD_ACCUM)"
            echo "  --lr RATE           Learning rate (default: $LR)"
            echo "  -h, --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate mode — supports comma-separated: nsys,pytorch
IFS=',' read -ra MODES <<< "$MODE"
for m in "${MODES[@]}"; do
    case "$m" in
        nsys|ncu|pytorch|all) ;;
        *)
            echo "❌ Invalid mode: $m"
            echo "   Must be one of: nsys, ncu, pytorch, all (comma-separated OK, e.g. nsys,pytorch)"
            exit 1
            ;;
    esac
done

# Helper: check if a mode is active
mode_active() {
    for m in "${MODES[@]}"; do
        [[ "$m" == "$1" || "$m" == "all" ]] && return 0
    done
    return 1
}

# =============================================================================
# Build script list
# =============================================================================
SCRIPTS=()

if [ "$ALL_SCRIPTS" = true ]; then
    for f in "$SCRIPT_DIR"/*.py; do
        [ -f "$f" ] && SCRIPTS+=("$(basename "$f")")
    done
    if [ ${#SCRIPTS[@]} -eq 0 ]; then
        echo "❌ No Python scripts found in $SCRIPT_DIR"
        exit 1
    fi
elif [ -n "$TARGET_SCRIPT" ]; then
    SCRIPTS=("$TARGET_SCRIPT")
else
    SCRIPTS=("$DEFAULT_SCRIPT")
fi

# =============================================================================
# Tool availability checks
# =============================================================================
HAS_NSYS=false
HAS_NCU=false

# Helper to install packages if missing
install_if_missing() {
    local cmd="$1"
    local pkg="$2"
    if ! command -v "$cmd" &>/dev/null; then
        echo "⚠️  $cmd not found — attempting to install $pkg..."
        if apt-get update && apt-get install -y "$pkg"; then
            echo "✅ Installed $pkg"
        else
            echo "❌ Failed to install $pkg"
        fi
    fi
}

# Check and Install Nsight Systems
install_if_missing "nsys" "nsight-systems-2025.5.2"

if command -v nsys &>/dev/null; then
    HAS_NSYS=true
    echo "✅ nsys found: $(which nsys)"
else
    echo "⚠️  nsys not found — Nsight Systems profiling will be skipped"
fi

# Check and Install Nsight Compute
install_if_missing "ncu" "nsight-compute-2025.4.1"

if command -v ncu &>/dev/null; then
    HAS_NCU=true
    echo "✅ ncu found: $(which ncu)"
else
    echo "⚠️  ncu not found — Nsight Compute profiling will be skipped"
fi

echo "✅ PyTorch Profiler: always available (built-in)"
echo ""

# =============================================================================
# Common training args (passed to every invocation)
# =============================================================================
TRAIN_ARGS=(
    --batch-size "$BATCH_SIZE"
    --seq-length "$SEQ_LENGTH"
    --max-steps "$MAX_STEPS"
    --warmup-steps "$WARMUP_STEPS"
    --num-profile-steps "$NUM_PROFILE_STEPS"
    --grad-accum "$GRAD_ACCUM"
    --lr "$LR"
)

# =============================================================================
# Profiling Functions
# =============================================================================

run_nsys() {
    local script_path="$1"
    local script_name="$2"
    local output_dir="$SCRIPT_DIR/output_nsys/${script_name}"
    mkdir -p "$output_dir"

    echo "============================================================"
    echo "🔬 NSIGHT SYSTEMS — $script_name"
    echo "============================================================"
    echo "  Output: $output_dir"
    echo ""

    nsys profile \
        --trace=cuda,nvtx,osrt,cudnn,cublas \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        -o "${output_dir}/${script_name}" \
        --force-overwrite=true \
        uv run python "$script_path" \
            "${TRAIN_ARGS[@]}" \
            --profile-mode nsys \
            --profile-output-dir "$output_dir"

    echo ""
    echo "✅ Nsight Systems complete → $output_dir"
    echo "   View: nsys stats ${output_dir}/${script_name}.nsys-rep"
    echo ""
}

run_ncu() {
    local script_path="$1"
    local script_name="$2"
    local output_dir="$SCRIPT_DIR/output_ncu/${script_name}"
    mkdir -p "$output_dir"

    # ncu-specific overrides: keep it short since ncu replays every kernel
    local NCU_WARMUP=3
    local NCU_PROFILE=2

    echo "============================================================"
    echo "🔬 NSIGHT COMPUTE — $script_name"
    echo "============================================================"
    echo "  Output: $output_dir"
    echo "  ncu warmup: ${NCU_WARMUP} steps, profile: ${NCU_PROFILE} steps"
    echo "  (using --set default for faster collection)"
    echo ""

    ncu \
        --set default \
        --replay-mode kernel \
        --target-processes all \
        --launch-skip-before-match 0 \
        -o "${output_dir}/${script_name}" \
        --force-overwrite \
        uv run python "$script_path" \
            --batch-size "$BATCH_SIZE" \
            --seq-length "$SEQ_LENGTH" \
            --max-steps "$((NCU_WARMUP + NCU_PROFILE + 1))" \
            --warmup-steps "$NCU_WARMUP" \
            --num-profile-steps "$NCU_PROFILE" \
            --grad-accum "$GRAD_ACCUM" \
            --lr "$LR" \
            --profile-mode ncu \
            --profile-output-dir "$output_dir"

    echo ""
    echo "✅ Nsight Compute complete → $output_dir"
    echo "   View: ncu-ui ${output_dir}/${script_name}.ncu-rep"
    echo ""
}

run_pytorch() {
    local script_path="$1"
    local script_name="$2"
    local output_dir="$SCRIPT_DIR/output_pytorch/${script_name}"
    mkdir -p "$output_dir"

    echo "============================================================"
    echo "🔬 PYTORCH PROFILER — $script_name"
    echo "============================================================"
    echo "  Output: $output_dir"
    echo ""

    uv run python "$script_path" \
        "${TRAIN_ARGS[@]}" \
        --profile-mode pytorch \
        --profile-output-dir "$output_dir"

    echo ""
    echo "✅ PyTorch Profiler complete → $output_dir"
    echo "   View: tensorboard --logdir $output_dir"
    echo ""
}

# =============================================================================
# Main Execution
# =============================================================================

echo "============================================================"
echo "PROFILING ORCHESTRATOR"
echo "============================================================"
echo "  Mode:          $MODE"
echo "  Scripts:       ${SCRIPTS[*]}"
echo "  Batch Size:       $BATCH_SIZE"
echo "  Seq Length:       $SEQ_LENGTH"
echo "  Max Steps:        $MAX_STEPS"
echo "  Warmup Steps:     $WARMUP_STEPS"
echo "  Profile Steps:    $NUM_PROFILE_STEPS"
echo "  Project Root:     $PROJECT_ROOT"
echo "============================================================"
echo ""

TOTAL_RUNS=0
SUCCEEDED=0
FAILED=0

for script_basename in "${SCRIPTS[@]}"; do
    script_path="$SCRIPT_DIR/$script_basename"
    # Strip .py extension for folder naming
    script_name="${script_basename%.py}"

    if [ ! -f "$script_path" ]; then
        echo "❌ Script not found: $script_path — skipping"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "📄 Processing: $script_basename"
    echo "=========================================="

    # --- Nsight Systems ---
    if mode_active nsys; then
        if [ "$HAS_NSYS" = true ]; then
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            if run_nsys "$script_path" "$script_name"; then
                SUCCEEDED=$((SUCCEEDED + 1))
            else
                echo "❌ Nsight Systems failed for $script_basename"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "⏭️  Skipping Nsight Systems (nsys not installed)"
        fi
    fi

    # --- Nsight Compute ---
    if mode_active ncu; then
        if [ "$HAS_NCU" = true ]; then
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            if run_ncu "$script_path" "$script_name"; then
                SUCCEEDED=$((SUCCEEDED + 1))
            else
                echo "❌ Nsight Compute failed for $script_basename"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "⏭️  Skipping Nsight Compute (ncu not installed)"
        fi
    fi

    # --- PyTorch Profiler ---
    if mode_active pytorch; then
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        if run_pytorch "$script_path" "$script_name"; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            echo "❌ PyTorch Profiler failed for $script_basename"
            FAILED=$((FAILED + 1))
        fi
    fi

done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PROFILING COMPLETE"
echo "============================================================"
echo "  Total runs:  $TOTAL_RUNS"
echo "  Succeeded:   $SUCCEEDED"
echo "  Failed:      $FAILED"
echo ""
echo "Output directories:"
if mode_active nsys; then
    echo "  Nsight Systems:  $SCRIPT_DIR/output_nsys/"
fi
if mode_active ncu; then
    echo "  Nsight Compute:  $SCRIPT_DIR/output_ncu/"
fi
if mode_active pytorch; then
    echo "  PyTorch Profiler: $SCRIPT_DIR/output_pytorch/"
fi
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
