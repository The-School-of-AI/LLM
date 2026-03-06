#!/usr/bin/env bash
# run_eval.sh — OLMES evaluation runner
# Clones/updates OLMES, installs deps via uv (GPU), then runs eval tasks.
#
# Usage:
#   ./run_eval.sh [OPTIONS]
#
# See README.md for full documentation.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TASKS="mmlu:mc::olmes mmlu_pro:mc::none triviaqa::olmes arc_challenge::olmes gsm8k::olmes minerva_math_500::olmo3:midtrain truthfulqa::olmo1 bbh:qa::none"
DEFAULT_MODEL="google/gemma-3-1b-it"
DEFAULT_OUTPUT_DIR="./results"
OLMES_REPO="https://github.com/allenai/olmes.git"
OLMES_DIR="olmes"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TASKS="${TASKS:-$DEFAULT_TASKS}"
MODEL="${MODEL:-$DEFAULT_MODEL}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-}"   # e.g. s3://my-bucket/results
MODEL_ARGS="${MODEL_ARGS:-}"                 # e.g. '{"trust_remote_code": true}'
                                             # Note: vLLM default sets trust_remote_code=false automatically
MODEL_TYPE="${MODEL_TYPE:-vllm}"               # hf | vllm | litellm
                                             # Note: vllm backend requires vLLM <0.8 for lm_eval compatibility
LIMIT="${LIMIT:-}"                           # optional: limit instances per task
DRY_RUN="${DRY_RUN:-false}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

All options can also be set as environment variables.

Options:
  -t, --tasks           Space-separated task list (env: TASKS)
                        Default: $DEFAULT_TASKS
  -m, --model           HuggingFace model ID or s3://bucket/path (env: MODEL)
                        Default: $DEFAULT_MODEL
  -o, --output-dir      Local output directory (env: OUTPUT_DIR)
                        Default: $DEFAULT_OUTPUT_DIR
  -r, --remote-output   S3 output URI, e.g. s3://bucket/results (env: REMOTE_OUTPUT_DIR)
  -a, --model-args      JSON model args, e.g. '{"trust_remote_code":true}' (env: MODEL_ARGS)
  --model-type          Backend: vllm | hf | litellm (env: MODEL_TYPE, default: vllm)
  --limit               Limit instances per task for quick tests (env: LIMIT)
  --dry-run             Print the olmes command without running it (env: DRY_RUN=true)
  -h, --help            Show this help message

Examples:
  # Run with defaults
  ./run_eval.sh

  # Custom HuggingFace model + local output
  ./run_eval.sh --model meta-llama/Llama-3.2-1B-Instruct --output-dir ./llama-results

  # Model from S3 + results to S3
  ./run_eval.sh --model s3://my-bucket/models/my-model --remote-output s3://my-bucket/results

  # Quick test with instance limit
  ./run_eval.sh --limit 10 --dry-run
EOF
}

# Positional arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t|--tasks)           TASKS="$2";            shift 2 ;;
    -m|--model)           MODEL="$2";            shift 2 ;;
    -o|--output-dir)      OUTPUT_DIR="$2";       shift 2 ;;
    -r|--remote-output)   REMOTE_OUTPUT_DIR="$2"; shift 2 ;;
    -a|--model-args)      MODEL_ARGS="$2";       shift 2 ;;
    --model-type)         MODEL_TYPE="$2";       shift 2 ;;
    --limit)              LIMIT="$2";            shift 2 ;;
    --dry-run)            DRY_RUN="true";        shift ;;
    -h|--help)            usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Step 1: Clone or update OLMES
# ---------------------------------------------------------------------------
echo "==> Checking OLMES repository..."
if [[ ! -d "$OLMES_DIR/.git" ]]; then
  echo "    Cloning $OLMES_REPO into ./$OLMES_DIR"
  git clone "$OLMES_REPO" "$OLMES_DIR"
else
  echo "    Found existing clone at ./$OLMES_DIR — pulling latest"
  git -C "$OLMES_DIR" pull --ff-only
fi

cd "$OLMES_DIR"

# ---------------------------------------------------------------------------
# Step 2: Install dependencies with uv (GPU group includes vLLM)
# ---------------------------------------------------------------------------
echo "==> Checking uv installation..."
if ! command -v uv &>/dev/null; then
  echo "    uv not found — installing via pip..."
  pip install uv --quiet
fi

echo "==> Installing OLMES dependencies (GPU group)..."
uv sync --group gpu

# ---------------------------------------------------------------------------
# Step 3: Resolve base output dir
# ---------------------------------------------------------------------------
# Resolve output dir relative to original working dir (we cd'd into olmes/)
OUTPUT_DIR_ABS="$(realpath --canonicalize-missing "../${OUTPUT_DIR#./}")"
# If the user passed an absolute path, use it directly
[[ "$OUTPUT_DIR" = /* ]] && OUTPUT_DIR_ABS="$OUTPUT_DIR"

# vLLM requires trust_remote_code as an explicit bool (not None).
if [[ -z "$MODEL_ARGS" && "$MODEL_TYPE" == "vllm" ]]; then
  MODEL_ARGS='{"trust_remote_code": false}'
fi

# ---------------------------------------------------------------------------
# Step 4: Print config summary
# ---------------------------------------------------------------------------
echo ""
echo "==> Evaluation configuration"
echo "    Model       : $MODEL"
echo "    Model type  : $MODEL_TYPE"
echo "    Tasks       : $TASKS"
echo "    Base output : $OUTPUT_DIR_ABS/<task_name>/"
[[ -n "$REMOTE_OUTPUT_DIR" ]] && echo "    Remote output: $REMOTE_OUTPUT_DIR/<task_name>/"
[[ -n "$MODEL_ARGS" ]]        && echo "    Model args  : $MODEL_ARGS"
[[ -n "$LIMIT" ]]             && echo "    Limit       : $LIMIT instances per task"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
  echo "==> Dry-run commands (one per task):"
  for task in $TASKS; do
    TASK_OUT="${OUTPUT_DIR_ABS}/${task}"
    CMD=(uv run olmes --model "$MODEL" #--model-type "$MODEL_TYPE"
      --task "$task" --output-dir "$TASK_OUT")
    [[ -n "$MODEL_ARGS" ]]        && CMD+=(--model-args "$MODEL_ARGS")
    [[ -n "$REMOTE_OUTPUT_DIR" ]] && CMD+=(--remote-output-dir "${REMOTE_OUTPUT_DIR}/${task}")
    [[ -n "$LIMIT" ]]             && CMD+=(--limit "$LIMIT")
    echo "    ${CMD[*]}"
  done
  echo ""
  echo "[dry-run] Exiting without executing."
  exit 0
fi

# ---------------------------------------------------------------------------
# Step 5: Run each task in its own output subfolder
# ---------------------------------------------------------------------------
FAILED_TASKS=()

for task in $TASKS; do
  TASK_OUT="${OUTPUT_DIR_ABS}/${task}"
  mkdir -p "$TASK_OUT"

  CMD=(uv run olmes
    --model "$MODEL"
    #--model-type "$MODEL_TYPE"
    --task "$task"
    --output-dir "$TASK_OUT"
  )
  [[ -n "$MODEL_ARGS" ]]        && CMD+=(--model-args "$MODEL_ARGS")
  [[ -n "$REMOTE_OUTPUT_DIR" ]] && CMD+=(--remote-output-dir "${REMOTE_OUTPUT_DIR}/${task}")
  [[ -n "$LIMIT" ]]             && CMD+=(--limit "$LIMIT")

  echo "==> Running task: $task"
  echo "    Output dir : $TASK_OUT"
  echo "    Command    : ${CMD[*]}"
  echo ""

  # vLLM v1 requires spawn start method to avoid CUDA re-init errors in forked subprocesses
  if VLLM_WORKER_MULTIPROC_METHOD=spawn "${CMD[@]}"; then
    echo "    [OK] $task"
  else
    echo "    [FAILED] $task — continuing with remaining tasks"
    FAILED_TASKS+=("$task")
  fi
  echo ""
done

echo "==> All tasks attempted. Results saved under: $OUTPUT_DIR_ABS/"
[[ -n "$REMOTE_OUTPUT_DIR" ]] && echo "    Remote copies under: $REMOTE_OUTPUT_DIR/"

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
  echo ""
  echo "WARNING: The following tasks failed:"
  for t in "${FAILED_TASKS[@]}"; do echo "    - $t"; done
  exit 1
fi
