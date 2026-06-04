#!/usr/bin/env bash
# Run 120B MoE with TQP — ZeRO-1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CFG="$ROOT_DIR/configs/train_120b_tqp.yaml"

# Create results directory
mkdir -p "$ROOT_DIR/results/120b_tqp/checkpoints"
export T19_REV_CKPT_USE_REENTRANT=1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SKIP_VERSION_CHECK=1 FORCE_REWRITE_INIT=0 bash "$ROOT_DIR/scripts/run_training.sh" "$@"
