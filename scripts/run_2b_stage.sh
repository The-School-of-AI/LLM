#!/usr/bin/env bash
# Run promoted 2B dense seed stage — ZeRO-1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export CFG="$ROOT_DIR/configs/train_2b.yaml"
SKIP_VERSION_CHECK=1 FORCE_REWRITE_INIT=1 bash "$ROOT_DIR/scripts/run_training.sh" "$@"
