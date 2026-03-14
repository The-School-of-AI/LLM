#!/usr/bin/env bash
# Run 8B MoE — ZeRO-3
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CFG="$SCRIPT_DIR/configs/train_8b_moe_z3.yaml"
SKIP_VERSION_CHECK=1 FORCE_REWRITE_INIT=1 bash "$SCRIPT_DIR/run.sh" "$@"
