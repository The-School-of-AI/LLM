#!/usr/bin/env bash
# Run 1B Non-Reversible Dense — ZeRO-1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CFG="$SCRIPT_DIR/configs/train_1b_nonrev_z1.yaml"
SKIP_VERSION_CHECK=1 FORCE_REWRITE_INIT=1 bash "$SCRIPT_DIR/run.sh" "$@"
