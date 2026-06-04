#!/usr/bin/env bash
# =============================================================================
# Resume Test Script
#
# Tests that checkpoint resume produces continuous loss (no spike).
#
# Usage:
#   bash scripts/test_resume.sh [CHECKPOINT_TAG]
#
# If CHECKPOINT_TAG is not provided, it finds the latest checkpoint in S3.
#
# What it does:
#   1. Finds the latest checkpoint tag (or uses the one you provide)
#   2. Downloads shard_metadata.json to show shard state
#   3. Creates a temporary config with resume_from_checkpoint set
#   4. Runs training for 20 steps from the checkpoint
#   5. You compare the first few steps' loss with the pre-checkpoint loss
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_CFG="${BASE_CFG:-$ROOT_DIR/configs/train_1b_nonrev_z1.yaml}"
S3_BUCKET="t1-dataacquisition-checkpoints-2"
S3_PREFIX="training/checkpoints"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_STEPS="${MAX_STEPS:-20}"

# ---- Resolve checkpoint tag ----
if [[ -n "${1:-}" ]]; then
    CKPT_TAG="$1"
    echo "[resume-test] Using provided checkpoint tag: $CKPT_TAG"
else
    echo "[resume-test] Finding latest checkpoint in s3://$S3_BUCKET/$S3_PREFIX/ ..."
    CKPT_TAG=$(aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/" \
        | grep 'PRE step_' \
        | awk '{print $2}' \
        | tr -d '/' \
        | sort -t_ -k2 -n \
        | tail -1)
    if [[ -z "$CKPT_TAG" ]]; then
        echo "[resume-test] ERROR: No checkpoints found in S3."
        exit 1
    fi
    echo "[resume-test] Latest checkpoint: $CKPT_TAG"
fi

# ---- Extract step number from tag ----
CKPT_STEP=$(echo "$CKPT_TAG" | grep -oP '(?<=step_)\d+')
echo "[resume-test] Checkpoint step: $CKPT_STEP"

# ---- Show shard metadata ----
echo ""
echo "[resume-test] Shard metadata from checkpoint:"
aws s3 cp "s3://$S3_BUCKET/$S3_PREFIX/$CKPT_TAG/shard_metadata.json" /dev/stdout 2>/dev/null \
    | "$PYTHON_BIN" -m json.tool \
    || echo "  (no shard_metadata.json found)"
echo ""

# ---- Create resume config ----
RESUME_CFG="/tmp/resume_test_config.yaml"
"$PYTHON_BIN" -c "
import yaml, sys, os

with open('$BASE_CFG') as f:
    cfg = yaml.safe_load(f)

cfg_dir = os.path.dirname(os.path.abspath('$BASE_CFG'))

# Resolve all relative paths to absolute (since temp config lives in /tmp)
def resolve(p):
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(cfg_dir, p))

data = cfg.get('data', {})
for key in ['shard_dir', 'eval_shard_dir', 'curriculum_config_path', 'manifest_dir']:
    if data.get(key):
        data[key] = resolve(data[key])

# Set resume fields
cfg['checkpoint']['resume_from_checkpoint'] = '$CKPT_TAG'
cfg['checkpoint']['resume_step'] = $CKPT_STEP

# Override max_steps to only run a few steps after resume
cfg['training']['max_steps'] = $CKPT_STEP + $MAX_STEPS

# Use a shorter periodic checkpoint interval for the test (5 min)
cfg['spot_checkpoint']['checkpoint_interval_seconds'] = 300

with open('$RESUME_CFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f'[resume-test] Config written to $RESUME_CFG')
print(f'[resume-test] Will resume from step {$CKPT_STEP} and run {$MAX_STEPS} more steps (to step {$CKPT_STEP + $MAX_STEPS})')
"

# ---- Show key config values ----
echo ""
echo "[resume-test] Key config values:"
"$PYTHON_BIN" -c "
import yaml
with open('$RESUME_CFG') as f:
    cfg = yaml.safe_load(f)
print(f\"  resume_from_checkpoint: {cfg['checkpoint']['resume_from_checkpoint']}\")
print(f\"  resume_step:            {cfg['checkpoint']['resume_step']}\")
print(f\"  max_steps:              {cfg['training']['max_steps']}\")
print(f\"  checkpoint_interval_s:  {cfg['spot_checkpoint']['checkpoint_interval_seconds']}\")
"
echo ""

# ---- Run training with resume ----
echo "[resume-test] Starting resumed training..."
echo "[resume-test] Watch for:"
echo "  1. 'Resumed from epoch X, step Y' message"
echo "  2. 'Curriculum shard state restored from checkpoint' message"
echo "  3. First few steps' loss should be close to pre-checkpoint loss (~7.x)"
echo "  4. No loss spike back to 10+"
echo ""
echo "=========================================="

CFG="$RESUME_CFG" bash "$ROOT_DIR/run.sh"
