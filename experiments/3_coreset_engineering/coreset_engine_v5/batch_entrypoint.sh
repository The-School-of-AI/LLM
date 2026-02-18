#!/usr/bin/env bash
# =============================================================================
# AWS Batch Entrypoint for Coreset Selection Engine
# =============================================================================
#
# This script is the per-container equivalent of the inner loop body in shard.sh.
# AWS Batch Array Jobs set AWS_BATCH_JOB_ARRAY_INDEX automatically on each
# container (0..N-1), which maps directly to --shard-id.
#
# Required environment variables:
#   TOTAL_TOKENS       - Total token count of the input dataset
#   NUM_SHARDS         - Must equal the AWS Batch arrayProperties.size
#
# Optional (with defaults):
#   S3_BUCKET          - S3 bucket name (default: t2-datacurriculum-353)
#   S3_INPUT_PATH      - Full S3 prefix for input data
#                        (default: s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/source=books/)
#
# Optional environment variables (with defaults):
#   STAGES             - Space-separated stage list (default: "1B 3B 8B 70B")
#   INPUT_FORMAT       - Input format: jsonl or parquet (default: jsonl)
#   CONFIG             - Pipeline config path (default: config/pipeline.yaml)
#   CURRICULUM         - Curriculum config path (default: config/curriculum.yaml)
#   CHECKPOINT_PREFIX  - S3 prefix for checkpoints
#                        (default: s3://${S3_BUCKET}/coreset-checkpoints)
#   OUTPUT_PREFIX      - S3 prefix for output coresets
#                        (default: s3://${S3_BUCKET}/coreset-output)
#   BATCH_SIZE         - Chunks per batch (default: 10000)
#   BAND_INFERENCE     - Band inference mode (default: none)
#   BAND_SCORE_SOURCE  - Band score source (default: auto)
#   STAGE_TARGET_SCALE - Scale factor for stage targets (default: 1.0)
#   RESUME             - Set to "true" to resume from last checkpoint (default: false)
#
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve shard identity from AWS Batch Array Job index
# ---------------------------------------------------------------------------
SHARD_ID="${AWS_BATCH_JOB_ARRAY_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"

# ---------------------------------------------------------------------------
# Required variables
# ---------------------------------------------------------------------------
: "${TOTAL_TOKENS:?ERROR: TOTAL_TOKENS environment variable is required}"

# ---------------------------------------------------------------------------
# S3 configuration (with sensible defaults)
# ---------------------------------------------------------------------------
S3_BUCKET="${S3_BUCKET:-t2-datacurriculum-353}"
S3_INPUT_PATH="${S3_INPUT_PATH:-s3://${S3_BUCKET}/processed_dataset/curriculum_pyspark_output/source=books/}"

# ---------------------------------------------------------------------------
# Optional variables with defaults
# ---------------------------------------------------------------------------
STAGES="${STAGES:-1B 3B 8B 70B}"
INPUT_FORMAT="${INPUT_FORMAT:-jsonl}"
CONFIG="${CONFIG:-config/pipeline.yaml}"
CURRICULUM="${CURRICULUM:-config/curriculum.yaml}"
CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-s3://${S3_BUCKET}/coreset-checkpoints}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-s3://${S3_BUCKET}/coreset-output}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
BAND_INFERENCE="${BAND_INFERENCE:-none}"
BAND_SCORE_SOURCE="${BAND_SCORE_SOURCE:-auto}"
STAGE_TARGET_SCALE="${STAGE_TARGET_SCALE:-1.0}"
RESUME="${RESUME:-false}"

# ---------------------------------------------------------------------------
# Per-shard checkpoint directory (unique per shard to avoid collisions)
# ---------------------------------------------------------------------------
SHARD_PAD=$(printf '%03d' "$SHARD_ID")
CHECKPOINT_DIR="${CHECKPOINT_PREFIX}/shard${SHARD_PAD}"

echo "============================================================"
echo "  Coreset Engine - AWS Batch Shard"
echo "  Shard ID     : ${SHARD_ID} / ${NUM_SHARDS}"
echo "  Stages       : ${STAGES}"
echo "  Input        : ${S3_INPUT_PATH} (${INPUT_FORMAT})"
echo "  Checkpoints  : ${CHECKPOINT_DIR}"
echo "  Output       : ${OUTPUT_PREFIX}"
echo "  Total Tokens : ${TOTAL_TOKENS}"
echo "  Batch Size   : ${BATCH_SIZE}"
echo "  Band Infer   : ${BAND_INFERENCE}"
echo "  Band Score   : ${BAND_SCORE_SOURCE}"
echo "  Scale        : ${STAGE_TARGET_SCALE}"
echo "  Resume       : ${RESUME}"
echo "============================================================"


# ---------------------------------------------------------------------------
# Run the coreset builder for this shard
# ---------------------------------------------------------------------------
python coreset_builder.py \
  --config "${CONFIG}" \
  --curriculum "${CURRICULUM}" \
  --input-path "${S3_INPUT_PATH}" \
  --input-format "${INPUT_FORMAT}" \
  --stages ${STAGES} \
  --num-shards "${NUM_SHARDS}" \
  --shard-id "${SHARD_ID}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --total-input-tokens-estimate "${TOTAL_TOKENS}" \
  --band-inference "${BAND_INFERENCE}" \
  --band-score-source "${BAND_SCORE_SOURCE}" \
  --stage-target-scale "${STAGE_TARGET_SCALE}" \
  --output-coreset-path "${OUTPUT_PREFIX}" \
  --output-manifest-path "s3://${S3_BUCKET}/coreset-manifests"

EXIT_CODE=$?

echo "============================================================"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "  Shard ${SHARD_ID} completed successfully."
else
  echo "  ERROR: Shard ${SHARD_ID} failed with exit code ${EXIT_CODE}."
fi
echo "  Manifests : ${OUTPUT_PREFIX}/*/manifest_shard${SHARD_PAD}.json"
echo "============================================================"

exit $EXIT_CODE
