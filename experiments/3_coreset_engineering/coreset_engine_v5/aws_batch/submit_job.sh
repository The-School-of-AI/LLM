#!/usr/bin/env bash
# =============================================================================
# Coreset Engine: AWS Batch Job Submission Script
# =============================================================================
# Use this script for subsequent runs once infrastructure is already deployed.
# It skips Docker builds and IAM/Queue provisioning.
#
# Usage:
#   ./aws_batch/submit_job.sh --bucket <my-bucket> --input <s3-path>
# =============================================================================

set -euo pipefail

# --- Defaults ---
REGION=$(aws configure get region || echo "ap-south-1")
JOB_DEFINITION="coreset-engine"
JOB_QUEUE="coreset-engine-queue"
JOB_NAME="coreset-engine-run"

# Sharding
NUM_SHARDS=8
STAGES="1B 3B 8B 70B"
TOTAL_TOKENS="4523096944"

# Pipeline Options
INPUT_FORMAT="jsonl"
BATCH_SIZE=10000
BAND_INFERENCE="none"
BAND_SCORE_SOURCE="auto"
STAGE_TARGET_SCALE="1.0"
RESUME="false"

function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options (Required):"
    echo "  --bucket   S3 Bucket name for outputs/checkpoints"
    echo "  --input    S3 Input path (e.g. s3://bucket/data/)"
    echo ""
    echo "Options (Optional):"
    echo "  --shards   Number of array shards (default: $NUM_SHARDS)"
    echo "  --stages   Space-separated stage list (default: \"$STAGES\")"
    echo "  --tokens   Total tokens in dataset (default: $TOTAL_TOKENS)"
    echo "  --resume   Set to 'true' to resume from checkpoints"
    echo "  --queue    Name or ARN of Job Queue (default: $JOB_QUEUE)"
    echo "  --job-def  Name or ARN of Job Definition (default: $JOB_DEFINITION)"
    echo "  --region   AWS Region (default: $REGION)"
    echo ""
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bucket) BUCKET="$2"; shift 2 ;;
        --input) INPUT_PATH="$2"; shift 2 ;;
        --shards) NUM_SHARDS="$2"; shift 2 ;;
        --stages) STAGES="$2"; shift 2 ;;
        --tokens) TOTAL_TOKENS="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --queue) JOB_QUEUE="$2"; shift 2 ;;
        --job-def) JOB_DEFINITION="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

if [[ -z "${BUCKET:-}" || -z "${INPUT_PATH:-}" ]]; then
    echo "ERROR: --bucket and --input are required."
    show_help
    exit 1
fi

echo "============================================================"
echo "  Submitting AWS Batch Array Job"
echo "  Job Name     : ${JOB_NAME}"
echo "  Job Queue    : ${JOB_QUEUE}"
echo "  Job Def      : ${JOB_DEFINITION}"
echo "  Array Size   : ${NUM_SHARDS} shards"
echo "  Stages       : ${STAGES}"
echo "  Input        : ${INPUT_PATH}"
echo "============================================================"

# --- Build Container Environment Overrides ---
CONTAINER_ENV=$(cat <<EOF
[
  {"name": "S3_BUCKET",            "value": "${BUCKET}"},
  {"name": "S3_INPUT_PATH",        "value": "${INPUT_PATH}"},
  {"name": "TOTAL_TOKENS",         "value": "${TOTAL_TOKENS}"},
  {"name": "NUM_SHARDS",           "value": "${NUM_SHARDS}"},
  {"name": "STAGES",               "value": "${STAGES}"},
  {"name": "INPUT_FORMAT",         "value": "${INPUT_FORMAT}"},
  {"name": "BATCH_SIZE",           "value": "${BATCH_SIZE}"},
  {"name": "BAND_INFERENCE",       "value": "${BAND_INFERENCE}"},
  {"name": "BAND_SCORE_SOURCE",    "value": "${BAND_SCORE_SOURCE}"},
  {"name": "STAGE_TARGET_SCALE",   "value": "${STAGE_TARGET_SCALE}"},
  {"name": "RESUME",               "value": "${RESUME}"}
]
EOF
)

# Check if Job Definition exists
if ! aws batch describe-job-definitions --job-definition-name "${JOB_DEFINITION}" --region "${REGION}" --status ACTIVE --query "jobDefinitions[0]" --output text | grep -q "${JOB_DEFINITION}" 2>/dev/null; then
    echo "❌ ERROR: Job Definition '${JOB_DEFINITION}' not found in ${REGION}."
    echo "💡 Have you run './aws_batch/deploy_infra_and_run.sh' yet to create the infrastructure?"
    exit 1
fi

# Submit the Job
RESPONSE=$(aws batch submit-job \
  --job-name "${JOB_NAME}" \
  --job-queue "${JOB_QUEUE}" \
  --job-definition "${JOB_DEFINITION}" \
  --array-properties "size=${NUM_SHARDS}" \
  --container-overrides "{\"environment\": ${CONTAINER_ENV}}" \
  --region "${REGION}" \
  --output json)

JOB_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['jobId'])")

echo ""
echo "  ✅ Job submitted successfully!"
echo "  Job ID: ${JOB_ID}"
echo "  Monitor: aws batch describe-jobs --jobs ${JOB_ID} --region ${REGION}"
echo "============================================================"
