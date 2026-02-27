#!/usr/bin/env bash
# =============================================================================
# run_new_datasets.sh
# Submit one EMR Serverless job per source, all in parallel.
#
# Usage:
#   ./run_new_datasets.sh                      # run all sources
#   ./run_new_datasets.sh gsm8k smoltalk       # run specific sources
#
# Prerequisites:
#   aws configure  (with credentials that can submit EMR Serverless jobs)
#   EMR Serverless application already created
#   Script uploaded to S3
#
# Set the variables below before running.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
APP_ID="${EMR_APP_ID:-}"                          # export EMR_APP_ID=00xxxx before calling
JOB_ROLE_ARN="${EMR_JOB_ROLE_ARN:-}"             # export EMR_JOB_ROLE_ARN=arn:aws:iam::...
SCRIPT_S3="s3://t2-datacurriculum-353/scripts/t2_curated_datasets_curriculum.py"
INPUT_BASE="s3://t1-dataacquisition-datasets/processed_dataset/normalized_data"
OUTPUT_BASE="s3://t2-datacurriculum-353/processed_dataset/curriculum_data/"
REGION="us-east-1"
LOG_URI="s3://t2-datacurriculum-353/emr-serverless-logs/"

# ── Validate required vars ────────────────────────────────────────────────────
if [[ -z "$APP_ID" || -z "$JOB_ROLE_ARN" ]]; then
    echo "ERROR: set EMR_APP_ID and EMR_JOB_ROLE_ARN before running."
    echo "  export EMR_APP_ID=<your-emr-serverless-app-id>"
    echo "  export EMR_JOB_ROLE_ARN=arn:aws:iam::<account>:role/<role>"
    exit 1
fi

# ── Source → worker config ────────────────────────────────────────────────────
# Format: "source_name cpu_cores memory_gb"
# These are all small datasets (<5 GB), so tiny clusters are fine.
# Adjust if you know the actual data size from T1.
declare -A SOURCES_CPU
declare -A SOURCES_MEM

# MATH / REASONING (compute-intensive inference signals)
SOURCES_CPU["nemotron_math"]="4";     SOURCES_MEM["nemotron_math"]="16"
SOURCES_CPU["ultradata_math"]="4";    SOURCES_MEM["ultradata_math"]="16"
SOURCES_CPU["skywork_reward"]="4";    SOURCES_MEM["skywork_reward"]="16"
SOURCES_CPU["hardgen"]="2";           SOURCES_MEM["hardgen"]="8"
SOURCES_CPU["teichai"]="2";           SOURCES_MEM["teichai"]="8"
SOURCES_CPU["gsm8k"]="2";             SOURCES_MEM["gsm8k"]="8"

# CODE
SOURCES_CPU["ling_coder"]="4";        SOURCES_MEM["ling_coder"]="16"

# SCIENCE / INSTRUCTION
SOURCES_CPU["megascience"]="4";       SOURCES_MEM["megascience"]="16"
SOURCES_CPU["helpsteer3"]="2";        SOURCES_MEM["helpsteer3"]="8"
SOURCES_CPU["nemotron_post_training"]="4"; SOURCES_MEM["nemotron_post_training"]="16"

# PREFERENCE / SFT MIXES
SOURCES_CPU["open_perfectblend"]="2"; SOURCES_MEM["open_perfectblend"]="8"
SOURCES_CPU["orpo_dpo_mix"]="2";      SOURCES_MEM["orpo_dpo_mix"]="8"
SOURCES_CPU["ultrafeedback"]="2";     SOURCES_MEM["ultrafeedback"]="8"
SOURCES_CPU["infinity_preference"]="2"; SOURCES_MEM["infinity_preference"]="8"
SOURCES_CPU["arena_preference_100k"]="2"; SOURCES_MEM["arena_preference_100k"]="8"

# CONVERSATION
# NOTE: smoltalk2 remains here; samvaad_hi is moved to the student-generated
#       pipeline (t2_student_curriculum.py) which handles language-literacy &
#       everyday Indic conversation.
SOURCES_CPU["smoltalk2"]="2";         SOURCES_MEM["smoltalk2"]="8"

# Ordered list for "run all"
ALL_SOURCES=(
    # Math / reasoning first (most distinct signals)
    "nemotron_math"
    "ultradata_math"
    "skywork_reward"
    "hardgen"
    "teichai"
    "gsm8k"
    # Code
    "ling_coder"
    # Science / instruction
    "megascience"
    "helpsteer3"
    "nemotron_post_training"
    # Preference / SFT mixes
    "open_perfectblend"
    "orpo_dpo_mix"
    "ultrafeedback"
    "infinity_preference"
    "arena_preference_100k"
    # Conversation (general SFT)
    "smoltalk2"
    # samvaad_hi → run via students_generated_data/run_student_jobs.sh
)

# ── Determine which sources to run ───────────────────────────────────────────
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=("${ALL_SOURCES[@]}")
fi

# ── Submit jobs ───────────────────────────────────────────────────────────────
declare -A JOB_IDS

submit_job() {
    local source="$1"
    local cpu="${SOURCES_CPU[$source]:-2}"
    local mem="${SOURCES_MEM[$source]:-8}"

    echo "──────────────────────────────────────"
    echo "Submitting: $source  (${cpu}vCPU / ${mem}GB)"

    local job_id
    job_id=$(aws emr-serverless start-job-run \
        --region "$REGION" \
        --application-id "$APP_ID" \
        --execution-role-arn "$JOB_ROLE_ARN" \
        --name "T2_Curated_${source}" \
        --job-driver "{
            \"sparkSubmit\": {
                \"entryPoint\": \"$SCRIPT_S3\",
                \"entryPointArguments\": [
                    \"--SOURCE\", \"$source\",
                    \"--INPUT_BASE\",  \"$INPUT_BASE\",
                    \"--OUTPUT_BASE\", \"$OUTPUT_BASE\"
                ],
                \"sparkSubmitParameters\": \"--conf spark.executor.cores=${cpu} --conf spark.executor.memory=${mem}g --conf spark.driver.cores=2 --conf spark.driver.memory=4g --conf spark.sql.adaptive.enabled=true --conf spark.sql.adaptive.coalescePartitions.enabled=true\"
            }
        }" \
        --configuration-overrides "{
            \"monitoringConfiguration\": {
                \"s3MonitoringConfiguration\": {
                    \"logUri\": \"$LOG_URI\"
                }
            }
        }" \
        --query "jobRunId" \
        --output text 2>&1) || true

    if [[ "$job_id" =~ ^jr_ ]]; then
        JOB_IDS["$source"]="$job_id"
        echo "  → Job ID: $job_id"
    else
        echo "  ✗ Failed to submit: $job_id"
    fi
}

echo "=============================================="
echo "T2 New Datasets — Parallel Submission"
echo "App ID   : $APP_ID"
echo "Sources  : ${#TARGETS[@]}"
echo "=============================================="

for src in "${TARGETS[@]}"; do
    submit_job "$src"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "Submitted ${#JOB_IDS[@]} job(s):"
echo "=============================================="
for src in "${!JOB_IDS[@]}"; do
    echo "  $src  →  ${JOB_IDS[$src]}"
done

echo ""
echo "Monitor with:"
echo "  aws emr-serverless list-job-runs --application-id $APP_ID --region $REGION"
echo ""
echo "Check a single job:"
echo "  aws emr-serverless get-job-run --application-id $APP_ID --job-run-id <JOB_ID> --region $REGION"
echo ""
echo "Logs at: $LOG_URI"
