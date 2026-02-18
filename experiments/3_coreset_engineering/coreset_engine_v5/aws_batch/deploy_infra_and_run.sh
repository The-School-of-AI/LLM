#!/usr/bin/env bash
# =============================================================================
# Coreset Engine: AWS Infrastructure & Job Deployment Script
# =============================================================================
# This script automates:
#   1. Creating IAM Roles and Policies
#   2. Creating CloudWatch Log Groups
#   3. Creating AWS Batch Compute Environment (Fargate)
#   4. Creating AWS Batch Job Queue
#   5. Building and Pushing Docker Image to ECR
#   6. Registering and Submitting the Batch Job
#
# Usage:
#   ./aws_batch/deploy_infra_and_run.sh --bucket <target-bucket> --input <s3-input-path>
# =============================================================================

set -euo pipefail

# --- Configuration (Can be overridden via ENV or Flags) ---
REGION=$(aws configure get region || echo "ap-south-1")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROJECT_NAME="coreset-engine"
IAM_ROLE_NAME="${PROJECT_NAME}-batch-role"
COMPUTE_ENV_NAME="${PROJECT_NAME}-fargate"
JOB_QUEUE_NAME="${PROJECT_NAME}-queue"
LOG_GROUP="/aws/batch/${PROJECT_NAME}"
IMAGE_NAME="${PROJECT_NAME}"
TAG="v5.3"

# Default values for processing
NUM_SHARDS=8
STAGES="1B 3B 8B 70B"
TOTAL_TOKENS="4523096944" # Default for books sample

function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --bucket   S3 Bucket name for outputs/checkpoints (required)"
    echo "  --input    S3 Input path (required, e.g. s3://bucket/data/)"
    echo "  --shards   Number of array shards (default: $NUM_SHARDS)"
    echo "  --stages   Space-separated stage list (default: \"$STAGES\")"
    echo "  --tokens   Total tokens in dataset (default: $TOTAL_TOKENS)"
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
echo "🚀 Starting Full Deployment for ${PROJECT_NAME}"
echo "📍 Location: ${ACCOUNT_ID} | ${REGION}"
echo "============================================================"

# 1. IAM Role for Batch/ECS
if ! aws iam get-role --role-name "${IAM_ROLE_NAME}" --region "${REGION}" >/dev/null 2>&1; then
    echo "Creating IAM Role: ${IAM_ROLE_NAME}..."
    TRUST_POLICY='{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": ["batch.amazonaws.com", "ecs-tasks.amazonaws.com"]},
            "Action": "sts:AssumeRole"
        }]
    }'
    aws iam create-role --role-name "${IAM_ROLE_NAME}" --assume-role-policy-document "${TRUST_POLICY}" --region "${REGION}"
    aws iam attach-role-policy --role-name "${IAM_ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
    aws iam attach-role-policy --role-name "${IAM_ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
    aws iam attach-role-policy --role-name "${IAM_ROLE_NAME}" --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    echo "✅ IAM Role created."
else
    echo "⏭️ IAM Role already exists."
fi
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${IAM_ROLE_NAME}"

# 2. CloudWatch Log Group
if ! aws logs describe-log-groups --log-group-name-prefix "${LOG_GROUP}" --region "${REGION}" --query "logGroups[?logGroupName=='${LOG_GROUP}']" --output text | grep -q "${LOG_GROUP}"; then
    echo "Creating Log Group: ${LOG_GROUP}..."
    aws logs create-log-group --log-group-name "${LOG_GROUP}" --region "${REGION}"
    echo "✅ Log Group created."
else
    echo "⏭️ Log Group already exists."
fi

# 3. ECR Repository
if ! aws ecr describe-repositories --repository-names "${IMAGE_NAME}" --region "${REGION}" >/dev/null 2>&1; then
    echo "Creating ECR repository: ${IMAGE_NAME}..."
    aws ecr create-repository --repository-name "${IMAGE_NAME}" --region "${REGION}"
fi
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${IMAGE_NAME}"

# 4. Networking Discovery (using Default VPC)
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --region "${REGION}" --query "Vpcs[0].VpcId" --output text)
SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${VPC_ID}" --region "${REGION}" --query "Subnets[*].SubnetId" --output json | jq -c '.')
SG_ID=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=${VPC_ID}" "Name=group-name,Values=default" --region "${REGION}" --query "SecurityGroups[0].GroupId" --output text)

# 5. Batch Compute Environment
if ! aws batch describe-compute-environments --compute-environments "${COMPUTE_ENV_NAME}" --region "${REGION}" --query "computeEnvironments[0]" --output text | grep -q "VALID"; then
    echo "Creating Compute Environment: ${COMPUTE_ENV_NAME}..."
    aws batch create-compute-environment \
        --compute-environment-name "${COMPUTE_ENV_NAME}" \
        --type MANAGED \
        --state ENABLED \
        --compute-resources "{
            \"type\": \"FARGATE\",
            \"maxvCpus\": 100,
            \"subnets\": ${SUBNETS},
            \"securityGroupIds\": [\"${SG_ID}\"]
        }" \
        --service-role "arn:aws:iam::${ACCOUNT_ID}:role/aws-service-role/batch.amazonaws.com/AWSServiceRoleForBatch" \
        --region "${REGION}"
    echo "⏳ Waiting for Compute Environment to be VALID..."
    aws batch wait compute-environments-stable --compute-environments "${COMPUTE_ENV_NAME}" --region "${REGION}"
    echo "✅ Compute Environment ready."
else
    echo "⏭️ Compute Environment already exists."
fi

# 6. Batch Job Queue
if ! aws batch describe-job-queues --job-queues "${JOB_QUEUE_NAME}" --region "${REGION}" --query "jobQueues[0]" --output text | grep -q "VALID"; then
    echo "Creating Job Queue: ${JOB_QUEUE_NAME}..."
    aws batch create-job-queue \
        --job-queue-name "${JOB_QUEUE_NAME}" \
        --state ENABLED \
        --priority 1 \
        --compute-environment-order "[{\"order\":1,\"computeEnvironment\":\"${COMPUTE_ENV_NAME}\"}]" \
        --region "${REGION}"
    echo "⏳ Waiting for Job Queue to be VALID..."
    sleep 5
    echo "✅ Job Queue ready."
else
    echo "⏭️ Job Queue already exists."
fi

# 7. Build and Push Image
echo "Logging into ECR..."
aws ecr get-login-password --region "${REGION}" | docker login --username AWS --password-stdin "${ECR_URI}"

echo "Building Image (ARM64)..."
docker build --network=host -t "${IMAGE_NAME}:${TAG}" .

echo "Pushing Image..."
docker tag "${IMAGE_NAME}:${TAG}" "${ECR_URI}:${TAG}"
docker push "${ECR_URI}:${TAG}"

# 8. Register Job Definition
echo "Registering Job Definition..."
TEMP_JOB_DEF="/tmp/coreset_job_def.json"
python3 -c "
import json
with open('aws_batch/job_definition.json') as f:
    d = json.load(f)
d.pop('_comment', None)
d['containerProperties']['image'] = '${ECR_URI}:${TAG}'
d['containerProperties']['jobRoleArn'] = '${ROLE_ARN}'
d['containerProperties']['executionRoleArn'] = '${ROLE_ARN}'
d['containerProperties']['logConfiguration']['options']['awslogs-region'] = '${REGION}'
print(json.dumps(d, indent=2))
" > "$TEMP_JOB_DEF"

REGISTERED=$(aws batch register-job-definition --cli-input-json file://"$TEMP_JOB_DEF" --region "${REGION}")
JOB_DEF_ARN=$(echo "$REGISTERED" | jq -r '.jobDefinitionArn')
echo "✅ Registered Revision: ${JOB_DEF_ARN}"

# 9. Submit the Job
echo "🚀 Submitting Array Job (${NUM_SHARDS} shards)..."
OVERRIDE=$(python3 -c "
import json
env = [
    {'name': 'S3_BUCKET', 'value': '${BUCKET}'},
    {'name': 'S3_INPUT_PATH', 'value': '${INPUT_PATH}'},
    {'name': 'TOTAL_TOKENS', 'value': '${TOTAL_TOKENS}'},
    {'name': 'NUM_SHARDS', 'value': '${NUM_SHARDS}'},
    {'name': 'STAGES', 'value': '${STAGES}'}
]
print(json.dumps({'environment': env}))
")

SUBMIT_RESPONSE=$(aws batch submit-job \
  --job-name "${PROJECT_NAME}-run" \
  --job-queue "${JOB_QUEUE_NAME}" \
  --job-definition "${JOB_DEF_ARN}" \
  --array-properties size="${NUM_SHARDS}" \
  --container-overrides "${OVERRIDE}" \
  --region "${REGION}")

JOB_ID=$(echo "$SUBMIT_RESPONSE" | jq -r '.jobId')

echo "============================================================"
echo "✅ SUCCESS: Comprehensive Deployment Complete"
echo "💰 Job ID: ${JOB_ID}"
echo "📊 Monitor logs: aws logs tail ${LOG_GROUP} --follow --region ${REGION}"
echo "============================================================"
