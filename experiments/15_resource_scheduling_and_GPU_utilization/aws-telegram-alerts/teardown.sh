#!/usr/bin/env bash
set -euo pipefail

#######################################
# Teardown all resources
#######################################

AWS_REGION="${AWS_REGION:-us-east-1}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-telegram-cpu-alerts}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-telegram-alert-forwarder}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-telegram-lambda-execution-role}"

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Account: ${AWS_ACCOUNT_ID}"
echo ""
echo "This will DELETE:"
echo "  - All cpu-idle-* CloudWatch alarms"
echo "  - SNS topic: ${SNS_TOPIC_NAME}"
echo "  - Lambda: ${LAMBDA_FUNCTION_NAME}"
echo "  - IAM role: ${LAMBDA_ROLE_NAME}"
echo ""
read -p "Continue? (y/N): " confirm
[[ "${confirm}" != "y" ]] && exit 0

# Delete alarms
echo "Deleting alarms..."
ALARMS=$(aws cloudwatch describe-alarms \
  --alarm-name-prefix "cpu-idle-" \
  --query "MetricAlarms[].AlarmName" \
  --output text \
  --region "${AWS_REGION}" 2>/dev/null || echo "")
[[ -n "${ALARMS}" ]] && aws cloudwatch delete-alarms --alarm-names ${ALARMS} --region "${AWS_REGION}"

# Delete SNS
echo "Deleting SNS topic..."
aws sns delete-topic --topic-arn "arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${SNS_TOPIC_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

# Delete Lambda
echo "Deleting Lambda..."
aws lambda delete-function --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

# Delete IAM role
echo "Deleting IAM role..."
aws iam detach-role-policy --role-name "${LAMBDA_ROLE_NAME}" \
  --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" 2>/dev/null || true
aws iam delete-role --role-name "${LAMBDA_ROLE_NAME}" 2>/dev/null || true

echo "Done."
