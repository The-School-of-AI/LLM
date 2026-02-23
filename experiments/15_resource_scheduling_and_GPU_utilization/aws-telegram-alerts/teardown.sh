#!/bin/sh
set -eu

#######################################
# Teardown resources (single or all accounts)
# Usage:
#   ./teardown.sh              # Current account
#   ./teardown.sh --all        # All accounts in accounts.txt
#######################################

PREFIX="T15-IdleCPUMonitor-410"
AWS_REGION="${AWS_REGION:-us-east-1}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-${PREFIX}-Telegram-alert-topic}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-${PREFIX}-Telegram-alert-forwarder}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-${PREFIX}-Telegram-alert-lambda-execution-role}"
EVENTBRIDGE_LAMBDA_NAME="${EVENTBRIDGE_LAMBDA_NAME:-${PREFIX}-ec2-launch-alarm-creator}"
EVENTBRIDGE_RULE_NAME="${EVENTBRIDGE_RULE_NAME:-${PREFIX}-ec2-launch-cpu-alarm-rule}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACCOUNTS_FILE="${SCRIPT_DIR}/accounts.txt"

teardown_account() {
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  POLICY_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${PREFIX}"

  echo "Account: ${AWS_ACCOUNT_ID} | Region: ${AWS_REGION}"

  # Delete CloudWatch alarms
  # --output text returns tab-separated names on one line; tr converts to newlines
  # so each alarm name is read as a single token regardless of spaces in names
  ALARMS=$(aws cloudwatch describe-alarms \
    --query "MetricAlarms[?ends_with(AlarmName, '-cpu-idle')].AlarmName" \
    --output text \
    --region "${AWS_REGION}" 2>/dev/null | tr '\t' '\n' || true)
  if [ -n "${ALARMS}" ]; then
    echo "  Deleting alarms..."
    printf '%s\n' "${ALARMS}" | while IFS= read -r alarm; do
      [ -n "${alarm}" ] && \
        aws cloudwatch delete-alarms --alarm-names "${alarm}" --region "${AWS_REGION}" 2>/dev/null || true
    done
  fi

  # Delete EventBridge rule targets first
  echo "  Deleting targets for EventBridge rule...${EVENTBRIDGE_RULE_NAME}"
  aws events remove-targets \
    --rule "${EVENTBRIDGE_RULE_NAME}" \
    --ids "1" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Delete EventBridge rule
  echo "  Deleting EventBridge rule...${EVENTBRIDGE_RULE_NAME}"
  aws events delete-rule \
    --name "${EVENTBRIDGE_RULE_NAME}" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Delete EventBridge Lambda
  echo "  Deleting EventBridge Lambda...${EVENTBRIDGE_LAMBDA_NAME}"
  aws lambda delete-function \
    --function-name "${EVENTBRIDGE_LAMBDA_NAME}" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Delete SNS topic
  echo "  Deleting SNS topic...${SNS_TOPIC_NAME}"
  aws sns delete-topic \
    --topic-arn "arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${SNS_TOPIC_NAME}" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Delete Telegram forwarder Lambda
  echo "  Deleting Telegram forwarder Lambda...${LAMBDA_FUNCTION_NAME}"
  aws lambda delete-function \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Delete SSM parameter
  echo "  Deleting SSM parameter.../${PREFIX}/telegram-bot-token"
  aws ssm delete-parameter \
    --name "/${PREFIX}/telegram-bot-token" \
    --region "${AWS_REGION}" 2>/dev/null || true

  # Detach managed policies and inline policies from IAM role
  echo "  Detaching ${POLICY_ARN} from role...${LAMBDA_ROLE_NAME}"
  aws iam detach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn "${POLICY_ARN}" 2>/dev/null || true

  echo "  Detaching AWSLambdaBasicExecutionRole from role...${LAMBDA_ROLE_NAME}"
  aws iam detach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" 2>/dev/null || true

  echo "  Deleting inline SSM policy from role...${LAMBDA_ROLE_NAME}"
  aws iam delete-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-name "${PREFIX}-ssm-read" 2>/dev/null || true

  # Delete IAM role
  echo "  Deleting IAM role...${LAMBDA_ROLE_NAME}"
  aws iam delete-role \
    --role-name "${LAMBDA_ROLE_NAME}" 2>/dev/null || true

  echo "  Done"
}

# Check for --all flag
if [ "${1:-}" = "--all" ]; then
  # Multi-account mode
  if [ ! -f "${ACCOUNTS_FILE}" ]; then
    echo "[ERROR] accounts.txt not found"
    exit 1
  fi

  # Read profiles (skip comments and empty lines)
  PROFILES=$(grep -v '^\s*#' "${ACCOUNTS_FILE}" | grep -v '^\s*$' || true)

  if [ -z "${PROFILES}" ]; then
    echo "[ERROR] No profiles in accounts.txt"
    exit 1
  fi

  PROFILE_COUNT=$(echo "${PROFILES}" | wc -l | tr -d ' ')

  echo "This will DELETE resources from ${PROFILE_COUNT} account(s):"
  echo "${PROFILES}" | while read -r p; do echo "  - $p"; done
  echo ""
  echo "Resources to be deleted:"
  echo "  - CloudWatch alarms (*-cpu-idle)"
  echo "  - EventBridge rule: ${EVENTBRIDGE_RULE_NAME}"
  echo "  - Lambda: ${EVENTBRIDGE_LAMBDA_NAME}"
  echo "  - Lambda: ${LAMBDA_FUNCTION_NAME}"
  echo "  - SNS topic: ${SNS_TOPIC_NAME}"
  echo "  - IAM role: ${LAMBDA_ROLE_NAME}"
  echo ""
  printf "Continue? (y/N): "
  read -r confirm
  if [ "${confirm}" != "y" ]; then
    exit 0
  fi

  echo ""
  echo "${PROFILES}" | while read -r PROFILE; do
    PROFILE=$(echo "${PROFILE}" | xargs)
    echo ">>> ${PROFILE}"
    AWS_PROFILE="${PROFILE}" teardown_account || echo "[FAILED] ${PROFILE}"
    echo ""
  done

  echo "Teardown complete"
else
  # Single account mode
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

  echo "Account: ${AWS_ACCOUNT_ID}"
  echo ""
  echo "This will DELETE:"
  echo "  - CloudWatch alarms (*-cpu-idle)"
  echo "  - EventBridge rule: ${EVENTBRIDGE_RULE_NAME}"
  echo "  - Lambda: ${EVENTBRIDGE_LAMBDA_NAME}"
  echo "  - Lambda: ${LAMBDA_FUNCTION_NAME}"
  echo "  - SNS topic: ${SNS_TOPIC_NAME}"
  echo "  - IAM role: ${LAMBDA_ROLE_NAME}"
  echo ""
  printf "Continue? (y/N): "
  read -r confirm
  if [ "${confirm}" != "y" ]; then
    exit 0
  fi

  teardown_account
fi