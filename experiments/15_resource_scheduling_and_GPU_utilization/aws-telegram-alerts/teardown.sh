#!/usr/bin/env bash
set -euo pipefail

#######################################
# Teardown resources (single or all accounts)
# Usage:
#   ./teardown.sh              # Current account
#   ./teardown.sh --all        # All accounts in accounts.txt
#######################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

AWS_REGION="${AWS_REGION:-us-east-1}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-telegram-cpu-alerts}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-telegram-alert-forwarder}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-telegram-lambda-execution-role}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNTS_FILE="${SCRIPT_DIR}/accounts.txt"

teardown_account() {
  local AWS_ACCOUNT_ID
  AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

  echo "Account: ${AWS_ACCOUNT_ID} | Region: ${AWS_REGION}"

  # Delete alarms
  echo "  Deleting alarms..."
  ALARMS=$(aws cloudwatch describe-alarms \
    --alarm-name-prefix "cpu-idle-" \
    --query "MetricAlarms[].AlarmName" \
    --output text \
    --region "${AWS_REGION}" 2>/dev/null || echo "")
  [[ -n "${ALARMS}" ]] && aws cloudwatch delete-alarms --alarm-names ${ALARMS} --region "${AWS_REGION}"

  # Delete SNS subscriptions first
  echo "  Deleting SNS topic..."
  aws sns delete-topic --topic-arn "arn:aws:sns:${AWS_REGION}:${AWS_ACCOUNT_ID}:${SNS_TOPIC_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

  # Delete Lambda
  echo "  Deleting Lambda..."
  aws lambda delete-function --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}" 2>/dev/null || true

  # Delete IAM role
  echo "  Deleting IAM role..."
  aws iam detach-role-policy --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole" 2>/dev/null || true
  aws iam delete-role --role-name "${LAMBDA_ROLE_NAME}" 2>/dev/null || true

  echo -e "  ${GREEN}Done${NC}"
}

# Check for --all flag
if [[ "${1:-}" == "--all" ]]; then
  # Multi-account mode
  if [[ ! -f "${ACCOUNTS_FILE}" ]]; then
    echo -e "${RED}[ERROR]${NC} accounts.txt not found"
    exit 1
  fi

  mapfile -t PROFILES < <(grep -v '^\s*#' "${ACCOUNTS_FILE}" | grep -v '^\s*$')

  if [[ ${#PROFILES[@]} -eq 0 ]]; then
    echo -e "${RED}[ERROR]${NC} No profiles in accounts.txt"
    exit 1
  fi

  echo "This will DELETE resources from ${#PROFILES[@]} account(s):"
  for p in "${PROFILES[@]}"; do echo "  - $p"; done
  echo ""
  read -p "Continue? (y/N): " confirm
  [[ "${confirm}" != "y" ]] && exit 0

  echo ""
  for PROFILE in "${PROFILES[@]}"; do
    PROFILE=$(echo "${PROFILE}" | xargs)
    echo -e "${YELLOW}>>> ${PROFILE}${NC}"
    AWS_PROFILE="${PROFILE}" teardown_account || echo -e "${RED}[FAILED]${NC} ${PROFILE}"
    echo ""
  done

  echo -e "${GREEN}Teardown complete${NC}"
else
  # Single account mode
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

  teardown_account
fi
