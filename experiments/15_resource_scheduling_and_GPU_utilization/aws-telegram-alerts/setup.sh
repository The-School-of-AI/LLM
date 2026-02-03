#!/usr/bin/env bash
set -euo pipefail

#######################################
# AWS CloudWatch CPU Alerts to Telegram
# Idempotent - safe to run multiple times
# Only creates alarms for new instances
#######################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Default values
AWS_REGION="${AWS_REGION:-us-east-1}"
CPU_THRESHOLD="${CPU_THRESHOLD:-10}"
EVALUATION_PERIODS="${EVALUATION_PERIODS:-3}"
PERIOD_SECONDS="${PERIOD_SECONDS:-300}"
SNS_TOPIC_NAME="${SNS_TOPIC_NAME:-telegram-cpu-alerts}"
LAMBDA_FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-telegram-alert-forwarder}"
LAMBDA_ROLE_NAME="${LAMBDA_ROLE_NAME:-telegram-lambda-execution-role}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --telegram-token) TELEGRAM_BOT_TOKEN="$2"; shift 2 ;;
    --telegram-chat-id) TELEGRAM_CHAT_ID="$2"; shift 2 ;;
    --region) AWS_REGION="$2"; shift 2 ;;
    --cpu-threshold) CPU_THRESHOLD="$2"; shift 2 ;;
    --env-file) source "$2"; shift 2 ;;
    --help)
      echo "Usage: $0 --telegram-token TOKEN --telegram-chat-id CHAT_ID [options]"
      echo ""
      echo "Required:"
      echo "  --telegram-token    Telegram Bot API token"
      echo "  --telegram-chat-id  Telegram group chat ID"
      echo ""
      echo "Optional:"
      echo "  --region            AWS region (default: us-east-1)"
      echo "  --cpu-threshold     CPU % threshold (default: 10)"
      echo "  --env-file          Path to .env file"
      exit 0
      ;;
    *) log_error "Unknown parameter: $1"; exit 1 ;;
  esac
done

# Validate required parameters
[[ -z "${TELEGRAM_BOT_TOKEN:-}" ]] && { log_error "TELEGRAM_BOT_TOKEN is required"; exit 1; }
[[ -z "${TELEGRAM_CHAT_ID:-}" ]] && { log_error "TELEGRAM_CHAT_ID is required"; exit 1; }

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "Account: ${AWS_ACCOUNT_ID} | Region: ${AWS_REGION}"

#######################################
# Step 1: Create IAM Role (if not exists)
#######################################
log_info "Checking IAM role..."

if ! aws iam get-role --role-name "${LAMBDA_ROLE_NAME}" &>/dev/null; then
  log_info "Creating IAM role..."
  aws iam create-role \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' > /dev/null

  aws iam attach-role-policy \
    --role-name "${LAMBDA_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"

  log_info "Waiting for role to propagate..."
  sleep 10
fi

LAMBDA_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${LAMBDA_ROLE_NAME}"

#######################################
# Step 2: Create/Update Lambda
#######################################
log_info "Checking Lambda function..."

TEMP_DIR=$(mktemp -d)
cat > "${TEMP_DIR}/lambda_function.py" << 'PYTHON_EOF'
import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

def lambda_handler(event, context):
    bot_token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    account_id = os.environ.get('AWS_ACCOUNT_ID', 'Unknown')

    try:
        if 'Records' in event:
            message = event['Records'][0]['Sns']['Message']
            try:
                alarm = json.loads(message)
                text = format_alarm(alarm, account_id)
            except json.JSONDecodeError:
                text = "📢 *AWS Alert* ({})\n\n{}".format(account_id, message)
        else:
            text = "📢 *AWS Alert* ({})\n\n```\n{}\n```".format(account_id, json.dumps(event, indent=2))

        send_telegram(bot_token, chat_id, text)
        return {'statusCode': 200}

    except Exception as e:
        send_telegram(bot_token, chat_id, "❌ *Error* ({}): {}".format(account_id, str(e)))
        raise

def format_alarm(alarm, account_id):
    name = alarm.get('AlarmName', 'Unknown')
    state = alarm.get('NewStateValue', 'Unknown')
    reason = alarm.get('NewStateReason', 'N/A')
    region = alarm.get('Region', 'Unknown')

    ts = alarm.get('StateChangeTime', '')
    try:
        utc_time = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        ist_time = utc_time + timedelta(hours=5, minutes=30)
        time_str = ist_time.strftime('%d-%b-%Y %I:%M:%S %p IST')
    except Exception:
        time_str = ts

    if state == 'ALARM':
        emoji = '🚨'
    elif state == 'OK':
        emoji = '✅'
    else:
        emoji = '⚠️'

    trigger = alarm.get('Trigger', {})
    metric = trigger.get('MetricName', 'N/A')
    threshold = trigger.get('Threshold', 'N/A')
    dims = trigger.get('Dimensions', [])
    if dims:
        dim_str = ', '.join(["{}={}".format(d['name'], d['value']) for d in dims])
    else:
        dim_str = 'N/A'

    lines = [
        "{} *CPU Idle Alert {}*".format(emoji, dim_str),
        "",
        "*Account:* {}".format(account_id),
        "*Alarm:* {}".format(name),
        "*Status:* {}".format(state),
        "*Region:* {}".format(region),
        "",
        "*Metric:* {}".format(metric),
        "*Dimensions:* {}".format(dim_str),
        "*Threshold:* {}".format(threshold),
        "",
        "*Reason:* {}".format(reason),
        "",
        "*Time:* {}".format(time_str)
    ]
    return "\n".join(lines)

def send_telegram(bot_token, chat_id, text):
    url = "https://api.telegram.org/bot{}/sendMessage".format(bot_token)
    if len(text) > 4000:
        text = text[:4000] + "\n...(truncated)"
    data = urllib.parse.urlencode({
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'Markdown'
    }).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()
PYTHON_EOF

cd "${TEMP_DIR}" && zip -q lambda.zip lambda_function.py

if aws lambda get-function --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}" &>/dev/null; then
  log_info "Updating Lambda..."
  aws lambda update-function-code \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --zip-file "fileb://lambda.zip" \
    --region "${AWS_REGION}" > /dev/null

  sleep 5
  aws lambda update-function-configuration \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --environment "Variables={TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}}" \
    --region "${AWS_REGION}" > /dev/null
else
  log_info "Creating Lambda..."
  aws lambda create-function \
    --function-name "${LAMBDA_FUNCTION_NAME}" \
    --runtime "python3.12" \
    --role "${LAMBDA_ROLE_ARN}" \
    --handler "lambda_function.lambda_handler" \
    --zip-file "fileb://lambda.zip" \
    --timeout 30 \
    --memory-size 128 \
    --environment "Variables={TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN},TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID},AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}}" \
    --region "${AWS_REGION}" > /dev/null

  aws lambda wait function-active --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}"
fi

cd - > /dev/null && rm -rf "${TEMP_DIR}"
LAMBDA_ARN="arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${LAMBDA_FUNCTION_NAME}"

#######################################
# Step 3: Create SNS Topic (if not exists)
#######################################
log_info "Checking SNS topic..."

SNS_TOPIC_ARN=$(aws sns create-topic --name "${SNS_TOPIC_NAME}" --region "${AWS_REGION}" --query 'TopicArn' --output text)

# Add Lambda permission (ignore if exists)
aws lambda add-permission \
  --function-name "${LAMBDA_FUNCTION_NAME}" \
  --statement-id "sns-invoke" \
  --action "lambda:InvokeFunction" \
  --principal "sns.amazonaws.com" \
  --source-arn "${SNS_TOPIC_ARN}" \
  --region "${AWS_REGION}" 2>/dev/null || true

# Subscribe Lambda (idempotent)
aws sns subscribe \
  --topic-arn "${SNS_TOPIC_ARN}" \
  --protocol "lambda" \
  --notification-endpoint "${LAMBDA_ARN}" \
  --region "${AWS_REGION}" > /dev/null

#######################################
# Step 4: Create alarms for NEW instances only
#######################################
log_info "Checking for new instances..."

# Get existing alarm instance IDs
EXISTING_ALARMS=$(aws cloudwatch describe-alarms \
  --alarm-name-prefix "cpu-idle-" \
  --query "MetricAlarms[].Dimensions[?Name=='InstanceId'].Value | []" \
  --output text \
  --region "${AWS_REGION}" 2>/dev/null | tr '\t' '\n' | sort -u)

# Get running instances
RUNNING_INSTANCES=$(aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,Tags[?Key==`Name`].Value | [0]]' \
  --output text \
  --region "${AWS_REGION}")

NEW_COUNT=0
SKIP_COUNT=0

if [[ -z "${RUNNING_INSTANCES}" ]]; then
  log_warn "No running instances found"
else
  while IFS=$'\t' read -r instance_id instance_name; do
    instance_name="${instance_name:-unnamed}"

    # Check if alarm already exists
    if echo "${EXISTING_ALARMS}" | grep -q "^${instance_id}$"; then
      ((SKIP_COUNT++))
      continue
    fi

    alarm_name="cpu-idle-${instance_id}"
    log_info "Creating alarm for ${instance_id} (${instance_name})..."

    aws cloudwatch put-metric-alarm \
      --alarm-name "${alarm_name}" \
      --alarm-description "CPU below ${CPU_THRESHOLD}% for ${instance_name}" \
      --metric-name "CPUUtilization" \
      --namespace "AWS/EC2" \
      --statistic "Average" \
      --period "${PERIOD_SECONDS}" \
      --threshold "${CPU_THRESHOLD}" \
      --comparison-operator "LessThanThreshold" \
      --evaluation-periods "${EVALUATION_PERIODS}" \
      --dimensions "Name=InstanceId,Value=${instance_id}" \
      --alarm-actions "${SNS_TOPIC_ARN}" \
      --ok-actions "${SNS_TOPIC_ARN}" \
      --treat-missing-data "notBreaching" \
      --region "${AWS_REGION}"

    ((NEW_COUNT++))
  done <<< "${RUNNING_INSTANCES}"
fi

#######################################
# Summary
#######################################
echo ""
echo "=========================================="
echo -e "${GREEN}Complete${NC}"
echo "=========================================="
echo "Account:            ${AWS_ACCOUNT_ID}"
echo "Region:             ${AWS_REGION}"
echo "New alarms created: ${NEW_COUNT}"
echo "Existing (skipped): ${SKIP_COUNT}"
echo ""
