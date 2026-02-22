#!/bin/bash
###############################################################################
# EC2 Cost Tracker - Setup Script
# ================================
# Deploys Lambda + EventBridge schedule for real-time EC2 cost tracking.
#
# What gets created:
#   - IAM Role:        ec2-cost-tracker-lambda-role
#   - Lambda Function: ec2-cost-tracker
#   - EventBridge Rule:ec2-cost-check-every-15m
#   - S3 Bucket:       (uses existing or creates new)
#
# Prerequisites:
#   - AWS CLI configured with appropriate credentials
#   - TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID set in environment or passed as args
#   - jq installed
#
# Usage:
#   export TELEGRAM_BOT_TOKEN="your-bot-token"
#   export TELEGRAM_CHAT_ID="your-chat-id"
#   ./setup-cost-alerts.sh [--bucket BUCKET_NAME] [--credit-limit 500] [--region us-east-1]
#
# Idempotent: Safe to run repeatedly.
###############################################################################

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ─── Defaults ────────────────────────────────────────────────────────────────
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
CREDIT_LIMIT="500"
STATE_BUCKET=""
ALERT_THRESHOLDS="60,80,90,95"
INSTANCE_HOUR_ALERT="4"
SUMMARY_INTERVAL="6"
SCHEDULE_RATE="rate(15 minutes)"

# Resource naming
ROLE_NAME="ec2-cost-tracker-lambda-role"
FUNCTION_NAME="ec2-cost-tracker"
RULE_NAME="ec2-cost-check-every-15m"
LAMBDA_ZIP="ec2_cost_tracker_lambda.zip"
LAMBDA_SOURCE="ec2_cost_tracker_lambda.py"

# Tags
PROJECT_TAG="ec2-cost-alerts"
TEAM_TAG="team14-cost-monitoring"

# ─── Parse Arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --bucket)      STATE_BUCKET="$2"; shift 2 ;;
        --credit-limit) CREDIT_LIMIT="$2"; shift 2 ;;
        --region)      REGION="$2"; shift 2 ;;
        --thresholds)  ALERT_THRESHOLDS="$2"; shift 2 ;;
        --hour-alert)  INSTANCE_HOUR_ALERT="$2"; shift 2 ;;
        --summary)     SUMMARY_INTERVAL="$2"; shift 2 ;;
        --schedule)    SCHEDULE_RATE="$2"; shift 2 ;;
        *)             error "Unknown argument: $1" ;;
    esac
done

# ─── Validate Prerequisites ──────────────────────────────────────────────────
command -v aws >/dev/null 2>&1 || error "AWS CLI not found. Install: https://aws.amazon.com/cli/"
command -v jq >/dev/null 2>&1 || error "jq not found. Install: sudo apt install jq"
[[ -f "$LAMBDA_SOURCE" ]] || error "$LAMBDA_SOURCE not found in current directory."

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    error "TELEGRAM_BOT_TOKEN not set. Export it or set in environment."
fi
if [[ -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    error "TELEGRAM_CHAT_ID not set. Export it or set in environment."
fi

# Get account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) \
    || error "Failed to get AWS account ID. Check credentials."

info "Account: ${ACCOUNT_ID}"
info "Region:  ${REGION}"
info "Credit Limit: \$${CREDIT_LIMIT}"

# ─── Step 1: S3 Bucket for State ────────────────────────────────────────────
if [[ -z "$STATE_BUCKET" ]]; then
    STATE_BUCKET="ec2-cost-state-${ACCOUNT_ID}-${REGION}"
fi

info "State bucket: ${STATE_BUCKET}"

if aws s3api head-bucket --bucket "$STATE_BUCKET" 2>/dev/null; then
    success "S3 bucket already exists: ${STATE_BUCKET}"
else
    info "Creating S3 bucket: ${STATE_BUCKET}"
    if [[ "$REGION" == "us-east-1" ]]; then
        aws s3api create-bucket \
            --bucket "$STATE_BUCKET" \
            --region "$REGION"
    else
        aws s3api create-bucket \
            --bucket "$STATE_BUCKET" \
            --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi

    # Enable versioning for safety
    aws s3api put-bucket-versioning \
        --bucket "$STATE_BUCKET" \
        --versioning-configuration Status=Enabled

    # Add tags
    aws s3api put-bucket-tagging \
        --bucket "$STATE_BUCKET" \
        --tagging "TagSet=[{Key=Project,Value=${PROJECT_TAG}},{Key=Team,Value=${TEAM_TAG}}]"

    success "S3 bucket created: ${STATE_BUCKET}"
fi

# ─── Step 2: IAM Role ────────────────────────────────────────────────────────
ROLE_ARN=""

if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
    ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)
    success "IAM role exists: ${ROLE_NAME}"
else
    info "Creating IAM role: ${ROLE_NAME}"

    # Trust policy for Lambda
    TRUST_POLICY=$(cat <<'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
)

    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --tags Key=Project,Value="$PROJECT_TAG" Key=Team,Value="$TEAM_TAG" \
        --query 'Role.Arn' --output text)

    success "IAM role created: ${ROLE_ARN}"
fi

# Attach/update inline policy
info "Attaching permissions policy..."

POLICY_DOC=$(cat <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "CloudWatchLogs",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:${REGION}:${ACCOUNT_ID}:*"
        },
        {
            "Sid": "EC2Describe",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus"
            ],
            "Resource": "*"
        },
        {
            "Sid": "STSIdentity",
            "Effect": "Allow",
            "Action": "sts:GetCallerIdentity",
            "Resource": "*"
        },
        {
            "Sid": "CostExplorer",
            "Effect": "Allow",
            "Action": "ce:GetCostAndUsage",
            "Resource": "*"
        },
        {
            "Sid": "S3State",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${STATE_BUCKET}",
                "arn:aws:s3:::${STATE_BUCKET}/*"
            ]
        }
    ]
}
EOF
)

aws iam put-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-name "ec2-cost-tracker-policy" \
    --policy-document "$POLICY_DOC"

success "Policy attached to role"

# Wait for IAM propagation
info "Waiting 10s for IAM role propagation..."
sleep 10

# ─── Step 3: Lambda Function ────────────────────────────────────────────────
info "Packaging Lambda function..."

# Create zip
cd "$(dirname "$LAMBDA_SOURCE")"
zip -j "$LAMBDA_ZIP" "$LAMBDA_SOURCE"
cd - >/dev/null

LAMBDA_EXISTS=$(aws lambda get-function --function-name "$FUNCTION_NAME" 2>/dev/null && echo "yes" || echo "no")

ENV_VARS=$(cat <<EOF
{
    "Variables": {
        "TELEGRAM_BOT_TOKEN": "${TELEGRAM_BOT_TOKEN}",
        "TELEGRAM_CHAT_ID": "${TELEGRAM_CHAT_ID}",
        "STATE_BUCKET": "${STATE_BUCKET}",
        "STATE_PREFIX": "ec2-cost-state",
        "CREDIT_LIMIT": "${CREDIT_LIMIT}",
        "ALERT_THRESHOLDS": "${ALERT_THRESHOLDS}",
        "INSTANCE_HOUR_ALERT": "${INSTANCE_HOUR_ALERT}",
        "SUMMARY_INTERVAL": "${SUMMARY_INTERVAL}"
    }
}
EOF
)

if [[ "$LAMBDA_EXISTS" == "yes" ]]; then
    info "Updating existing Lambda function..."

    aws lambda update-function-code \
        --function-name "$FUNCTION_NAME" \
        --zip-file "fileb://$(dirname "$LAMBDA_SOURCE")/$LAMBDA_ZIP" \
        --region "$REGION" >/dev/null

    # Wait for update to complete
    aws lambda wait function-updated --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null || sleep 5

    aws lambda update-function-configuration \
        --function-name "$FUNCTION_NAME" \
        --environment "$ENV_VARS" \
        --timeout 60 \
        --memory-size 256 \
        --region "$REGION" >/dev/null

    LAMBDA_ARN=$(aws lambda get-function --function-name "$FUNCTION_NAME" \
        --query 'Configuration.FunctionArn' --output text --region "$REGION")

    success "Lambda function updated: ${FUNCTION_NAME}"
else
    info "Creating Lambda function: ${FUNCTION_NAME}"

    LAMBDA_ARN=$(aws lambda create-function \
        --function-name "$FUNCTION_NAME" \
        --runtime python3.12 \
        --handler "ec2_cost_tracker_lambda.lambda_handler" \
        --role "$ROLE_ARN" \
        --zip-file "fileb://$(dirname "$LAMBDA_SOURCE")/$LAMBDA_ZIP" \
        --timeout 60 \
        --memory-size 256 \
        --environment "$ENV_VARS" \
        --tags Project="$PROJECT_TAG",Team="$TEAM_TAG" \
        --region "$REGION" \
        --query 'FunctionArn' --output text)

    success "Lambda function created: ${LAMBDA_ARN}"
fi

# Clean up zip
rm -f "$(dirname "$LAMBDA_SOURCE")/$LAMBDA_ZIP"

# ─── Step 4: EventBridge Scheduled Rule ──────────────────────────────────────
info "Setting up EventBridge schedule: ${RULE_NAME}"

RULE_ARN=$(aws events put-rule \
    --name "$RULE_NAME" \
    --schedule-expression "$SCHEDULE_RATE" \
    --state ENABLED \
    --description "Triggers EC2 cost tracker every 15 minutes" \
    --tags Key=Project,Value="$PROJECT_TAG" Key=Team,Value="$TEAM_TAG" \
    --region "$REGION" \
    --query 'RuleArn' --output text)

success "EventBridge rule created: ${RULE_NAME}"

# Add Lambda as target
aws events put-targets \
    --rule "$RULE_NAME" \
    --targets "Id=ec2-cost-tracker-target,Arn=${LAMBDA_ARN}" \
    --region "$REGION" >/dev/null

success "Lambda target added to EventBridge rule"

# Grant EventBridge permission to invoke Lambda
aws lambda add-permission \
    --function-name "$FUNCTION_NAME" \
    --statement-id "eventbridge-cost-check" \
    --action "lambda:InvokeFunction" \
    --principal "events.amazonaws.com" \
    --source-arn "$RULE_ARN" \
    --region "$REGION" 2>/dev/null || warn "Permission already exists (idempotent)"

success "EventBridge -> Lambda permission configured"

# ─── Step 5: Test Invocation ─────────────────────────────────────────────────
info "Running test invocation..."

TEST_RESULT=$(aws lambda invoke \
    --function-name "$FUNCTION_NAME" \
    --payload '{}' \
    --region "$REGION" \
    /tmp/cost-tracker-test-output.json 2>&1)

if [[ -f /tmp/cost-tracker-test-output.json ]]; then
    OUTPUT=$(cat /tmp/cost-tracker-test-output.json)
    STATUS=$(echo "$OUTPUT" | jq -r '.statusCode // "error"' 2>/dev/null || echo "parse-error")
    if [[ "$STATUS" == "200" ]]; then
        BODY=$(echo "$OUTPUT" | jq -r '.body' 2>/dev/null | jq '.' 2>/dev/null || echo "$OUTPUT")
        success "Test passed! Response:"
        echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    else
        warn "Test invocation returned unexpected output:"
        cat /tmp/cost-tracker-test-output.json
    fi
    rm -f /tmp/cost-tracker-test-output.json
else
    warn "Could not read test output. Check CloudWatch logs for ${FUNCTION_NAME}."
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN} EC2 Cost Tracker Deployed Successfully! ${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Account:        ${ACCOUNT_ID}"
echo "  Region:         ${REGION}"
echo "  Lambda:         ${FUNCTION_NAME}"
echo "  Schedule:       Every 15 minutes"
echo "  State Bucket:   ${STATE_BUCKET}"
echo "  Credit Limit:   \$${CREDIT_LIMIT}"
echo "  Thresholds:     ${ALERT_THRESHOLDS}%"
echo ""
echo "  Resources created:"
echo "    IAM Role:        ${ROLE_NAME}"
echo "    Lambda Function: ${FUNCTION_NAME}"
echo "    EventBridge:     ${RULE_NAME}"
echo "    S3 Bucket:       ${STATE_BUCKET}"
echo ""
echo "  What happens now:"
echo "    • Every 15 min: scans running EC2 instances"
echo "    • Calculates runtime hours × instance hourly rate"
echo "    • Sends periodic summaries to Telegram every ${SUMMARY_INTERVAL}h"
echo "    • Alerts on budget thresholds: ${ALERT_THRESHOLDS}%"
echo "    • Alerts on instances running > ${INSTANCE_HOUR_ALERT}h"
echo "    • Alerts when instances start or stop"
echo ""
echo "═══════════════════════════════════════════════════════════════"
