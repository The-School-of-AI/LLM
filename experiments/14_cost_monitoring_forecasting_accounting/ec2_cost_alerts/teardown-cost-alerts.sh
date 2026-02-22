#!/bin/bash
###############################################################################
# EC2 Cost Tracker - Teardown Script
# ====================================
# Removes all resources created by setup-cost-alerts.sh
#
# Usage:
#   ./teardown-cost-alerts.sh [--keep-bucket] [--region us-east-1]
#
# Options:
#   --keep-bucket  Don't delete the S3 state bucket (preserves cost history)
###############################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
KEEP_BUCKET=false

ROLE_NAME="ec2-cost-tracker-lambda-role"
FUNCTION_NAME="ec2-cost-tracker"
RULE_NAME="ec2-cost-check-every-15m"

while [[ $# -gt 0 ]]; do
    case $1 in
        --keep-bucket) KEEP_BUCKET=true; shift ;;
        --region)      REGION="$2"; shift 2 ;;
        *)             shift ;;
    esac
done

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
STATE_BUCKET="ec2-cost-state-${ACCOUNT_ID}-${REGION}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${YELLOW} EC2 Cost Tracker - Teardown ${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Account: ${ACCOUNT_ID}"
echo "  Region:  ${REGION}"
echo ""

# 1. Remove EventBridge targets and rule
info "Removing EventBridge rule: ${RULE_NAME}"
aws events remove-targets --rule "$RULE_NAME" --ids "ec2-cost-tracker-target" --region "$REGION" 2>/dev/null \
    && success "Removed EventBridge target" || warn "No target found"
aws events delete-rule --name "$RULE_NAME" --region "$REGION" 2>/dev/null \
    && success "Deleted EventBridge rule" || warn "Rule not found"

# 2. Delete Lambda function
info "Deleting Lambda function: ${FUNCTION_NAME}"
aws lambda delete-function --function-name "$FUNCTION_NAME" --region "$REGION" 2>/dev/null \
    && success "Deleted Lambda function" || warn "Lambda not found"

# 3. Delete IAM role
info "Deleting IAM role: ${ROLE_NAME}"
# Remove inline policies first
for policy in $(aws iam list-role-policies --role-name "$ROLE_NAME" --query 'PolicyNames[]' --output text 2>/dev/null); do
    aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$policy"
    success "Removed inline policy: ${policy}"
done
# Detach managed policies
for policy_arn in $(aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
    aws iam detach-role-policy --role-name "$ROLE_NAME" --policy-arn "$policy_arn"
    success "Detached policy: ${policy_arn}"
done
aws iam delete-role --role-name "$ROLE_NAME" 2>/dev/null \
    && success "Deleted IAM role" || warn "Role not found"

# 4. Optionally delete S3 bucket
if [[ "$KEEP_BUCKET" == "true" ]]; then
    warn "Keeping S3 bucket: ${STATE_BUCKET} (--keep-bucket flag)"
else
    info "Deleting S3 bucket: ${STATE_BUCKET}"
    if aws s3api head-bucket --bucket "$STATE_BUCKET" 2>/dev/null; then
        aws s3 rm "s3://${STATE_BUCKET}" --recursive 2>/dev/null
        # Also remove versioned objects
        aws s3api list-object-versions --bucket "$STATE_BUCKET" --output json 2>/dev/null | \
            jq -r '.Versions[]? | "\(.Key) \(.VersionId)"' 2>/dev/null | \
            while read -r key vid; do
                aws s3api delete-object --bucket "$STATE_BUCKET" --key "$key" --version-id "$vid" 2>/dev/null
            done
        aws s3api list-object-versions --bucket "$STATE_BUCKET" --output json 2>/dev/null | \
            jq -r '.DeleteMarkers[]? | "\(.Key) \(.VersionId)"' 2>/dev/null | \
            while read -r key vid; do
                aws s3api delete-object --bucket "$STATE_BUCKET" --key "$key" --version-id "$vid" 2>/dev/null
            done
        aws s3api delete-bucket --bucket "$STATE_BUCKET" --region "$REGION" 2>/dev/null \
            && success "Deleted S3 bucket" || warn "Could not delete bucket"
    else
        warn "Bucket not found: ${STATE_BUCKET}"
    fi
fi

echo ""
echo -e "${GREEN}Teardown complete.${NC}"
echo ""
