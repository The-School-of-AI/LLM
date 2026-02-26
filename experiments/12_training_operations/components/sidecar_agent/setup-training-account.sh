#!/usr/bin/env bash
# =============================================================================
# setup-training-account.sh — run in the TRAINING account (Account A)
#
# Creates the EC2 instance role and instance profile for training instances,
# then attaches the policy that allows cross-account Secrets Manager access
# and CloudWatch metric writes.
#
# If the role already exists, the policy is updated in place (idempotent).
#
# Usage:
#   bash setup-training-account.sh \
#     --infra-account-id <INFRA_AWS_ACCOUNT_ID> \
#     --role-name        <TRAINING_INSTANCE_ROLE_NAME> \
#     [--region          <AWS_REGION>]
#
# Example:
#   bash setup-training-account.sh \
#     --infra-account-id 205991465724 \
#     --role-name        t12-traininginstance-239-role
# =============================================================================
set -euo pipefail

REGION="us-east-1"
INFRA_ACCOUNT_ID=""
ROLE_NAME=""

usage() {
  echo "Usage: $0 --infra-account-id <ID> --role-name <NAME> [--region <REGION>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --infra-account-id) INFRA_ACCOUNT_ID="$2"; shift 2 ;;
    --role-name)        ROLE_NAME="$2";        shift 2 ;;
    --region)           REGION="$2";           shift 2 ;;
  *) usage ;;
  esac
done

[[ -z "$INFRA_ACCOUNT_ID" || -z "$ROLE_NAME" ]] && usage

TRAINING_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
echo "Training account: ${TRAINING_ACCOUNT_ID}, region: ${REGION}"

# Create role if it doesn't exist
if aws iam get-role --role-name "$ROLE_NAME" &>/dev/null; then
  echo "Role ${ROLE_NAME} already exists — updating policy only."
else
  echo "Creating role ${ROLE_NAME}..."
  aws iam create-role \
    --role-name "$ROLE_NAME" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'
fi

# Create instance profile if it doesn't exist (checked independently — a role
# can exist without one if it was created manually or a prior run was interrupted)
if aws iam get-instance-profile --instance-profile-name "$ROLE_NAME" &>/dev/null; then
  echo "Instance profile ${ROLE_NAME} already exists."
else
  aws iam create-instance-profile --instance-profile-name "$ROLE_NAME"
  aws iam add-role-to-instance-profile \
    --instance-profile-name "$ROLE_NAME" \
    --role-name "$ROLE_NAME"
  echo "Created instance profile: ${ROLE_NAME}"
fi

# Attach SSM Session Manager managed policy for shell access (no SSH needed)
if aws iam list-attached-role-policies --role-name "$ROLE_NAME" --query 'AttachedPolicies[?PolicyName==`AmazonSSMManagedInstanceCore`].PolicyName' --output text | grep -q "AmazonSSMManagedInstanceCore"; then
  echo "SSM policy already attached."
else
  aws iam attach-role-policy \
    --role-name "$ROLE_NAME" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  echo "Attached SSM Session Manager policy."
fi

# Attach (or overwrite) the cross-account access policy
aws iam put-role-policy \
  --role-name "$ROLE_NAME" \
  --policy-name t12-cross-account-secrets \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"AssumeSecretsReaderRoleInInfraAccount\",
        \"Effect\": \"Allow\",
        \"Action\": \"sts:AssumeRole\",
        \"Resource\": \"arn:aws:iam::${INFRA_ACCOUNT_ID}:role/t12-secrets-reader\"
      },
      {
        \"Sid\": \"CloudWatchPushHealthMetrics\",
        \"Effect\": \"Allow\",
        \"Action\": \"cloudwatch:PutMetricData\",
        \"Resource\": \"*\",
        \"Condition\": {
          \"StringEquals\": {
            \"cloudwatch:namespace\": \"t12/Training\"
          }
        }
      }
    ]
  }"

echo "✓ Role ${ROLE_NAME} ready."
echo "  Attach instance profile '${ROLE_NAME}' to your training EC2 instances."
