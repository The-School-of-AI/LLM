#!/usr/bin/env bash
# =============================================================================
# setup-infra-account.sh — run in the INFRA account (Account B)
#
# Creates the t12-secrets-reader IAM role that training instances assume
# to read the t12/clickhouse secret from Secrets Manager.
#
# Usage:
#   bash setup-infra-account.sh \
#     --training-account-id <TRAINING_AWS_ACCOUNT_ID> \
#     --training-role-name  <TRAINING_INSTANCE_ROLE_NAME> \
#     [--region <AWS_REGION>]
#
# Example:
#   bash setup-infra-account.sh \
#     --training-account-id 111122223333 \
#     --training-role-name  t12-traininginstance-239-role
# =============================================================================
set -euo pipefail

REGION="us-east-1"
TRAINING_ACCOUNT_ID=""
TRAINING_ROLE_NAME=""

usage() {
  echo "Usage: $0 --training-account-id <ID> --training-role-name <NAME> [--region <REGION>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --training-account-id) TRAINING_ACCOUNT_ID="$2"; shift 2 ;;
    --training-role-name)  TRAINING_ROLE_NAME="$2";  shift 2 ;;
    --region)              REGION="$2";              shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$TRAINING_ACCOUNT_ID" || -z "$TRAINING_ROLE_NAME" ]] && usage

INFRA_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
echo "Infra account: ${INFRA_ACCOUNT_ID}, region: ${REGION}"
echo "Trusting: arn:aws:iam::${TRAINING_ACCOUNT_ID}:role/${TRAINING_ROLE_NAME}"

# Create the role with a trust policy that allows the training instance role to assume it
aws iam create-role \
  --role-name t12-secrets-reader \
  --assume-role-policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [{
      \"Effect\": \"Allow\",
      \"Principal\": {
        \"AWS\": \"arn:aws:iam::${TRAINING_ACCOUNT_ID}:role/${TRAINING_ROLE_NAME}\"
      },
      \"Action\": \"sts:AssumeRole\"
    }]
  }"

# Attach an inline policy granting read access to t12/clickhouse only
aws iam put-role-policy \
  --role-name t12-secrets-reader \
  --policy-name t12-secrets-read-access \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {
        \"Sid\": \"SecretsManagerGetClickHouseCredentials\",
        \"Effect\": \"Allow\",
        \"Action\": \"secretsmanager:GetSecretValue\",
        \"Resource\": \"arn:aws:secretsmanager:${REGION}:${INFRA_ACCOUNT_ID}:secret:t12/clickhouse*\"
      },
      {
        \"Sid\": \"KMSDecryptSecretsManagerValues\",
        \"Effect\": \"Allow\",
        \"Action\": \"kms:Decrypt\",
        \"Resource\": \"*\",
        \"Condition\": {
          \"StringEquals\": {
            \"kms:ViaService\": \"secretsmanager.${REGION}.amazonaws.com\"
          }
        }
      }
    ]
  }"

echo "✓ Created: arn:aws:iam::${INFRA_ACCOUNT_ID}:role/t12-secrets-reader"
