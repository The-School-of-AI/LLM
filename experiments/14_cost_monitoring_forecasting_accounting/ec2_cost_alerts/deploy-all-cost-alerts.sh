#!/bin/bash
###############################################################################
# EC2 Cost Tracker - Multi-Account Deployment
# =============================================
# Deploys ec2-cost-tracker Lambda to multiple AWS accounts.
#
# Usage:
#   1. Configure AWS CLI profiles for each account:
#
#   2. Set Telegram credentials (shared across all accounts):
#      export TELEGRAM_BOT_TOKEN="your-bot-token"
#      export TELEGRAM_CHAT_ID="your-chat-id"
#
#   3. Run:
#      ./deploy-all-cost-alerts.sh
#
#   Or deploy to specific profiles:
#      ./deploy-all-cost-alerts.sh account1 account2 ...
###############################################################################

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ─── Configuration ───────────────────────────────────────────────────────────

# Add all 20 account AWS CLI profile names here.
# Format: "profile_name:credit_limit"
# Default credit limit is 500 if not specified.
ALL_PROFILES=(
    "account1:500"
    "account2:500"
    # "account3:500"
    # "account4:500"
    # ... add all 20 accounts
    # Uncomment and fill in as accounts are onboarded
)

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─── Validate ────────────────────────────────────────────────────────────────

if [[ -z "${TELEGRAM_BOT_TOKEN:-}" ]]; then
    echo -e "${RED}[ERROR]${NC} TELEGRAM_BOT_TOKEN not set"
    exit 1
fi

if [[ -z "${TELEGRAM_CHAT_ID:-}" ]]; then
    echo -e "${RED}[ERROR]${NC} TELEGRAM_CHAT_ID not set"
    exit 1
fi

# ─── Determine target profiles ──────────────────────────────────────────────

TARGETS=()

if [[ $# -gt 0 ]]; then
    # Specific profiles passed as arguments
    for arg in "$@"; do
        # Find matching profile in ALL_PROFILES
        found=false
        for entry in "${ALL_PROFILES[@]}"; do
            profile="${entry%%:*}"
            if [[ "$profile" == "$arg" ]]; then
                TARGETS+=("$entry")
                found=true
                break
            fi
        done
        if [[ "$found" == "false" ]]; then
            # Use default credit limit
            TARGETS+=("${arg}:500")
        fi
    done
else
    TARGETS=("${ALL_PROFILES[@]}")
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${BLUE} EC2 Cost Tracker - Multi-Account Deployment ${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Deploying to ${#TARGETS[@]} account(s)"
echo "  Region: ${REGION}"
echo ""

# ─── Deploy to each account ─────────────────────────────────────────────────

SUCCEEDED=0
FAILED=0
FAILED_PROFILES=()

for entry in "${TARGETS[@]}"; do
    profile="${entry%%:*}"
    credit_limit="${entry##*:}"

    echo ""
    echo "───────────────────────────────────────────────────────────"
    echo -e "${BLUE}Deploying to profile: ${profile} (credit limit: \$${credit_limit})${NC}"
    echo "───────────────────────────────────────────────────────────"

    # Verify profile works
    if ! ACCOUNT_ID=$(AWS_PROFILE="$profile" aws sts get-caller-identity \
            --query Account --output text 2>/dev/null); then
        echo -e "${RED}[FAILED]${NC} Cannot authenticate with profile: ${profile}"
        FAILED=$((FAILED + 1))
        FAILED_PROFILES+=("$profile")
        continue
    fi

    echo -e "${BLUE}[INFO]${NC} Account ID: ${ACCOUNT_ID}"

    # Run setup script with this profile
    if AWS_PROFILE="$profile" \
       AWS_DEFAULT_REGION="$REGION" \
       TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN" \
       TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID" \
       bash "${SCRIPT_DIR}/setup-cost-alerts.sh" \
           --credit-limit "$credit_limit" \
           --region "$REGION"; then

        echo -e "${GREEN}[✓]${NC} Successfully deployed to ${profile} (${ACCOUNT_ID})"
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo -e "${RED}[✗]${NC} Failed to deploy to ${profile} (${ACCOUNT_ID})"
        FAILED=$((FAILED + 1))
        FAILED_PROFILES+=("$profile")
    fi
done

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${BLUE} Deployment Summary ${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo -e "  ${GREEN}Succeeded: ${SUCCEEDED}${NC}"
echo -e "  ${RED}Failed:    ${FAILED}${NC}"

if [[ ${#FAILED_PROFILES[@]} -gt 0 ]]; then
    echo ""
    echo "  Failed profiles:"
    for fp in "${FAILED_PROFILES[@]}"; do
        echo -e "    ${RED}• ${fp}${NC}"
    done
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"

exit $FAILED
