#!/usr/bin/env bash
set -euo pipefail

#######################################
# Deploy to multiple AWS accounts
# Reads AWS profile names from accounts.txt
#######################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNTS_FILE="${1:-${SCRIPT_DIR}/accounts.txt}"

# Check for required env vars
[[ -z "${TELEGRAM_BOT_TOKEN:-}" ]] && { echo -e "${RED}[ERROR]${NC} TELEGRAM_BOT_TOKEN not set"; exit 1; }
[[ -z "${TELEGRAM_CHAT_ID:-}" ]] && { echo -e "${RED}[ERROR]${NC} TELEGRAM_CHAT_ID not set"; exit 1; }

# Check accounts file
if [[ ! -f "${ACCOUNTS_FILE}" ]]; then
  echo -e "${RED}[ERROR]${NC} Accounts file not found: ${ACCOUNTS_FILE}"
  echo ""
  echo "Create accounts.txt with one AWS profile per line:"
  echo "  profile1"
  echo "  profile2"
  echo "  profile3"
  exit 1
fi

# Read profiles (skip empty lines and comments)
mapfile -t PROFILES < <(grep -v '^\s*#' "${ACCOUNTS_FILE}" | grep -v '^\s*$')

if [[ ${#PROFILES[@]} -eq 0 ]]; then
  echo -e "${RED}[ERROR]${NC} No profiles found in ${ACCOUNTS_FILE}"
  exit 1
fi

echo ""
echo "=========================================="
echo "Deploying to ${#PROFILES[@]} account(s)"
echo "=========================================="

SUCCESS=0
FAILED=0

for PROFILE in "${PROFILES[@]}"; do
  PROFILE=$(echo "${PROFILE}" | xargs)  # trim whitespace
  
  echo ""
  echo -e "${YELLOW}>>> ${PROFILE}${NC}"
  
  if AWS_PROFILE="${PROFILE}" "${SCRIPT_DIR}/setup.sh" \
    --telegram-token "${TELEGRAM_BOT_TOKEN}" \
    --telegram-chat-id "${TELEGRAM_CHAT_ID}" \
    ${AWS_REGION:+--region "${AWS_REGION}"} \
    ${CPU_THRESHOLD:+--cpu-threshold "${CPU_THRESHOLD}"}; then
    ((SUCCESS++))
  else
    echo -e "${RED}[FAILED]${NC} ${PROFILE}"
    ((FAILED++))
  fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo -e "Success: ${GREEN}${SUCCESS}${NC}"
echo -e "Failed:  ${RED}${FAILED}${NC}"
echo ""

[[ ${FAILED} -gt 0 ]] && exit 1
exit 0
