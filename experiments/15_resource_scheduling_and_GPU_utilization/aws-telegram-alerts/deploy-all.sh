#!/bin/bash
set -eu

#######################################
# Deploy to multiple AWS accounts
# Reads AWS profile names from accounts.txt
#######################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ACCOUNTS_FILE="${1:-${SCRIPT_DIR}/accounts.txt}"

# Check for required env vars
if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "[ERROR] TELEGRAM_BOT_TOKEN not set"
  exit 1
fi

if [ -z "${TELEGRAM_CHAT_ID:-}" ]; then
  echo "[ERROR] TELEGRAM_CHAT_ID not set"
  exit 1
fi

# Check accounts file
if [ ! -f "${ACCOUNTS_FILE}" ]; then
  echo "[ERROR] Accounts file not found: ${ACCOUNTS_FILE}"
  echo ""
  echo "Create accounts.txt with one AWS profile per line:"
  echo "  profile1"
  echo "  profile2"
  echo "  profile3"
  exit 1
fi

# Read profiles (skip comments and empty lines)
PROFILES=$(grep -v '^\s*#' "${ACCOUNTS_FILE}" | grep -v '^\s*$' || true)

if [ -z "${PROFILES}" ]; then
  echo "[ERROR] No profiles found in ${ACCOUNTS_FILE}"
  exit 1
fi

PROFILE_COUNT=$(echo "${PROFILES}" | wc -l | tr -d ' ')

echo ""
echo "=========================================="
echo "Deploying to ${PROFILE_COUNT} account(s)"
echo "=========================================="

SUCCESS=0
FAILED=0

while IFS= read -r PROFILE; do
  PROFILE=$(echo "${PROFILE}" | xargs)

  echo ""
  echo ">>> ${PROFILE}"

  # Build args array so each value is a separate word — safe for tokens with spaces
  SETUP_ARGS=(
    --telegram-token "${TELEGRAM_BOT_TOKEN}"
    --telegram-chat-id "${TELEGRAM_CHAT_ID}"
  )
  if [ -n "${AWS_REGION:-}" ]; then
    SETUP_ARGS+=(--region "${AWS_REGION}")
  fi
  if [ -n "${CPU_THRESHOLD:-}" ]; then
    SETUP_ARGS+=(--cpu-threshold "${CPU_THRESHOLD}")
  fi

  if AWS_PROFILE="${PROFILE}" "${SCRIPT_DIR}/setup.sh" "${SETUP_ARGS[@]}"; then
    SUCCESS=$((SUCCESS + 1))
  else
    echo "[FAILED] ${PROFILE}"
    FAILED=$((FAILED + 1))
  fi
done <<< "${PROFILES}"

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Success: ${SUCCESS}"
echo "Failed:  ${FAILED}"
echo ""

if [ "${FAILED}" -gt 0 ]; then
  exit 1
fi
exit 0