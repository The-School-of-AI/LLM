#!/usr/bin/env bash
# ============================================================
# validate_infra.sh — EC2 Pre-Run Infrastructure Validation
# ============================================================
# Run this script on an EC2 instance BEFORE launching the
# coreset pipeline. Default thresholds target c7gd.16xlarge
# but ALL thresholds are overridable via environment variables.
#
# Usage:
#   chmod +x validate_infra.sh
#   sudo -E ./validate_infra.sh
#
# Override thresholds (examples):
#   export EXPECTED_INSTANCE_TYPE="c5.4xlarge"
#   export MIN_VCPU=16
#   export MIN_RAM_GB=30
#   export MIN_NVME_FREE_GB=100
#   export MIN_EBS_FREE_GB=200
#   export S3_BUCKET="other-bucket"
#   sudo -E ./validate_infra.sh
#
# Exit codes:
#   0  All checks passed
#   1  One or more checks failed
# ============================================================

set -uo pipefail

# ── Colours ──────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# ── Helpers ──────────────────────────────────────────────────
pass() {
  echo -e "  ${GREEN}✅ PASS${NC}  $1"
  ((PASS_COUNT++))
}

fail() {
  echo -e "  ${RED}❌ FAIL${NC}  $1  (got: $2)"
  ((FAIL_COUNT++))
}

warn() {
  echo -e "  ${YELLOW}⚠️  WARN${NC}  $1  ($2)"
  ((WARN_COUNT++))
}

header() {
  echo ""
  echo -e "${CYAN}── $1 ──${NC}"
}

SKIP_EBS_VALIDATION="${SKIP_EBS_VALIDATION:-false}"

# ── Configurable Thresholds ──────────────────────────────────
# All thresholds can be overridden via environment variables.
# Defaults target c7gd.16xlarge.
EXPECTED_INSTANCE_TYPE="${EXPECTED_INSTANCE_TYPE:-c7gd.16xlarge}"
MIN_VCPU="${MIN_VCPU:-64}"
MIN_RAM_GB="${MIN_RAM_GB:-120}"
MAX_CPU_STEAL_PCT="${MAX_CPU_STEAL_PCT:-1}"
NVME_MOUNT="${NVME_MOUNT:-/mnt/nvme}"
MAX_NVME_LATENCY_US="${MAX_NVME_LATENCY_US:-200}"
MIN_NVME_IOPS="${MIN_NVME_IOPS:-100000}"
MIN_EBS_ROOT_GB="${MIN_EBS_ROOT_GB:-1000}"
MIN_EBS_IOPS="${MIN_EBS_IOPS:-16000}"
MAX_EBS_AWAIT_MS="${MAX_EBS_AWAIT_MS:-3}"
MIN_NVME_FREE_GB="${MIN_NVME_FREE_GB:-400}"
MIN_EBS_FREE_GB="${MIN_EBS_FREE_GB:-800}"
MIN_OPEN_FILES="${MIN_OPEN_FILES:-65536}"
MAX_SWAPPINESS="${MAX_SWAPPINESS:-1}"
MIN_PYTHON_MAJOR="${MIN_PYTHON_MAJOR:-3}"
MIN_PYTHON_MINOR="${MIN_PYTHON_MINOR:-10}"
S3_BUCKET="${S3_BUCKET:-t2-datacurriculum-353}"
S3_PREFIX="${S3_PREFIX:-processed_dataset/curriculum_pyspark_output/source=C4/bands/band=B0/}"
S3_TEST_COUNT="${S3_TEST_COUNT:-50}"
MIN_S3_SPEED_MBS="${MIN_S3_SPEED_MBS:-500}"

# ENABLE_NVME: auto-detected using two strategies:
#   1. Instance type name ('d' suffix = local NVMe, e.g. c7gd, m5d)
#   2. Actual device/mount presence (handles manually attached storage)
# Override with ENABLE_NVME=true/false to force.
if [ -z "${ENABLE_NVME+x}" ]; then
    if [[ "${EXPECTED_INSTANCE_TYPE}" =~ d\. ]]; then
        # Instance family has local NVMe (c7gd, m5d, r5d, i3en, etc.)
        ENABLE_NVME=true
    elif mountpoint -q "${NVME_MOUNT}" 2>/dev/null; then
        # Mount point exists — someone mounted storage here manually
        ENABLE_NVME=true
    elif [[ -b /dev/nvme1n1 ]] || [[ -b /dev/nvme2n1 ]]; then
        # Ephemeral NVMe device detected even on non-'d' instance
        ENABLE_NVME=true
    else
        ENABLE_NVME=false
    fi
fi

# ── IMDS v2 Token ────────────────────────────────────────────
TOKEN=$(curl -sX PUT \
  "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 120" \
  2>/dev/null || echo "")

imds() {
  curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" \
    "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null
}

# ============================================================
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   EC2 Infrastructure Validation Report       ║${NC}"
echo -e "${CYAN}║   Instance target: ${EXPECTED_INSTANCE_TYPE}             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Timestamp : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Hostname  : $(hostname)"
echo "  Thresholds: VCPU>=${MIN_VCPU} RAM>=${MIN_RAM_GB}GB NVMe=${ENABLE_NVME} EBS>=${MIN_EBS_FREE_GB}GB"
echo ""

# ============================================================
# 1. Instance Type
# ============================================================
header "1. Instance Type"
INST_TYPE=$(imds "instance-type" || echo "unknown")
if [[ "$INST_TYPE" == "${EXPECTED_INSTANCE_TYPE}" ]]; then
  pass "Instance type = $INST_TYPE"
else
  fail "Instance type" "$INST_TYPE (expected ${EXPECTED_INSTANCE_TYPE})"
fi

# ============================================================
# 2. vCPU Count
# ============================================================
header "2. CPU Count"
VCPU=$(nproc 2>/dev/null || echo 0)
if (( VCPU >= MIN_VCPU )); then
  pass "vCPU count = $VCPU"
else
  fail "vCPU count" "$VCPU (expected >= ${MIN_VCPU})"
fi

# ============================================================
# 3. NUMA Topology
# ============================================================
header "3. NUMA Topology"
if command -v numactl &>/dev/null; then
  NUMA_NODES=$(numactl --hardware 2>/dev/null \
    | grep "^available" | awk '{print $2}')
  if [[ "$NUMA_NODES" == "1" ]]; then
    pass "Single NUMA node"
  else
    warn "NUMA nodes" "$NUMA_NODES (prefer 1)"
  fi
else
  warn "numactl not installed" "install with: apt install numactl"
fi

# ============================================================
# 4. CPU Steal
# ============================================================
header "4. CPU Steal"
if command -v mpstat &>/dev/null; then
  STEAL=$(mpstat 1 1 2>/dev/null \
    | awk '/^Average.*all/{print $NF}' \
    | head -1)
  # mpstat last column is %idle; steal is column 9
  STEAL=$(mpstat 1 1 2>/dev/null \
    | awk '/^Average.*all/{print $6}' \
    | head -1)
  STEAL_INT=${STEAL%%.*}
  if (( STEAL_INT < MAX_CPU_STEAL_PCT )); then
    pass "CPU steal = ${STEAL}%"
  else
    fail "CPU steal" "${STEAL}% (expected < ${MAX_CPU_STEAL_PCT}%)"
  fi
else
  warn "mpstat not installed" "install with: apt install sysstat"
fi

# ============================================================
# 5. RAM
# ============================================================
header "5. RAM"
RAM_KB=$(awk '/^MemTotal/{print $2}' /proc/meminfo)
RAM_GB=$(( RAM_KB / 1024 / 1024 ))
if (( RAM_GB >= MIN_RAM_GB )); then
  pass "Total RAM = ${RAM_GB} GiB"
else
  fail "Total RAM" "${RAM_GB} GiB (expected >= ${MIN_RAM_GB})"
fi

# ============================================================
# 6. Swap
# ============================================================
header "6. Swap"
SWAP_KB=$(awk '/^SwapTotal/{print $2}' /proc/meminfo)
if (( SWAP_KB == 0 )); then
  pass "Swap disabled (SwapTotal = 0)"
else
  SWAP_MB=$(( SWAP_KB / 1024 ))
  fail "Swap enabled" "${SWAP_MB} MB (run: sudo swapoff -a)"
fi

# ============================================================
# 7. NVMe Device Present
# ============================================================
header "7. NVMe Discovery"
if [ "${ENABLE_NVME}" = "true" ]; then
  # Look for ephemeral NVMe (not the root EBS-backed NVMe)
  NVME_DEV=""
  for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/xvdb; do
    if [[ -b "$dev" ]]; then
      NVME_DEV="$dev"
      break
    fi
  done

  if [[ -n "$NVME_DEV" ]]; then
    pass "Ephemeral NVMe found at $NVME_DEV"
  else
    fail "NVMe device" "no ephemeral NVMe found"
  fi
else
  warn "NVMe discovery skipped" "ENABLE_NVME=false (instance ${EXPECTED_INSTANCE_TYPE} has no local NVMe)"
fi

# ============================================================
# 8. NVMe Mount
# ============================================================
header "8. NVMe Mount"
if [ "${ENABLE_NVME}" = "true" ]; then
  if mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
    NVME_FREE_GB=$(df -BG "$NVME_MOUNT" \
      | awk 'NR==2{gsub("G","",$4); print $4}')
    pass "Mounted at $NVME_MOUNT (${NVME_FREE_GB} GB free)"
  else
    fail "NVMe mount" "$NVME_MOUNT not mounted"
    echo "       Fix:  sudo mkfs.ext4 $NVME_DEV"
    echo "             sudo mkdir -p $NVME_MOUNT"
    echo "             sudo mount $NVME_DEV $NVME_MOUNT"
    echo "             sudo chown ubuntu:ubuntu $NVME_MOUNT"
  fi
else
  warn "NVMe mount skipped" "ENABLE_NVME=false"
fi

# ============================================================
# 9. NVMe Latency (fio quick test — 5 seconds)
# ============================================================
header "9. NVMe Latency Benchmark"
if [ "${ENABLE_NVME}" != "true" ]; then
  warn "NVMe benchmark skipped" "ENABLE_NVME=false"
elif command -v fio &>/dev/null && mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
  FIO_OUT=$(fio --name=val_test \
    --filename="$NVME_MOUNT/fio_validation" \
    --rw=randread --bs=4k --ioengine=libaio \
    --iodepth=32 --numjobs=1 --size=256M \
    --runtime=5 --time_based --group_reporting \
    --output-format=json 2>/dev/null)

  NVME_LAT_US=$(echo "$FIO_OUT" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(int(d['jobs'][0]['read']['clat_ns']['mean'] / 1000))
" 2>/dev/null || echo "0")

  NVME_IOPS=$(echo "$FIO_OUT" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(int(d['jobs'][0]['read']['iops']))
" 2>/dev/null || echo "0")

  rm -f "$NVME_MOUNT/fio_validation"

  if (( NVME_LAT_US < MAX_NVME_LATENCY_US )); then
    pass "NVMe avg latency = ${NVME_LAT_US} µs"
  else
    fail "NVMe latency" "${NVME_LAT_US} µs (expected < ${MAX_NVME_LATENCY_US})"
  fi

  if (( NVME_IOPS > MIN_NVME_IOPS )); then
    pass "NVMe IOPS = ${NVME_IOPS}"
  else
    warn "NVMe IOPS" "${NVME_IOPS} (expected > ${MIN_NVME_IOPS})"
  fi
else
  warn "fio benchmark skipped" \
    "install fio or mount $NVME_MOUNT first"
fi

# ============================================================
# 10. EBS Volume Size
# ============================================================
header "10. EBS Volume"
if [ "${SKIP_EBS_VALIDATION}" = "true" ]; then
  warn "EBS volume check skipped" "SKIP_EBS_VALIDATION=true"
else
  ROOT_SIZE_GB=$(df -BG / \
    | awk 'NR==2{gsub("G","",$2); print $2}')
  if (( ROOT_SIZE_GB >= MIN_EBS_ROOT_GB )); then
    pass "EBS root volume = ${ROOT_SIZE_GB} GB"
  else
    warn "EBS root volume" \
      "${ROOT_SIZE_GB} GB (recommended >= ${MIN_EBS_ROOT_GB})"
  fi
fi

# ============================================================
# 11. EBS IOPS (via AWS CLI)
# ============================================================
header "11. EBS Provisioned IOPS"
if [ "${SKIP_EBS_VALIDATION}" = "true" ]; then
  warn "EBS IOPS check skipped" "SKIP_EBS_VALIDATION=true"
elif command -v aws &>/dev/null; then
  ROOT_DEV=$(lsblk -ndo NAME / 2>/dev/null \
    | head -1 || echo "")
  # Try to get volume ID from instance metadata
  INST_ID=$(imds "instance-id" || echo "")
  if [[ -n "$INST_ID" ]]; then
    VOL_ID=$(aws ec2 describe-instances \
      --instance-ids "$INST_ID" \
      --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId' \
      --output text 2>/dev/null || echo "")
    if [[ -n "$VOL_ID" && "$VOL_ID" != "None" ]]; then
      EBS_IOPS=$(aws ec2 describe-volumes \
        --volume-ids "$VOL_ID" \
        --query 'Volumes[0].Iops' \
        --output text 2>/dev/null || echo "0")
      if (( EBS_IOPS >= MIN_EBS_IOPS )); then
        pass "EBS IOPS = ${EBS_IOPS} provisioned"
      else
        warn "EBS IOPS" \
          "${EBS_IOPS} (recommended >= ${MIN_EBS_IOPS})"
      fi
    else
      warn "EBS IOPS" "could not resolve volume ID"
    fi
  else
    warn "EBS IOPS" "instance ID not available"
  fi
else
  warn "AWS CLI not installed" \
    "install with: apt install awscli"
fi

# ============================================================
# 12. EBS Latency
# ============================================================
header "12. EBS Latency"
if [ "${SKIP_EBS_VALIDATION}" = "true" ]; then
  warn "EBS latency check skipped" "SKIP_EBS_VALIDATION=true"
elif command -v iostat &>/dev/null; then
  EBS_AWAIT=$(iostat -xd 1 1 2>/dev/null \
    | awk '/nvme0n1/{print $10}' | head -1)
  if [[ -n "$EBS_AWAIT" ]]; then
    AWAIT_INT=${EBS_AWAIT%%.*}
    if (( AWAIT_INT < MAX_EBS_AWAIT_MS )); then
      pass "EBS await = ${EBS_AWAIT} ms"
    else
      fail "EBS await" "${EBS_AWAIT} ms (expected < ${MAX_EBS_AWAIT_MS})"
    fi
  else
    warn "EBS await" "could not read iostat for nvme0n1"
  fi
else
  warn "iostat not installed" \
    "install with: apt install sysstat"
fi

# ============================================================
# 13. S3 Connectivity
# ============================================================
header "13. S3 Connectivity"

# Debug credentials under sudo
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] && [ -z "${AWS_PROFILE:-}" ] && [ -z "${AWS_CONTAINER_CREDENTIALS_RELATIVE_URI:-}" ]; then
  warn "S3 Credentials" "No AWS environment variables found (Identity might be missing under sudo)"
fi

# Ensure S3_PREFIX ends with a slash and no double slashes
S3_URL="s3://${S3_BUCKET}/${S3_PREFIX%/}/"
echo "S3 URL: $S3_URL"

if command -v aws &>/dev/null; then
  # List just 1 item to check connectivity. Capture output and status.
  S3_CHECK=$(aws s3 ls "$S3_URL" 2>&1)
  S3_STATUS=$?
  if [ $S3_STATUS -eq 0 ] && [ -n "$S3_CHECK" ]; then
    pass "S3 accessible ($S3_URL)"
  else
    S3_ERR_MSG=$(echo "$S3_CHECK" | head -n 1)
    fail "S3 access" "cannot list $S3_URL. Error: ${S3_ERR_MSG:-empty response}"
  fi
else
  warn "AWS CLI not installed" \
    "install with: apt install awscli"
fi

# ============================================================
# 14. S3 Download Speed (multiple files, parallel)
# ============================================================
header "14. S3 Download Speed"
# S3_TEST_COUNT already set in config block above
# Use NVMe for scratch if available, otherwise /tmp
if mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
  S3_TEST_DIR="$NVME_MOUNT/_s3_speed_test"
else
  S3_TEST_DIR="/tmp/_s3_speed_test"
fi

if command -v aws &>/dev/null; then

  # Discover first N files from the prefix
  mapfile -t S3_FILES < <(
    aws s3 ls "$S3_URL" --recursive 2>/dev/null \
    | awk '{print $NF}' \
    | head -n "$S3_TEST_COUNT"
  )
  FILE_COUNT=${#S3_FILES[@]}

  if (( FILE_COUNT > 0 )); then
    echo "  Downloading $FILE_COUNT files in parallel..."
    mkdir -p "$S3_TEST_DIR"

    START_T=$(date +%s%N)

    # Download all files in parallel (up to 10 concurrent)
    printf '%s\n' "${S3_FILES[@]}" \
      | xargs -P 10 -I {} \
        aws s3 cp "s3://$S3_BUCKET/{}" \
          "$S3_TEST_DIR/" --quiet 2>/dev/null

    END_T=$(date +%s%N)

    # Calculate total size and speed
    TOTAL_MB=$(du -sm "$S3_TEST_DIR" 2>/dev/null \
      | awk '{print $1}')
    ELAPSED_MS=$(( (END_T - START_T) / 1000000 ))

    if (( ELAPSED_MS > 0 && TOTAL_MB > 0 )); then
      SPEED_MBS=$(( TOTAL_MB * 1000 / ELAPSED_MS ))
      pass "Downloaded ${FILE_COUNT} files (${TOTAL_MB} MB) at ~${SPEED_MBS} MB/s"
      if (( SPEED_MBS >= MIN_S3_SPEED_MBS )); then
        pass "S3 throughput >= ${MIN_S3_SPEED_MBS} MB/s"
      else
        warn "S3 throughput" \
          "${SPEED_MBS} MB/s (expected >= ${MIN_S3_SPEED_MBS})"
      fi
    else
      warn "S3 speed" "could not measure throughput"
    fi

    # Clean up
    rm -rf "$S3_TEST_DIR"
  else
    warn "S3 speed" \
      "no files found under s3://$S3_BUCKET/$S3_PREFIX"
  fi
else
  warn "S3 speed" \
    "AWS CLI missing or NVMe not mounted"
fi

# ============================================================
# 15. Python Version
# ============================================================
header "15. Python"
if command -v python3 &>/dev/null; then
  PY_VER=$(python3 -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
  if (( PY_MAJOR >= MIN_PYTHON_MAJOR && PY_MINOR >= MIN_PYTHON_MINOR )); then
    pass "Python $PY_VER"
  else
    fail "Python version" "$PY_VER (expected >= ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR})"
  fi
else
  fail "Python" "python3 not found"
fi

# ============================================================
# 16. Python Packages
# ============================================================
header "16. Python Packages"
PKGS_OK=true
for pkg in yaml; do
  if python3 -c "import $pkg" 2>/dev/null; then
    VER=$(python3 -c \
      "import ${pkg}; print(getattr(${pkg},'__version__','ok'))")
    pass "$pkg ($VER)"
  else
    fail "$pkg" "not importable"
    PKGS_OK=false
  fi
done

# ============================================================
# 17. NVMe Free Space
# ============================================================
header "17. NVMe Free Space"
if [ "${ENABLE_NVME}" != "true" ]; then
  warn "NVMe free space skipped" "ENABLE_NVME=false"
elif mountpoint -q "$NVME_MOUNT" 2>/dev/null; then
  NVME_FREE=$(df -BG "$NVME_MOUNT" \
    | awk 'NR==2{gsub("G","",$4); print $4}')
  if (( NVME_FREE >= MIN_NVME_FREE_GB )); then
    pass "NVMe free = ${NVME_FREE} GB"
  else
    fail "NVMe free" "${NVME_FREE} GB (expected >= ${MIN_NVME_FREE_GB})"
  fi
else
  warn "NVMe free" "$NVME_MOUNT not mounted"
fi

# ============================================================
# 18. EBS Free Space
# ============================================================
header "18. EBS Free Space"
if [ "${SKIP_EBS_VALIDATION}" = "true" ]; then
  warn "EBS free space skipped" "SKIP_EBS_VALIDATION=true"
else
  EBS_FREE=$(df -BG / \
    | awk 'NR==2{gsub("G","",$4); print $4}')
  if (( EBS_FREE >= MIN_EBS_FREE_GB )); then
    pass "EBS free = ${EBS_FREE} GB"
  else
    fail "EBS free" "${EBS_FREE} GB (expected >= ${MIN_EBS_FREE_GB})"
  fi
fi

# ============================================================
# 19. Open Files Limit
# ============================================================
header "19. OS Limits – Open Files"
# Check the ubuntu user's limit (pipeline runs as ubuntu, not root)
OPEN_FILES=$(su - ubuntu -c 'ulimit -n' 2>/dev/null \
  || ulimit -n 2>/dev/null || echo 0)
if (( OPEN_FILES >= MIN_OPEN_FILES )); then
  pass "Open files limit = $OPEN_FILES (ubuntu user)"
else
  warn "Open files limit" \
    "$OPEN_FILES (add '* soft nofile ${MIN_OPEN_FILES}' to /etc/security/limits.conf and re-login)"
fi

# ============================================================
# 20. Swappiness
# ============================================================
header "20. Swappiness"
SWAPPINESS=$(cat /proc/sys/vm/swappiness 2>/dev/null \
  || echo 60)
if (( SWAPPINESS <= MAX_SWAPPINESS )); then
  pass "vm.swappiness = $SWAPPINESS"
else
  warn "vm.swappiness" \
    "$SWAPPINESS (fix: sudo sysctl -w vm.swappiness=0)"
fi

# ============================================================
# 21. Time Sync
# ============================================================
header "21. Time Sync"
if command -v timedatectl &>/dev/null; then
  NTP_SYNC=$(timedatectl show \
    --property=NTPSynchronized --value 2>/dev/null \
    || echo "no")
  if [[ "$NTP_SYNC" == "yes" ]]; then
    pass "NTP synchronized"
  else
    warn "NTP sync" "not synchronized"
  fi
else
  warn "timedatectl" "not available"
fi

# ============================================================
# 22. Monitoring Tool Dependencies
# ============================================================
header "22. Monitoring Tools"
MONITOR_TOOLS=("dstat" "htop" "nload" "iotop")
MONITOR_MISSING=()
for tool in "${MONITOR_TOOLS[@]}"; do
  if command -v "$tool" &>/dev/null; then
    pass "$tool installed"
  else
    MONITOR_MISSING+=("$tool")
  fi
done

if (( ${#MONITOR_MISSING[@]} > 0 )); then
  warn "Missing monitoring tools" \
    "${MONITOR_MISSING[*]} (fix: sudo apt-get install -y ${MONITOR_MISSING[*]})"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}✅ Passed : $PASS_COUNT${NC}"
echo -e "  ${RED}❌ Failed : $FAIL_COUNT${NC}"
echo -e "  ${YELLOW}⚠️  Warned : $WARN_COUNT${NC}"
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""

if (( FAIL_COUNT > 0 )); then
  echo -e "${RED}Action required: fix the failures above before running the pipeline.${NC}"
  exit 1
else
  echo -e "${GREEN}Infrastructure is ready. You may start the coreset pipeline.${NC}"
  exit 0
fi
