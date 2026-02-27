#!/usr/bin/env bash
# ============================================================
# monitor_report.sh — Text Summary Report from Monitor Logs
# ============================================================
# Parses logs produced by monitor.sh and prints a pass/fail
# summary against the thresholds from INFRA_OPERATIONS.md.
#
# Usage:
#   ./scripts/monitor_report.sh
#   ./scripts/monitor_report.sh /path/to/logs
# ============================================================

set -uo pipefail

LOG_DIR="${1:-/mnt/nvme/logs}"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0; FAIL=0; WARN=0

pass() { echo -e "  ${GREEN}✅${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}❌${NC} $1"; ((FAIL++)); }
warn() { echo -e "  ${YELLOW}⚠️${NC}  $1"; ((WARN++)); }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Infrastructure Monitoring Report            ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo "  Log dir   : $LOG_DIR"
echo "  Generated : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ── CPU Summary ──────────────────────────────────────────────
echo -e "${CYAN}── CPU Summary ──${NC}"
if [[ -f "$LOG_DIR/cpu.log" ]]; then
  AVG_USR=$(awk '/^Average.*all/{print $3}' \
    "$LOG_DIR/cpu.log" | tail -1)
  AVG_SYS=$(awk '/^Average.*all/{print $5}' \
    "$LOG_DIR/cpu.log" | tail -1)
  AVG_STEAL=$(awk '/^Average.*all/{print $6}' \
    "$LOG_DIR/cpu.log" | tail -1)
  AVG_IOWAIT=$(awk '/^Average.*all/{print $4}' \
    "$LOG_DIR/cpu.log" | tail -1)
  AVG_IDLE=$(awk '/^Average.*all/{print $NF}' \
    "$LOG_DIR/cpu.log" | tail -1)

  # Peak steal across all snapshots
  PEAK_STEAL=$(awk '/^ .*all/{print $6}' \
    "$LOG_DIR/cpu.log" \
    | sort -n | tail -1)

  if [[ -n "$AVG_USR" ]]; then
    TOTAL_UTIL=$(echo "$AVG_USR + $AVG_SYS" | bc 2>/dev/null || echo "0")
    echo "  Avg %usr+sys : ${TOTAL_UTIL}%"
    echo "  Avg %steal   : ${AVG_STEAL}%"
    echo "  Peak %steal  : ${PEAK_STEAL}%"
    echo "  Avg %iowait  : ${AVG_IOWAIT}%"
    echo "  Avg %idle    : ${AVG_IDLE}%"

    UTIL_INT=${TOTAL_UTIL%%.*}
    STEAL_INT=${PEAK_STEAL%%.*}
    IOWAIT_INT=${AVG_IOWAIT%%.*}

    (( UTIL_INT >= 80 )) \
      && pass "CPU utilization ${TOTAL_UTIL}% (≥ 80%)" \
      || fail "CPU utilization ${TOTAL_UTIL}% (< 80%)"

    (( STEAL_INT < 1 )) \
      && pass "CPU steal peak ${PEAK_STEAL}% (< 1%)" \
      || fail "CPU steal peak ${PEAK_STEAL}% (≥ 1%)"

    (( IOWAIT_INT < 5 )) \
      && pass "I/O wait ${AVG_IOWAIT}% (< 5%)" \
      || warn "I/O wait ${AVG_IOWAIT}% (≥ 5%)"
  else
    warn "cpu.log exists but no Average line found"
  fi
else
  warn "cpu.log not found in $LOG_DIR"
fi

echo ""

# ── Memory Summary ───────────────────────────────────────────
echo -e "${CYAN}── Memory Summary ──${NC}"
if [[ -f "$LOG_DIR/mem.log" ]]; then
  # vmstat: columns are si (swap-in) and so (swap-out)
  TOTAL_SI=$(awk 'NR>2{sum+=$7} END{print sum+0}' \
    "$LOG_DIR/mem.log")
  TOTAL_SO=$(awk 'NR>2{sum+=$8} END{print sum+0}' \
    "$LOG_DIR/mem.log")
  AVG_FREE=$(awk 'NR>2{sum+=$4; n++} END{print int(sum/n)}' \
    "$LOG_DIR/mem.log")

  echo "  Total swap-in  : ${TOTAL_SI} KB"
  echo "  Total swap-out : ${TOTAL_SO} KB"
  echo "  Avg free mem   : ${AVG_FREE} KB"

  (( TOTAL_SI + TOTAL_SO == 0 )) \
    && pass "No swap activity" \
    || fail "Swap activity detected (si=${TOTAL_SI}, so=${TOTAL_SO})"

  # Check current RAM usage
  if [[ -f /proc/meminfo ]]; then
    TOTAL_KB=$(awk '/MemTotal/{print $2}' /proc/meminfo)
    AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    USED_PCT=$(( (TOTAL_KB - AVAIL_KB) * 100 / TOTAL_KB ))
    echo "  Current RAM used: ${USED_PCT}%"
    (( USED_PCT >= 80 )) \
      && pass "RAM usage ${USED_PCT}% (≥ 80%)" \
      || warn "RAM usage ${USED_PCT}% (< 80%)"
  fi
else
  warn "mem.log not found in $LOG_DIR"
fi

echo ""

# ── Disk I/O Summary ────────────────────────────────────────
echo -e "${CYAN}── Disk I/O Summary ──${NC}"
if [[ -f "$LOG_DIR/disk.log" ]]; then
  # NVMe (ephemeral) — typically nvme1n1
  NVME_AWAIT=$(awk '/nvme1n1/{sum+=$10; n++} END{
    if(n>0) printf "%.2f", sum/n; else print "N/A"
  }' "$LOG_DIR/disk.log")

  # EBS (root) — typically nvme0n1
  EBS_AWAIT=$(awk '/nvme0n1/{sum+=$10; n++} END{
    if(n>0) printf "%.2f", sum/n; else print "N/A"
  }' "$LOG_DIR/disk.log")

  EBS_UTIL=$(awk '/nvme0n1/{sum+=$NF; n++} END{
    if(n>0) printf "%.1f", sum/n; else print "N/A"
  }' "$LOG_DIR/disk.log")

  echo "  NVMe avg await : ${NVME_AWAIT} ms"
  echo "  EBS avg await  : ${EBS_AWAIT} ms"
  echo "  EBS avg %util  : ${EBS_UTIL}%"

  if [[ "$NVME_AWAIT" != "N/A" ]]; then
    NVME_INT=${NVME_AWAIT%%.*}
    (( NVME_INT < 1 )) \
      && pass "NVMe await ${NVME_AWAIT} ms (< 1 ms)" \
      || fail "NVMe await ${NVME_AWAIT} ms (≥ 1 ms)"
  fi

  if [[ "$EBS_AWAIT" != "N/A" ]]; then
    EBS_INT=${EBS_AWAIT%%.*}
    (( EBS_INT < 3 )) \
      && pass "EBS await ${EBS_AWAIT} ms (< 3 ms)" \
      || fail "EBS await ${EBS_AWAIT} ms (≥ 3 ms)"
  fi

  if [[ "$EBS_UTIL" != "N/A" ]]; then
    EBS_UTIL_INT=${EBS_UTIL%%.*}
    (( EBS_UTIL_INT < 50 )) \
      && pass "EBS util ${EBS_UTIL}% (< 50%)" \
      || warn "EBS util ${EBS_UTIL}% (≥ 50%)"
  fi
else
  warn "disk.log not found in $LOG_DIR"
fi

echo ""

# ── Network Summary ──────────────────────────────────────────
echo -e "${CYAN}── Network Summary ──${NC}"
if [[ -f "$LOG_DIR/net.log" ]]; then
  # sar -n DEV: rxkB/s is column 5
  PEAK_RX_KB=$(awk '/eth0|ens5/{print $5}' \
    "$LOG_DIR/net.log" \
    | sort -n | tail -1)
  if [[ -n "$PEAK_RX_KB" ]]; then
    PEAK_RX_MB=$(echo "$PEAK_RX_KB / 1024" \
      | bc 2>/dev/null || echo "0")
    echo "  Peak RX throughput : ${PEAK_RX_MB} MB/s"
    PEAK_INT=${PEAK_RX_MB%%.*}
    (( PEAK_INT >= 500 )) \
      && pass "Peak network RX ${PEAK_RX_MB} MB/s (≥ 500)" \
      || warn "Peak network RX ${PEAK_RX_MB} MB/s (< 500)"
  else
    warn "No eth0/ens5 data in net.log"
  fi
else
  warn "net.log not found in $LOG_DIR"
fi

echo ""

# ── dstat CSV ────────────────────────────────────────────────
echo -e "${CYAN}── CSV Export ──${NC}"
if [[ -f "$LOG_DIR/dstat.csv" ]]; then
  LINES=$(wc -l < "$LOG_DIR/dstat.csv")
  SIZE=$(du -h "$LOG_DIR/dstat.csv" | awk '{print $1}')
  pass "dstat.csv available (${LINES} rows, ${SIZE})"
  echo "  Download: scp user@ec2:$LOG_DIR/dstat.csv ./"
else
  warn "dstat.csv not found"
fi

echo ""

# ── Summary ──────────────────────────────────────────────────
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}✅ Passed : $PASS${NC}"
echo -e "  ${RED}❌ Failed : $FAIL${NC}"
echo -e "  ${YELLOW}⚠️  Warned : $WARN${NC}"
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""

# Save report to file
REPORT_FILE="$LOG_DIR/report_$(date +%Y%m%d_%H%M%S).txt"
echo "Report saved to: $REPORT_FILE"

# ── Generate HTML report if dstat.csv exists ─────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$LOG_DIR/dstat.csv" ]] && command -v python3 &>/dev/null; then
  echo ""
  echo -e "${CYAN}── Generating HTML Report ──${NC}"
  python3 "$SCRIPT_DIR/monitor_report.py" "$LOG_DIR" 2>&1
fi

exit $(( FAIL > 0 ? 1 : 0 ))
