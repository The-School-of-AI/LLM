#!/usr/bin/env bash
# ============================================================
# monitor.sh — Unified Runtime Monitoring for EC2
# ============================================================
# Captures CPU, memory, disk I/O, and network metrics
# into /mnt/nvme/logs/ while the coreset pipeline runs.
#
# Usage:
#   chmod +x monitor.sh
#   nohup ./monitor.sh &
#
# After the pipeline finishes:
#   kill $(cat /mnt/nvme/logs/monitor.pid)
#   ./scripts/monitor_report.sh            # text summary
#   python3 scripts/monitor_report.py      # HTML charts
#   python3 scripts/monitor_report.py --upload s3://bucket/path
#
# Stop all monitors:
#   kill $(cat /mnt/nvme/logs/monitor.pid)
# ============================================================

set -euo pipefail

LOG_DIR="${LOG_DIR:-/mnt/nvme/logs}"
INTERVAL="${INTERVAL:-10}"

mkdir -p "$LOG_DIR"

echo "$$" > "$LOG_DIR/monitor.pid"
echo "=== Monitoring started ==="
echo "  PID       : $$"
echo "  Log dir   : $LOG_DIR"
echo "  Interval  : ${INTERVAL}s"
echo "  Timestamp : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ── CPU (mpstat) ─────────────────────────────────────────────
if command -v mpstat &>/dev/null; then
  mpstat -P ALL "$INTERVAL" \
    >> "$LOG_DIR/cpu.log" 2>&1 &
  echo "  ✅ CPU monitor     → cpu.log"
else
  echo "  ⚠️  mpstat not found (apt install sysstat)"
fi

# ── Memory (vmstat) ──────────────────────────────────────────
if command -v vmstat &>/dev/null; then
  vmstat "$INTERVAL" \
    >> "$LOG_DIR/mem.log" 2>&1 &
  echo "  ✅ Memory monitor  → mem.log"
else
  echo "  ⚠️  vmstat not found"
fi

# ── Disk I/O (iostat) ────────────────────────────────────────
if command -v iostat &>/dev/null; then
  iostat -xdmt "$INTERVAL" \
    >> "$LOG_DIR/disk.log" 2>&1 &
  echo "  ✅ Disk I/O monitor → disk.log"
else
  echo "  ⚠️  iostat not found (apt install sysstat)"
fi

# ── Network (sar) ────────────────────────────────────────────
if command -v sar &>/dev/null; then
  sar -n DEV "$INTERVAL" \
    >> "$LOG_DIR/net.log" 2>&1 &
  echo "  ✅ Network monitor → net.log"
else
  echo "  ⚠️  sar not found (apt install sysstat)"
fi

# ── Combined view (dstat) ────────────────────────────────────
if command -v dstat &>/dev/null; then
  dstat --cpu --io --disk --net --output \
    "$LOG_DIR/dstat.csv" "$INTERVAL" \
    > /dev/null 2>&1 &
  echo "  ✅ Combined monitor → dstat.csv"
else
  echo "  ⚠️  dstat not found (apt install dstat)"
fi

echo ""
echo "=== All monitors running in background ==="
echo "  View logs : tail -f $LOG_DIR/cpu.log"
echo "  Stop all  : kill \$(cat $LOG_DIR/monitor.pid)"
echo ""

# Wait for all background jobs
wait
