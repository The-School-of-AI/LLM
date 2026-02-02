#!/bin/bash
# Monitor a dataset production job by PID and log file

if [ $# -ne 2 ]; then
  echo "Usage: $0 <pid_file> <log_file>"
  exit 1
fi

PID_FILE="$1"
LOG_FILE="$2"

if [ ! -f "$PID_FILE" ]; then
  echo "PID file not found: $PID_FILE"
  exit 2
fi

PID=$(cat "$PID_FILE")

echo "Monitoring process PID: $PID"
echo "Log file: $LOG_FILE"

echo "--- Log output (tail -f) ---"
tail -f "$LOG_FILE" &
TAIL_PID=$!

while kill -0 $PID 2>/dev/null; do
  sleep 5
done

echo "\nProcess $PID has finished. Stopping log tail."
kill $TAIL_PID
