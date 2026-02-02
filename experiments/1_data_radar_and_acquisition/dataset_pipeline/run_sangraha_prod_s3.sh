#!/bin/bash
# Production run for Sangraha dataset with S3 storage

LOG_DIR="$(pwd)/logs"
LOG_FILE="$LOG_DIR/sangraha.log"
mkdir -p "$LOG_DIR"

nohup sudo uv run python main.py --dataset sangraha --scope production --format parquet --storage s3 --s3-bucket "$1" --s3-region "$2" --s3-prefix "$3" > "$LOG_FILE" 2>&1 &
PID=$!
echo "Sangraha production job started in background. PID: $PID"
echo "Log file: $LOG_FILE"
echo $PID > "$LOG_DIR/sangraha.pid"
