#!/bin/bash
# Production SPDL run script
# Usage: ./run_spdl_production.sh <CONFIG_FILE> <TOKEN_FOLDER>

set -e

if [ $# -ne 2 ]; then
  echo "Usage: $0 <CONFIG_FILE> <TOKEN_FOLDER>"
  exit 1
fi

CONFIG_FILE="$1"
TOKEN_FOLDER="$2"

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: Config file $CONFIG_FILE not found!"
  exit 2
fi
if [ ! -d "$TOKEN_FOLDER" ]; then
  echo "Error: Token folder $TOKEN_FOLDER not found!"
  exit 3
fi

# Ensure uv-based venv and dependencies
source setup_venv.sh

# Activate virtual environment
if [ ! -d ".venv" ]; then
  echo "Error: .venv directory not found. Please set up the environment."
  exit 4
fi
source .venv/bin/activate

# Set config for SPDL
export SPDL_CONFIG="$CONFIG_FILE"

# Print header
echo "=========================================="
echo "SPDL Production Run"
echo "=========================================="
echo "Date: $(date '+%d %B %Y %H:%M:%S')"
echo "Test File: test_spdl_bin_idx_dataloader.py"
echo -n "SPDL Version: "; python -c 'import spdl; print(spdl.__version__)'

# Hardware info
echo ""
echo "Hardware Configuration"
python -c '
import torch
import platform
import os
import subprocess
print("Platform:", platform.platform())
print("CPU Cores:", os.cpu_count())
try:
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    memory_gb = int(mem.stdout.strip()) // (1024**3)
    print("Memory:", memory_gb, "GB")
except Exception:
    print("Memory: Unable to determine")
print("CUDA Available:", torch.cuda.is_available())
print("Python Version:", platform.python_version())
'

# Run and time the pipeline
start_time=$(date +%s)
echo ""
echo "Running SPDL pipeline with config: $CONFIG_FILE and token folder: $TOKEN_FOLDER"
output=$(python dataloader.py --token-folder "$TOKEN_FOLDER" --batches 10 --log-level INFO)
end_time=$(date +%s)
duration=$((end_time - start_time))

# Print test results and performance
echo ""
echo "Test Results"
echo "$output"
echo "Processing Performance"
echo "Processing Time: ${duration} seconds"
# Try to extract batch/token info from output
batches=$(echo "$output" | grep -oE '([0-9]+) batches' | grep -oE '[0-9]+' | head -1)
tokens=$(echo "$output" | grep -oE '([0-9]+) tokens processed' | grep -oE '[0-9]+' | head -1)
if [ -n "$batches" ]; then
  echo "Batches Processed: $batches"
fi
if [ -n "$tokens" ] && [ "$duration" -gt 0 ]; then
  throughput=$(awk "BEGIN {printf \"%.2f\", $tokens/$duration}")
  echo "Throughput: $throughput tokens/second"
fi
echo "Device: CPU (CUDA not available)"

# Deactivate environment
deactivate
