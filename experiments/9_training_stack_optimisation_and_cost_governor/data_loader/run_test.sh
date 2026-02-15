#!/bin/bash

# SPDL DataLoader Test Runner Script
# This script activates the virtual environment, runs the test, and gathers system information

set -e  # Exit on any error

echo "=========================================="
echo "SPDL DataLoader Test Runner"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Check if we're in the right directory
if [ ! -f "test_spdl_dataloader.py" ]; then
    echo "Error: test_spdl_dataloader.py not found in current directory"
    echo "Please run this script from the data_loader directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: .venv directory not found"
    echo "Please create the virtual environment first: python3.11 -m venv .venv"
    exit 1
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
python -c "
import torch
import platform
import os
import subprocess

print('CUDA available:', torch.cuda.is_available())
print('Platform:', platform.platform())
print('CPU cores:', os.cpu_count())

# Get memory info (macOS specific)
try:
    mem = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
    memory_gb = int(mem.stdout.strip()) // (1024**3)
    print('Memory:', memory_gb, 'GB')
except:
    print('Memory: Unable to determine')

print('Python version:', platform.python_version())
print('PyTorch version:', torch.__version__)
"

echo ""
echo "=========================================="
echo "Running SPDL DataLoader Test"
echo "=========================================="
python test_spdl_dataloader.py

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="

# Deactivate virtual environment
deactivate