#!/bin/bash
# setup_venv.sh: Ensure uv-based .venv is created and dependencies are installed
# Usage: source setup_venv.sh

set -e

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[setup_venv.sh] Installing uv..."
    pip install uv || { echo "Failed to install uv"; exit 1; }
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    echo "[setup_venv.sh] Creating .venv using uv..."
    uv venv
fi

# Activate venv
source .venv/bin/activate

# Sync dependencies
if [ -f requirements.uv.txt ]; then
    echo "[setup_venv.sh] Syncing dependencies with uv..."
    uv pip sync requirements.uv.txt
else
    echo "[setup_venv.sh] requirements.uv.txt not found!"
    exit 1
fi

echo "[setup_venv.sh] .venv is ready and activated."
