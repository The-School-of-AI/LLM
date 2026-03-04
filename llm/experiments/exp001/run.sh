#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd

TOKENIZER_DIR="_data/tokenizer"
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Error: please place the tokenizer json files in $(pwd)/_data/tokenizer"
    exit 1
fi

# Resolve proxy.local_path from config.yaml and verify it exists.
PROXY_PATH=$(uv run python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('proxy',{}).get('local_path',''))")
if [ -z "$PROXY_PATH" ]; then
    echo "Error: proxy.local_path is not set in config.yaml"
    exit 1
fi
if [[ "$PROXY_PATH" != /* ]]; then
    PROXY_PATH="$(pwd)/$PROXY_PATH"
fi
if [ ! -d "$PROXY_PATH" ] || [ -z "$(ls -A "$PROXY_PATH" 2>/dev/null)" ]; then
    echo "Error: proxy dataset folder missing/empty: $PROXY_PATH"
    echo "Set proxy.local_path in config.yaml to your local dataset folder."
    exit 1
fi

# Find the number of GPUs (Linux)
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM_GPUS=1
fi

# Fallback if nvidia-smi exists but returns 0
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi

exec uv run deepspeed --num_gpus=$NUM_GPUS main.py --config config.yaml "$@" 2>&1 | tee _data/train.log
