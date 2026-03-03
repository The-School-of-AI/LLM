#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd

TOKENIZER_DIR="_data/tokenizer"
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Error: please place the tokenizer json files in $(pwd)/_data/tokenizer"
    exit 1
fi

uv run download_synth_shard.py -o _data/synth_local_en

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

uv run deepspeed --num_gpus=$NUM_GPUS main.py --config config.yaml 2>&1 | tee _data/train.log
