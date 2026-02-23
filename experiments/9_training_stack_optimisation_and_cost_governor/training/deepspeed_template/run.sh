#!/bin/bash
export CUDA_HOME=/opt/pytorch/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTHONPATH="$(pwd)/src/models:$PYTHONPATH"

# Pick a random free port in 29000-29999 to avoid rendezvous collisions
MASTER_PORT=$(python3 -c "
import socket, random
for _ in range(100):
    p = random.randint(29000, 29999)
    with socket.socket() as s:
        if s.connect_ex(('localhost', p)) != 0:
            print(p); break
")
echo "Using MASTER_PORT=${MASTER_PORT}"

uv run deepspeed --num_gpus=4 --master_port=${MASTER_PORT} main.py
