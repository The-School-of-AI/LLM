
import os
import time
import torch
import boto3
import threading
import json
import math
from collections import deque

########################################
# GPU Monitoring (auto disables on CPU)
########################################

GPU_ENABLED = True
try:
    from pynvml import *
    nvmlInit()
    GPU_COUNT = nvmlDeviceGetCount()
    HANDLES = [nvmlDeviceGetHandleByIndex(i) for i in range(GPU_COUNT)]
except:
    GPU_ENABLED = False
    print("GPU monitoring disabled — running on CPU.")

########################################
# CONFIG
########################################

BUCKET = "YOUR_BUCKET_NAME"
PREFIX = "latest"

FORCE_FILE = "/tmp/FORCE_CHECKPOINT"
METRICS_FILE = "/tmp/training_metrics.json"
CKPT_FILE = "/tmp/checkpoint.pt"

METRIC_INTERVAL = 30

# Skip divergence detection for this many steps. During 70B warmup the loss
# can legitimately swing by large multiples before settling, so firing on the
# first few steps would produce false halts.
WARMUP_STEPS = 100

s3 = boto3.client("s3")

HALT = False
tokens_accum = 0
last_metric_time = time.time()

loss_window = deque(maxlen=5)
gpu_window = deque(maxlen=10)

########################################
# HALT LISTENER
########################################

def halt_listener():
    global HALT
    while True:
        if os.path.exists(FORCE_FILE):
            HALT = True
            return
        time.sleep(3)

threading.Thread(target=halt_listener, daemon=True).start()

########################################
# GPU UTIL
########################################

def avg_gpu_util():
    if not GPU_ENABLED:
        return None

    utils = [nvmlDeviceGetUtilizationRates(h).gpu for h in HANDLES]
    avg = sum(utils) / len(utils)

    gpu_window.append(avg)

    if len(gpu_window) < gpu_window.maxlen:
        return None

    return sum(gpu_window) / len(gpu_window)

########################################
# METRICS + HEARTBEAT
########################################

def write_metrics(loss, tps=None, nan=False, diverged=False, gpu=None):

    metrics = {
        "loss": loss,
        "tokens_per_sec": tps,
        "nan": nan,
        "diverged": diverged,
        "gpu_util": gpu,
        "heartbeat": time.time()
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)

########################################
# CHECKPOINT
########################################

def save_checkpoint(model, optimizer, step):

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "rng": torch.get_rng_state()
    }

    torch.save(state, CKPT_FILE)

    print("Uploading checkpoint to S3...")
    s3.upload_file(CKPT_FILE, BUCKET, f"{PREFIX}/checkpoint.pt")

    s3.put_object(
        Bucket=BUCKET,
        Key=f"{PREFIX}/_SUCCESS",
        Body=b"ok"
    )

    print("Checkpoint uploaded successfully.")

########################################
# TRAIN LOOP (example)
########################################

model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters())

step = 0

print("Training started...")

while True:

    x = torch.randn(32, 10)
    y = model(x).sum()
    loss = y.item()

    ################################
    # NaN detection
    ################################
    if not math.isfinite(loss):
        write_metrics(loss, nan=True)
        save_checkpoint(model, optimizer, step)
        # NOTE for distributed use: see divergence detection comment below.
        os._exit(1)

    ################################
    # Divergence detection
    ################################
    loss_window.append(loss)

    # Skip divergence check during warmup: loss swings are expected and large
    # during the first WARMUP_STEPS before the LR schedule stabilises.
    if step >= WARMUP_STEPS and len(loss_window) == loss_window.maxlen:
        mean_loss = sum(loss_window) / len(loss_window)

        if loss > mean_loss * 5:
            write_metrics(loss, diverged=True)
            save_checkpoint(model, optimizer, step)
            # NOTE for distributed use: os._exit() on one rank will leave all
            # other ranks hanging on the next NCCL collective. In the real
            # DeepSpeed pipeline (train.py), exit is handled via a distributed-
            # safe break + sys.exit(). This prototype is single-process only.
            os._exit(1)

    ################################
    # Train
    ################################
    y.backward()
    optimizer.step()
    optimizer.zero_grad()

    ################################
    # Metrics
    ################################
    tokens_accum += x.numel()  # batch_size * seq_len; was hardcoded 320
    now = time.time()

    if now - last_metric_time > METRIC_INTERVAL:

        tps = tokens_accum / (now - last_metric_time)
        gpu = avg_gpu_util()

        write_metrics(loss, tps=tps, gpu=gpu)

        tokens_accum = 0
        last_metric_time = now

    step += 1

    ################################
    # Graceful HALT
    ################################
    if HALT:
        print("HALT signal received. Saving checkpoint...")
        save_checkpoint(model, optimizer, step)
        os._exit(0)