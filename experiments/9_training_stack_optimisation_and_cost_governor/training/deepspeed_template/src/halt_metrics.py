"""Minimal metrics writer for halt-controller compatibility."""

import json
import time

import torch


def write_metrics(
    loss,
    path="/tmp/training_metrics.json",
    tokens_per_sec=None,
    nan=False,
    diverged=False,
    gpu_util=None,
    grad_norm=None,
):
    """Write current metrics to a single JSON file (overwrite). Controller reads this."""
    # Compute GPU memory utilisation from the current process's CUDA context.
    # Using memory_reserved() (what the allocator holds from the driver) rather than
    # memory_allocated() (what tensors actively use) gives a more conservative and
    # accurate picture of OOM risk for the controller's memory_pressure trigger.
    gpu_memory_pct = None
    if torch.cuda.is_available():
        try:
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            if total > 0:
                gpu_memory_pct = reserved / total * 100
        except Exception:
            pass

    payload = {
        "loss": float(loss) if loss is not None else None,
        "tokens_per_sec": tokens_per_sec,
        "nan": bool(nan),
        "diverged": bool(diverged),
        "gpu_util": gpu_util,
        "gpu_memory_pct": gpu_memory_pct,
        "grad_norm": float(grad_norm) if grad_norm is not None else None,
        "heartbeat": time.time(),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
