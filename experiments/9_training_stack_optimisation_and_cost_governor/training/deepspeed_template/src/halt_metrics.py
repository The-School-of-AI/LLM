"""Minimal metrics writer for halt-controller compatibility."""

import json
import time


def write_metrics(
    loss,
    path="/tmp/training_metrics.json",
    tokens_per_sec=None,
    nan=False,
    diverged=False,
    gpu_util=None,
):
    """Write current metrics to a single JSON file (overwrite). Controller reads this."""
    payload = {
        "loss": float(loss) if loss is not None else None,
        "tokens_per_sec": tokens_per_sec,
        "nan": bool(nan),
        "diverged": bool(diverged),
        "gpu_util": gpu_util,
        "heartbeat": time.time(),
    }
    with open(path, "w") as f:
        json.dump(payload, f)
