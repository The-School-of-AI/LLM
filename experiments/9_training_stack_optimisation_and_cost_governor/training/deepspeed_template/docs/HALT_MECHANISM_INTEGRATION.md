# Halt Mechanism Integration

This document summarizes the changes integrated into the DeepSpeed training template to support the external **halt controller**: graceful checkpoint-on-signal, metrics for the controller to read, and S3 sentinel under your prefix.

---

## Overview

When enabled, the trainer:

1. **Writes a single metrics JSON file** (overwritten periodically) that the halt controller reads to detect heartbeat stall, NaN, divergence, throughput collapse, or GPU underutilisation.
2. **Polls for a halt signal file** (`/tmp/FORCE_CHECKPOINT`). When the controller (or you) creates this file, the trainer breaks out of the loop, saves a checkpoint, writes an S3 sentinel so the controller knows the save is done, then exits.
3. **Detects NaN** in loss and writes metrics with `nan: true` before exiting so the controller can react.

The controller runs separately (e.g. in another terminal on the same machine or on a different host). It reads the metrics file, and when a trigger fires it creates `FORCE_CHECKPOINT` on the trainer instance(s), waits for the S3 sentinel, then can terminate instances.

---

## Changes Integrated

### 1. New module: `src/halt_metrics.py`

- **`write_metrics(loss, path=..., tokens_per_sec=None, nan=False, diverged=False, gpu_util=None)`**  
  Writes a single JSON file (overwrite only) with keys: `loss`, `tokens_per_sec`, `nan`, `diverged`, `gpu_util`, `heartbeat` (Unix time). No extra dependencies (e.g. no pynvml); `gpu_util` is only set if the caller passes it.

### 2. Training loop: `src/train.py`

- **Optional arguments on `train_epoch`** (all default to “off” when not set):
  - `metrics_file` — path for the metrics JSON (e.g. `/tmp/training_metrics.json`). When `None`, halt integration is disabled.
  - `metrics_interval` — seconds between metrics writes (default `30`).
  - `force_checkpoint_file` — path to poll for the halt signal (default `/tmp/FORCE_CHECKPOINT`).

- **Return value**: `train_epoch` now returns `(avg_loss, global_step, exit_reason)` where `exit_reason` is `None`, `"halt"`, or `"nan"`.

- **When `metrics_file` is set**:
  - At the start of each step: if `force_checkpoint_file` exists, break and return `exit_reason="halt"`.
  - After each step: accumulate tokens, detect non-finite loss and write metrics with `nan=True` then break with `exit_reason="nan"`.
  - Every `metrics_interval` seconds (on rank 0): write metrics with current loss and tokens/sec (overwrite the same file).

### 3. Main pipeline: `main.py`

- **Config**: Reads optional `halt` section from `config.yaml`: `metrics_file`, `metrics_interval`, `force_checkpoint_file`. If `halt` is missing or `metrics_file` is null, halt integration is off.

- **After `train_epoch`**:
  - **`exit_reason == "halt"`**: Save checkpoint with tag `"halt"`, call `wait_for_uploads()` if using S3, then `put_halt_sentinel()`, and `sys.exit(0)`.
  - **`exit_reason == "nan"`**: Optionally save checkpoint with tag `"nan"`, then `sys.exit(1)`.
  - **`exit_reason is None`**: Continue as before (evaluate, epoch checkpoint, etc.).

### 4. S3 checkpoint manager: `src/checkpoint.py`

- **`put_halt_sentinel(key=None)`**  
  Writes a small object to S3 so the controller can detect that the halt checkpoint upload is complete. When `key` is `None`, the key is derived from the template’s S3 prefix: `{s3_prefix}/latest/_SUCCESS` (e.g. `nishant/LLM/latest/_SUCCESS`). This keeps the sentinel under the same prefix as your checkpoints.

### 5. Halt controller: sentinel under prefix

- The **halt controller** (in `training/halt_mechanism/halt_controller.py`) now has:
  - **`S3_PREFIX`** — must match the template’s `s3.prefix` (e.g. `nishant/LLM`).
  - **`SENTINEL_KEY`** — derived as `{S3_PREFIX}/latest/_SUCCESS` (or `latest/_SUCCESS` at bucket root if `S3_PREFIX` is empty).
- The controller’s `wait_for_checkpoint()` uses `SENTINEL_KEY` so it waits for the same object the trainer writes.

### 6. Config YAML: `config.yaml`

- Optional **`halt`** block (commented out by default). When uncommented and `metrics_file` is set, the trainer writes metrics and polls for `FORCE_CHECKPOINT`:

```yaml
halt:
  metrics_file: "/tmp/training_metrics.json"
  metrics_interval: 30
  force_checkpoint_file: "/tmp/FORCE_CHECKPOINT"
```

---

## How the controller conveys “stop” to the trainer

The controller does **not** talk to the trainer over the network. It creates a **file on the same filesystem** as the trainer:

- **Path**: `/tmp/FORCE_CHECKPOINT` (or whatever you set in `force_checkpoint_file`).
- **Mechanism**: The trainer checks `os.path.exists(force_checkpoint_file)` at the start of each training step. When the file appears, it breaks, saves a “halt” checkpoint, writes the S3 sentinel, and exits.
- If the controller runs on the **same machine** as the trainer, it can create this file locally (e.g. `touch /tmp/FORCE_CHECKPOINT`). If it runs elsewhere, the current controller uses AWS SSM to run `touch /tmp/FORCE_CHECKPOINT` on the tagged EC2 instances.

---

## Enabling and running

1. **Template config**  
   In `config.yaml`: uncomment the `halt` section and set `metrics_file` (and optionally `metrics_interval`, `force_checkpoint_file`). Set S3 as needed (e.g. `s3.enabled: true`, `s3.bucket`, `s3.prefix`).

2. **Controller config**  
   In `halt_mechanism/halt_controller.py`: set `BUCKET` to your bucket and `S3_PREFIX` to the same value as the template’s `s3.prefix` (e.g. `nishant/LLM`) so the sentinel key matches.

3. **Run trainer**  
   Start training as usual (e.g. `deepspeed main.py`). Rank 0 will write the metrics file and poll for the halt file.

4. **Run controller**  
   On the same machine as rank 0 (so it can read the metrics file), or with metrics synced elsewhere, run `python halt_controller.py`. It will loop, read the metrics file, and when a trigger fires it will create the halt file (e.g. via SSM), wait for `s3://BUCKET/SENTINEL_KEY`, then terminate instances if configured.

---

## File and S3 layout summary

| Item | Location / key |
|------|-----------------|
| Metrics file (trainer writes, controller reads) | `/tmp/training_metrics.json` (or `halt.metrics_file`) |
| Halt signal (controller creates, trainer polls) | `/tmp/FORCE_CHECKPOINT` (or `halt.force_checkpoint_file`) |
| Halt checkpoint (trainer) | Same as other checkpoints: `s3://bucket/prefix/halt/...` |
| Sentinel (trainer writes, controller waits) | `s3://bucket/{s3_prefix}/latest/_SUCCESS` (e.g. `s3://quizizz-static-dev/nishant/LLM/latest/_SUCCESS`) |

---

## Backward compatibility

- If the `halt` section is missing or `metrics_file` is null, no metrics are written, no halt file is polled, and `train_epoch` still returns three values with `exit_reason=None`. Behaviour matches the previous template except for the extra return value.
- Callers of `train_epoch` must unpack the third value (`exit_reason`); existing call sites were updated in `main.py`.
