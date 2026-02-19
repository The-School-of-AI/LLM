# Halt Mechanism — Changelog

## [Unreleased] — 2026-02-17

Hardening pass for 70B LLM training. All changes are backward-compatible:
existing behaviour is preserved when the `halt:` block is absent from
`config.yaml`.

---

### halt_mechanism/halt_controller.py

#### Fixed

- **TPS baseline now uses a rolling median, not the first sample.**
  The previous code set `baseline_tps` from the very first `tokens_per_sec`
  reading. During 70B startup (JIT compilation, ZeRO-3 initialisation,
  data-loader pipeline fill) that reading is always the slowest of the run,
  making a false "throughput collapse" halt almost inevitable in the first few
  minutes. The controller now discards the first `TPS_WARMUP_SAMPLES=5`
  readings and computes a **median** over the subsequent `TPS_BASELINE_WINDOW=10`
  samples before enabling the trigger. Median is used instead of mean because
  it is not skewed by outlier checkpoint-save dips.

- **GPU idle threshold lowered from 20 % to 5 %.**
  With ZeRO-3 and CPU offloading (standard for 70B), GPUs legitimately sit at
  15–35 % during gradient reductions and optimizer steps. The previous 20 %
  threshold would fire false halts during every optimizer step on most
  ZeRO-3 configurations.

- **`wait_for_checkpoint()` now has a 60-minute timeout.**
  The previous implementation polled S3 in an infinite loop. If the trainer
  OOMed or hung during the checkpoint save itself, the sentinel would never
  be written and the controller would spin indefinitely while GPU instances
  continued to run and incur cost. The timeout is configurable via
  `CHECKPOINT_TIMEOUT` (default `3600` s). After the deadline the controller
  logs an error and proceeds to terminate anyway.

- **SSM command is now verified after sending.**
  `ssm.send_command()` returns before the remote shell has executed.
  SSM can silently drop or fail commands against degraded instances without
  raising a client-side exception. The controller now polls
  `ssm.get_command_invocation()` for each instance (up to 60 s) and logs a
  warning if the command did not reach `Success` state. This makes it visible
  when the halt signal was not delivered, rather than silently proceeding to
  wait for a sentinel that will never arrive.

- **`read_metrics()` bare `except` replaced with typed exception handling.**
  `FileNotFoundError` (normal during controller startup before the trainer
  writes its first metrics) is silenced. Any other read or parse error now
  logs a warning, making disk-full and permission errors visible instead of
  causing silent `None` returns.

- **Controller exits after a successful halt (`sys.exit(0)`).**
  Previously, after `halt_cluster()` completed the `while True` loop continued
  running, making API calls against already-terminated instances on every
  subsequent iteration.

#### Added

- **Consecutive-trigger gate (`check_trigger()`).**
  Transient events — a mid-checkpoint TPS dip, a ZeRO allreduce pause, a
  single slow step — can produce single-cycle bad readings that are not
  indicative of a real failure. All sustained triggers (heartbeat, throughput,
  GPU idle, memory pressure) now require `CONSECUTIVE_THRESHOLD=3` consecutive
  20-second polling cycles (~60 s of sustained bad signal) before a halt is
  issued. NaN and divergence are unambiguous single-step events and remain
  immediate.

- **Memory pressure trigger (`memory_pressure()`).**
  Reads `gpu_memory_pct` from the metrics file and halts if it is sustained
  above `GPU_MEMORY_MAX=95 %`. For 70B training, GPU OOM is the most common
  pre-crash failure mode. Memory pressure builds gradually and is detectable
  before the crash; the trainer cannot report OOM after the fact.

#### Constants added

| Constant | Default | Purpose |
|---|---|---|
| `GPU_MEMORY_MAX` | `95` | Halt threshold for GPU memory utilisation (%) |
| `CONSECUTIVE_THRESHOLD` | `3` | Consecutive cycles before a sustained trigger fires |
| `TPS_WARMUP_SAMPLES` | `5` | TPS samples discarded during startup |
| `TPS_BASELINE_WINDOW` | `10` | TPS samples used for median baseline |
| `CHECKPOINT_TIMEOUT` | `3600` | Max seconds to wait for S3 sentinel |

---

### deepspeed_template/src/halt_metrics.py

#### Added

- **`gpu_memory_pct` field** written on every metrics update.
  Computed as `memory_reserved() / total_memory * 100` using the current
  device's CUDA context. `memory_reserved()` (allocator headroom) is used
  rather than `memory_allocated()` (active tensors) because it more accurately
  represents OOM risk — the allocator holds pages the OS will not reclaim until
  they are explicitly freed.

- **`grad_norm` parameter** added to `write_metrics()`.
  Callers can pass the current global gradient norm. Gradient norm spikes are
  a leading indicator of training instability, typically preceding NaN loss by
  several steps. Adding the field here makes it available to the controller
  for future use without requiring a schema change.

---

### deepspeed_template/src/train.py

#### Changed

- **Gradient norm is now included in periodic metrics writes.**
  After `model_engine.step()`, `model_engine.get_global_grad_norm()` is called
  on rank 0 and passed to `write_metrics()`. Errors (e.g. ZeRO stage does not
  expose the norm) are caught and silenced; the field is omitted rather than
  crashing. This gives the halt controller a signal that precedes NaN by
  several steps.

---

### halt_mechanism/trainer_node.py

> This file is a standalone prototype used for unit-testing the halt mechanism
> without a full DeepSpeed stack. The real training pipeline is
> `deepspeed_template/main.py` + `src/train.py`.

#### Fixed

- **Divergence detection skipped during warmup.**
  Added `WARMUP_STEPS = 100`. The divergence check (`loss > mean_loss * 5`)
  is now bypassed for the first 100 steps. During 70B LR warmup, loss can
  legitimately swing by large multiples before settling; the previous code
  would have fired immediately on most realistic loss curves.

- **`tokens_accum` now derived from actual batch shape.**
  Was `+= 320` (hardcoded). Changed to `+= x.numel()` so the reported
  tokens/sec is correct for whatever batch size and sequence length the
  prototype is run with.

#### Documentation

- Added comments at both `os._exit(1)` call sites explaining that in a real
  multi-rank distributed run, calling `os._exit()` on one rank leaves all
  other ranks hanging on the next NCCL collective. The correct handling (used
  in `train.py`) is a distributed-safe break out of the training loop followed
  by `sys.exit()`.
