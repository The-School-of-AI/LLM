# Loss Spike Detection and Recovery

## Problem

During LLM pretraining, loss spikes — sudden large increases in training loss — can destabilize training, waste compute, and in severe cases lead to divergence. Previously, the training loop only detected **NaN losses** (raising a hard `RuntimeError`) but had no mechanism to detect or recover from non-NaN loss spikes.

## Solution

A two-signal detection system with automatic escalating recovery, integrated directly into the training loop.

### Detection Signals

**1. Loss spike detection** (after forward pass, before backward)

A sliding window of the last `N` loss values (default: 100) tracks running statistics. A spike is flagged when **all** of these conditions are met:

- The window is fully populated (warmup period complete).
- The current loss exceeds `mean + z_threshold * std` **or** exceeds `min_spike_ratio * mean`.
- The absolute increase `(current_loss - mean)` exceeds `min_abs_delta`.

The z-score threshold adapts to the current training phase — early training has higher loss and variance, so the threshold naturally adjusts. The ratio guard handles edge cases where variance is near-zero during stable training. The minimum delta prevents nuisance alerts on tiny fluctuations.

Spike values are excluded from the window to prevent a single spike from corrupting running statistics. The cooldown mechanism (described below) prevents cascading re-detection.

Before feeding the loss to the detector, it is all-reduced across ranks (`dist.all_reduce` with `ReduceOp.AVG`) so that every rank makes the same spike decision and enters collective operations simultaneously.

**2. Gradient norm monitoring** (after backward, before optimizer step)

The pre-clip gradient L2 norm is compared against a hard threshold (default: 100.0). When exceeded, the same recovery flow is triggered with reason `"GRADIENT NORM SPIKE"`.

The grad norm is obtained from DeepSpeed's `get_global_grad_norm()` when available (correct under ZeRO Stage 3 where gradients are sharded across ranks), falling back to a local `clip_grad_norm_` computation otherwise.

### Recovery Actions

When a spike is detected, one of four actions is taken:

| Action | What happens |
|---|---|
| **SKIP_BATCH** | Discard the batch — skip backward/optimizer step, continue training from the same model state. |
| **REDUCE_LR** | Multiply the learning rate by `lr_reduction_factor` (default: 0.5x), then skip the batch. |
| **ROLLBACK_CHECKPOINT** | Reload model, optimizer, and training state from the last saved checkpoint, then skip `rollback_skip_batches` (default: 200) data batches to move past the problematic data-state region. The spike detector window is also reset since model weights have changed. |
| **IGNORE** | Proceed with backward and optimizer step as normal. |

### Automatic Escalation Policy (default)

In production (`auto_recover=True`, the default), an escalating policy selects the action based on how many consecutive spikes have occurred:

```
spike_count <= patience_skip (3)   ->  SKIP_BATCH
spike_count <= patience_lr  (10)   ->  REDUCE_LR
spike_count >  patience_lr         ->  ROLLBACK_CHECKPOINT
```

If no checkpoint is available for rollback, falls back to `SKIP_BATCH`.

### Cooldown

After any spike action, detection is suppressed for `cooldown_steps` (default: 50) steps. This prevents cascading alerts in noisy regions and gives the model time to recover. Normal loss values still enter the window during cooldown. If cooldown expires without another spike, the consecutive spike counter resets to zero.

### Interactive Mode (opt-in)

Setting `auto_recover=False` enables an interactive mode where rank-0 prompts the user via stdin to choose an action. This mode is only supported for single-GPU runs — multi-GPU would cause a collective deadlock since rank-0 blocks on stdin while other ranks continue.

### Embedding Norm Tracking

Embedding weight norms are logged every `emb_norm_log_interval` steps (default: 50) to detect embedding divergence. Tracked parameters:

- **Kronecker path**: `pf_to_model` projection weight norm, `embed_norm` (RMSNorm) scale parameter norms
- **Standard path**: `token_embed` weight norm
- **Both paths**: `lm_head` (output embedding) weight norm

## Configuration

All settings live under `training.loss_spike` in the config YAML:

```yaml
training:
  loss_spike:
    enabled: true
    window_size: 100
    z_threshold: 3.0
    min_spike_ratio: 2.0
    min_abs_delta: 0.5
    grad_norm_threshold: 100.0
    lr_reduction_factor: 0.5
    emb_norm_log_interval: 50
    auto_recover: true
    patience_skip: 3
    patience_lr: 10
    cooldown_steps: 50
    rollback_skip_batches: 200
    user_prompt_timeout: 300
```

Set `enabled: false` to disable loss spike detection entirely. Set `grad_norm_threshold: null` to disable gradient norm monitoring independently.

## Files

### New file: `llm/src/llm/loss_spike_recovery.py`

Contains all spike detection and recovery logic:

- `RecoveryAction` — IntEnum with `SKIP_BATCH`, `REDUCE_LR`, `ROLLBACK_CHECKPOINT`, `IGNORE`.
- `LossSpikeDetector` — Sliding window spike detector with cooldown and consecutive spike counter.
- `compute_grad_norm(model)` — Computes total L2 gradient norm using fused C++ via `clip_grad_norm_`.
- `compute_embedding_norms(model)` — Computes weight norms for embedding-related parameters.
- `auto_select_action(...)` — Escalating policy: spike count to recovery action.
- `prompt_user_for_action(...)` — Interactive stdin prompt (opt-in fallback).
- `broadcast_action(action, src)` — Broadcasts chosen action across ranks via `torch.distributed`.

### New file: `llm/tests/test_loss_spike_recovery.py`

40 tests covering the detector, cooldown, escalation policy, user input parsing, config defaults, factory, grad norm computation, and embedding norm computation.

## Changes to Existing Files

### `llm/src/llm/config.py`

Added `LossSpikeConfig` dataclass (lines 44-99) with 14 configuration fields. Added `loss_spike` field to `TrainingConfig`:

```python
@dataclass
class LossSpikeConfig:
    enabled: bool = True
    window_size: int = 100
    z_threshold: float = 3.0
    min_spike_ratio: float = 2.0
    min_abs_delta: float = 0.5
    grad_norm_threshold: float | None = 100.0
    lr_reduction_factor: float = 0.5
    emb_norm_log_interval: int = 50
    auto_recover: bool = True
    patience_skip: int = 3
    patience_lr: int = 10
    cooldown_steps: int = 50
    rollback_skip_batches: int = 200
    user_prompt_timeout: int = 300

@dataclass
class TrainingConfig:
    ...
    loss_spike: LossSpikeConfig = field(default_factory=LossSpikeConfig)
```

### `llm/src/llm/factories.py`

Added `build_loss_spike_detector` factory function (lines 175-190) and `LossSpikeConfig` import:

```python
def build_loss_spike_detector(
    cfg: LossSpikeConfig,
) -> "LossSpikeDetector | None":
    if not cfg.enabled:
        return None
    from llm.loss_spike_recovery import LossSpikeDetector
    return LossSpikeDetector(
        window_size=cfg.window_size,
        z_threshold=cfg.z_threshold,
        min_spike_ratio=cfg.min_spike_ratio,
        min_abs_delta=cfg.min_abs_delta,
        cooldown_steps=cfg.cooldown_steps,
    )
```

### `llm/src/llm/pretrainer.py`

**`__init__`** — Added spike detector initialization, spike config reference, and checkpoint tag tracking (lines 73-75):

```python
self._spike_detector = ft.build_loss_spike_detector(c.training.loss_spike)
self._spike_config = c.training.loss_spike
self._last_checkpoint_tag: str | None = None
```

**`run()`** — Converted `for epoch in range(...)` to `while epoch < max_epochs` to support rollback restarting at a prior epoch. Added `rolled_back` flag to skip epoch advancement after rollback. Two detection points added to the training loop:

1. **After forward, before backward** (lines 118-136) — Loss spike detection. All-reduces loss across ranks, feeds to detector, handles recovery action (skip/rollback/ignore).

2. **After backward, before optimizer step** (lines 141-168) — Gradient norm check. Gets global grad norm from DeepSpeed (or local fallback), compares against threshold, same recovery flow.

**Metrics logging** (lines 198, 200-204) — Added `grad_norm` as a per-step pbar metric. Embedding norms logged every `emb_norm_log_interval` steps.

**Checkpoint tag tracking** (lines 189, 232) — `_last_checkpoint_tag` updated after each checkpoint save so rollback knows which tag to reload.

**`_handle_loss_spike()`** (lines 496-592) — New method that selects a recovery action (auto or interactive), logs spike metrics, executes side-effects (LR reduction), and advances the cooldown counter.

**`_rollback_to_checkpoint()`** (lines 594-632) — New method that reloads the last checkpoint, advances `start_step` by `rollback_skip_batches` to skip past the problematic data region, and resets the spike detector window.

## References

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — describes loss spike recovery during large-scale training: restarting from a checkpoint roughly 100 steps before the spike and skipping 200-500 batches of data around the spike. This approach directly informed the `rollback_skip_batches` mechanism and the escalating recovery policy in this implementation.
- LLaMA training report — applies the same PaLM-style mitigation strategy for loss spikes encountered during LLaMA pretraining.
