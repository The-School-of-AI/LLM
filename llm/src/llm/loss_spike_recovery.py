"""
Loss Spike Detection and Recovery.

Monitors training loss via a sliding window with z-score threshold,
and optionally monitors gradient norms against a hard threshold.

Two recovery modes are supported:

* **auto_recover=True** (default, production):  An escalating policy
  automatically selects the recovery action based on how many
  consecutive spikes have occurred:
    spike_count <= patience_skip  -> skip batch
    spike_count <= patience_lr    -> reduce LR + skip batch
    spike_count >  patience_lr    -> rollback checkpoint + skip batches

* **auto_recover=False** (interactive / debug):  The user is prompted
  via stdin to choose an action.

A configurable cooldown suppresses detection for N steps after any
spike action to prevent cascading alerts in noisy regions.
"""

import select
import sys
from collections import deque
from enum import IntEnum
from statistics import mean, stdev

import torch
import torch.distributed as dist

from llm.utils import is_main_process


class RecoveryAction(IntEnum):
    """Recovery actions available when a loss spike is detected."""

    SKIP_BATCH = 1
    REDUCE_LR = 2
    ROLLBACK_CHECKPOINT = 3
    IGNORE = 4


class LossSpikeDetector:
    """
    Detects loss spikes using a sliding window with z-score threshold.

    A spike is flagged when ALL of these conditions are met:
    1. The window has enough samples (at least ``window_size`` values).
    2. The current loss exceeds ``mean + z_threshold * std`` OR
       exceeds ``min_spike_ratio * mean``.
    3. The absolute increase ``(current - mean)`` exceeds ``min_abs_delta``.

    Additionally tracks a **cooldown** counter: after any spike action,
    detection is suppressed for ``cooldown_steps`` steps so that the
    model has time to recover before another alert fires.

    A **consecutive spike counter** is maintained for the automatic
    escalation policy.  It resets to zero when cooldown expires
    without another spike.
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        min_spike_ratio: float = 2.0,
        min_abs_delta: float = 0.5,
        cooldown_steps: int = 50,
    ):
        self._window: deque[float] = deque(maxlen=window_size)
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._min_spike_ratio = min_spike_ratio
        self._min_abs_delta = min_abs_delta
        self._last_loss: float | None = None

        # Cooldown state
        self._cooldown_steps = cooldown_steps
        self._cooldown_remaining: int = 0

        # Consecutive spike counter for escalation policy
        self._spike_count: int = 0

    # -- public API ----------------------------------------------------------

    def update(self, loss: float) -> bool:
        """
        Record a new loss value and return True if it constitutes a spike.

        During cooldown the value is still appended to the window but
        detection is suppressed (always returns False).
        """
        self._last_loss = loss

        # Not enough history yet — still in warmup.
        if len(self._window) < self._window_size:
            self._window.append(loss)
            return False

        # Cooldown active — tick down, add to window, no detection.
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                # Cooldown expired without another spike → reset counter.
                self._spike_count = 0
            self._window.append(loss)
            return False

        is_spike = self._check_spike(loss)

        # Only add non-spike values to the window so that a single spike
        # doesn't corrupt the running statistics.
        if not is_spike:
            self._window.append(loss)

        return is_spike

    def record_spike_action(self) -> None:
        """
        Called by the trainer after a spike is handled.

        Increments the consecutive spike counter and starts the cooldown
        timer.
        """
        self._spike_count += 1
        self._cooldown_remaining = self._cooldown_steps

    @property
    def spike_count(self) -> int:
        """Number of consecutive spikes since the last cooldown reset."""
        return self._spike_count

    def get_stats(self) -> dict:
        """Return current window statistics for display / logging."""
        w_mean = mean(self._window) if self._window else 0.0
        w_std = stdev(self._window) if len(self._window) > 1 else 0.0
        return {
            "current_loss": self._last_loss,
            "window_mean": w_mean,
            "window_std": w_std,
            "spike_ratio": (
                self._last_loss / w_mean if w_mean > 0 and self._last_loss else 0.0
            ),
            "spike_count": self._spike_count,
        }

    # -- internal ------------------------------------------------------------

    def _check_spike(self, loss: float) -> bool:
        """Check whether *loss* qualifies as a spike against the current window."""
        w_mean = mean(self._window)
        w_std = stdev(self._window) if len(self._window) > 1 else 0.0
        delta = loss - w_mean

        # Guard: absolute delta must exceed minimum threshold.
        if delta < self._min_abs_delta:
            return False

        # Check z-score threshold.
        if w_std > 0 and loss > w_mean + self._z_threshold * w_std:
            return True

        # Fallback: ratio-based check (handles near-zero variance).
        if w_mean > 0 and loss > self._min_spike_ratio * w_mean:
            return True

        return False


# ---------------------------------------------------------------------------
# Gradient norm utilities
# ---------------------------------------------------------------------------


def compute_grad_norm(model: torch.nn.Module) -> float:
    """
    Compute the total L2 gradient norm across all parameters.

    Uses ``torch.nn.utils.clip_grad_norm_`` with ``max_norm=inf`` so no
    clipping occurs — only the fused C++ norm computation runs.  This is
    significantly faster than a Python loop over parameters for large models.

    Call after ``engine.backward()`` but before ``engine.step()``
    (which applies its own gradient clipping).
    """
    params = [p for p in model.parameters() if p.grad is not None]
    if not params:
        return 0.0
    return torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf")).item()


def compute_embedding_norms(model: torch.nn.Module) -> dict[str, float]:
    """
    Compute weight norms for embedding-related parameters.

    Supports both Kronecker and standard embedding architectures.
    Returns a dict of ``{metric_name: norm_value}`` suitable for logging.
    """
    norms: dict[str, float] = {}

    # Kronecker path: pf_to_model projection + embed_norm (RMSNorm scale)
    pf_to_model = getattr(model, "pf_to_model", None)
    if pf_to_model is not None:
        norms["emb_proj_norm"] = torch.norm(
            pf_to_model.weight.detach().float(), 2
        ).item()

    embed_norm_layer = getattr(model, "embed_norm", None)
    if embed_norm_layer is not None:
        for name, p in embed_norm_layer.named_parameters():
            norms[f"emb_rmsnorm_{name}_norm"] = torch.norm(
                p.detach().float(), 2
            ).item()

    # Standard embedding path
    token_embed = getattr(model, "token_embed", None)
    if token_embed is not None:
        norms["token_emb_norm"] = torch.norm(
            token_embed.weight.detach().float(), 2
        ).item()

    # lm_head (output embedding) — often tied with token_embed but worth
    # tracking separately to catch divergence.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        norms["lm_head_norm"] = torch.norm(
            lm_head.weight.detach().float(), 2
        ).item()

    return norms


# ---------------------------------------------------------------------------
# Automatic escalation policy
# ---------------------------------------------------------------------------


def auto_select_action(
    spike_count: int,
    patience_skip: int,
    patience_lr: int,
    last_checkpoint_tag: str | None,
) -> RecoveryAction:
    """
    Automatically choose a recovery action based on the consecutive spike count.

    Escalation tiers:
        spike_count <= patience_skip  -> SKIP_BATCH
        spike_count <= patience_lr    -> REDUCE_LR
        spike_count >  patience_lr    -> ROLLBACK_CHECKPOINT (or SKIP_BATCH
                                         if no checkpoint is available)
    """
    if spike_count <= patience_skip:
        return RecoveryAction.SKIP_BATCH
    if spike_count <= patience_lr:
        return RecoveryAction.REDUCE_LR
    if last_checkpoint_tag is not None:
        return RecoveryAction.ROLLBACK_CHECKPOINT
    return RecoveryAction.SKIP_BATCH


# ---------------------------------------------------------------------------
# User interaction (opt-in via auto_recover=False)
# ---------------------------------------------------------------------------


def prompt_user_for_action(
    stats: dict,
    epoch: int,
    step: int,
    global_step: int,
    current_lr: float,
    lr_reduction_factor: float,
    last_checkpoint_tag: str | None,
    timeout: int = 300,
    spike_reason: str = "LOSS SPIKE",
    grad_norm: float | None = None,
) -> RecoveryAction:
    """
    Print a spike alert and read the user's recovery choice from stdin.

    Only rank-0 should call this function.  Non-interactive environments
    (no tty on stdin) or timeouts fall back to the default action
    (SKIP_BATCH).
    """
    new_lr = current_lr * lr_reduction_factor
    ckpt_label = last_checkpoint_tag or "none available"

    grad_line = ""
    if grad_norm is not None:
        grad_line = f"    Grad norm:     {grad_norm:.4f}\n"

    print(
        f"\n{'=' * 70}\n"
        f"  {spike_reason} DETECTED at epoch {epoch}, step {step} (global_step {global_step})\n"
        f"    Current loss:  {stats['current_loss']:.4f}\n"
        f"    Window mean:   {stats['window_mean']:.4f}\n"
        f"    Window std:    {stats['window_std']:.4f}\n"
        f"    Spike ratio:   {stats['spike_ratio']:.2f}x mean\n"
        f"    Spike count:   {stats['spike_count']}\n"
        f"{grad_line}"
        f"\n"
        f"  Choose recovery action:\n"
        f"    [1] Skip batch (recommended - discard this batch, continue training)\n"
        f"    [2] Reduce LR by {lr_reduction_factor}x "
        f"(current LR: {current_lr:.2e} -> {new_lr:.2e}) and skip batch\n"
        f"    [3] Rollback to last checkpoint ({ckpt_label})\n"
        f"    [4] Ignore and continue training\n"
        f"\n"
        f"  Enter choice [1/2/3/4] (default=1, auto-selects in {timeout}s):\n"
        f"{'=' * 70}",
        flush=True,
    )

    choice = _read_choice_with_timeout(timeout)
    action = _parse_choice(choice, last_checkpoint_tag)
    print(f"  -> Selected: {action.name}\n", flush=True)
    return action


def _read_choice_with_timeout(timeout: int) -> str:
    """Read a single line from stdin with a timeout. Returns '' on timeout or error."""
    if not sys.stdin.isatty():
        print("  (stdin not interactive, auto-selecting default)", flush=True)
        return ""

    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.readline().strip()
        print(f"  (no input after {timeout}s, auto-selecting default)", flush=True)
        return ""
    except Exception:
        return ""


def _parse_choice(raw: str, last_checkpoint_tag: str | None) -> RecoveryAction:
    """Map user input to a RecoveryAction. Invalid / empty -> SKIP_BATCH."""
    if raw in ("1", ""):
        return RecoveryAction.SKIP_BATCH
    if raw == "2":
        return RecoveryAction.REDUCE_LR
    if raw == "3":
        if last_checkpoint_tag is None:
            print(
                "  No checkpoint available for rollback — falling back to skip batch.",
                flush=True,
            )
            return RecoveryAction.SKIP_BATCH
        return RecoveryAction.ROLLBACK_CHECKPOINT
    if raw == "4":
        return RecoveryAction.IGNORE

    print(f"  Unrecognised input '{raw}' — defaulting to skip batch.", flush=True)
    return RecoveryAction.SKIP_BATCH


# ---------------------------------------------------------------------------
# Distributed broadcast
# ---------------------------------------------------------------------------


def broadcast_action(action: RecoveryAction, src: int = 0) -> RecoveryAction:
    """
    Broadcast the chosen recovery action from *src* rank to all other ranks.

    If distributed is not initialised (single-GPU), the action is returned
    unchanged.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return action

    tensor = torch.tensor([int(action)], dtype=torch.int64, device="cuda")
    dist.broadcast(tensor, src=src)
    return RecoveryAction(tensor.item())
