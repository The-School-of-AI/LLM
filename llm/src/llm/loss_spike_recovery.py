"""
Loss Spike Detection and Interactive Recovery.

Monitors training loss using a sliding window and z-score threshold.
When a spike is detected, training pauses and the user is prompted
to choose a recovery action: skip batch, reduce LR, rollback checkpoint,
or ignore.
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
    1. The window has enough samples (at least `window_size` values).
    2. The current loss exceeds `mean + z_threshold * std` OR
       exceeds `min_spike_ratio * mean`.
    3. The absolute increase `(current - mean)` exceeds `min_abs_delta`.
    """

    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        min_spike_ratio: float = 2.0,
        min_abs_delta: float = 0.5,
    ):
        self._window: deque[float] = deque(maxlen=window_size)
        self._window_size = window_size
        self._z_threshold = z_threshold
        self._min_spike_ratio = min_spike_ratio
        self._min_abs_delta = min_abs_delta
        self._last_loss: float | None = None

    def update(self, loss: float) -> bool:
        """
        Record a new loss value and return True if it constitutes a spike.

        The value is always appended to the window regardless of whether
        a spike is detected (so the window reflects actual training history).
        """
        self._last_loss = loss

        # Not enough history yet — still in warmup.
        if len(self._window) < self._window_size:
            self._window.append(loss)
            return False

        is_spike = self._check_spike(loss)

        # Only add non-spike values to the window so that a single spike
        # doesn't corrupt the running statistics.
        if not is_spike:
            self._window.append(loss)

        return is_spike

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

    def get_stats(self) -> dict:
        """Return current window statistics for display in the user prompt."""
        w_mean = mean(self._window) if self._window else 0.0
        w_std = stdev(self._window) if len(self._window) > 1 else 0.0
        return {
            "current_loss": self._last_loss,
            "window_mean": w_mean,
            "window_std": w_std,
            "spike_ratio": (
                self._last_loss / w_mean if w_mean > 0 and self._last_loss else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------------------

_ACTION_LABELS = {
    RecoveryAction.SKIP_BATCH: "Skip batch (recommended - discard this batch, continue training)",
    RecoveryAction.REDUCE_LR: "Reduce LR by {factor}x (current LR: {lr:.2e} -> {new_lr:.2e}) and skip batch",
    RecoveryAction.ROLLBACK_CHECKPOINT: "Rollback to last checkpoint ({tag})",
    RecoveryAction.IGNORE: "Ignore and continue training",
}


def prompt_user_for_action(
    stats: dict,
    epoch: int,
    step: int,
    global_step: int,
    current_lr: float,
    lr_reduction_factor: float,
    last_checkpoint_tag: str | None,
    timeout: int = 300,
) -> RecoveryAction:
    """
    Print a spike alert and read the user's recovery choice from stdin.

    Only rank-0 should call this function.  Non-interactive environments
    (no tty on stdin) or timeouts fall back to the default action
    (SKIP_BATCH).

    Args:
        stats: dict from ``LossSpikeDetector.get_stats()``.
        epoch / step / global_step: current training position.
        current_lr: current learning rate for display.
        lr_reduction_factor: factor by which LR will be multiplied.
        last_checkpoint_tag: tag of the most recent checkpoint, or None.
        timeout: seconds to wait for input before auto-selecting default.

    Returns:
        The chosen ``RecoveryAction``.
    """
    new_lr = current_lr * lr_reduction_factor
    ckpt_label = last_checkpoint_tag or "none available"

    print(
        f"\n{'=' * 70}\n"
        f"  LOSS SPIKE DETECTED at epoch {epoch}, step {step} (global_step {global_step})\n"
        f"    Current loss:  {stats['current_loss']:.4f}\n"
        f"    Window mean:   {stats['window_mean']:.4f}\n"
        f"    Window std:    {stats['window_std']:.4f}\n"
        f"    Spike ratio:   {stats['spike_ratio']:.2f}x mean\n"
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
    # If stdin is not a tty (e.g. launched by DeepSpeed without --tty),
    # fall back immediately to the default.
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
