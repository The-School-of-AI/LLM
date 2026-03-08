"""Guard module that detects loss discontinuities after checkpoint resume.

A sudden loss spike after resume usually indicates that some training state
was not restored correctly (optimizer state, LR scheduler state, gradient
accumulation, etc.).  This module records recent optimizer-step losses,
saves statistics in checkpoint metadata, and verifies continuity on resume.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class LossContinuityGuard:
    """Lightweight guard that verifies loss continuity after checkpoint resume.

    Usage::

        # Initialization
        guard = LossContinuityGuard()
        guard.restore(client_state.get("loss_guard", {}))

        # Training loop (after each optimizer step)
        guard.observe(loss)

        # Checkpoint save
        client_state["loss_guard"] = guard.state_dict()

    Args:
        window_size: Number of optimizer-step losses to keep in the rolling
            window.  Used to compute pre-resume statistics and to collect
            enough post-resume data before verification.
        tolerance_sigma: Number of standard deviations allowed for the
            post-resume mean to deviate from the pre-resume mean.
    """

    def __init__(self, window_size: int = 50, tolerance_sigma: float = 3.0) -> None:
        self._window_size = window_size
        self._tolerance_sigma = tolerance_sigma

        self._loss_window: deque[float] = deque(maxlen=window_size)

        # Pre-resume statistics (populated by ``restore``)
        self._pre_mean: float | None = None
        self._pre_std: float | None = None

        # Verification state
        self._pending_verification = False
        self._verification_done = False
        self._verification_result: bool = True
        self._post_resume_losses: list[float] = []

    # ------------------------------------------------------------------
    # Checkpoint integration
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint-safe state dictionary.

        Intended to be stored as ``client_state["loss_guard"]``.
        """
        window = list(self._loss_window)
        if len(window) == 0:
            return {}

        mean = sum(window) / len(window)
        variance = sum((x - mean) ** 2 for x in window) / len(window)
        std = variance**0.5

        return {
            "loss_window": window,
            "loss_mean": mean,
            "loss_std": std,
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore guard state from checkpoint metadata.

        If *state* is empty or missing expected keys the guard starts fresh
        with verification disabled.

        Args:
            state: Dictionary previously returned by :meth:`state_dict`,
                typically ``client_state.get("loss_guard", {})``.
        """
        if not state:
            self._pending_verification = False
            return

        window = state.get("loss_window")
        mean = state.get("loss_mean")
        std = state.get("loss_std")

        if window is None or mean is None or std is None:
            self._pending_verification = False
            return

        self._loss_window = deque(window, maxlen=self._window_size)
        self._pre_mean = mean
        self._pre_std = std
        self._pending_verification = True
        self._verification_done = False
        self._post_resume_losses = []

        logger.info(
            "LossContinuityGuard restored: pre_mean=%.4f, pre_std=%.4f, "
            "window_len=%d",
            mean,
            std,
            len(window),
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def observe(self, loss: torch.Tensor) -> None:
        """Record a loss value after an optimizer step.

        The loss is globally averaged across all ranks when running in
        distributed mode, then appended to the rolling window.

        Args:
            loss: Scalar loss tensor from the current optimizer step.
        """
        loss_tensor = loss.detach().float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensor = loss_tensor / dist.get_world_size()

        loss_value = loss_tensor.item()
        self._loss_window.append(loss_value)

        if self._pending_verification and not self._verification_done:
            self._post_resume_losses.append(loss_value)
            if len(self._post_resume_losses) >= self._window_size:
                self.verify()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> bool:
        """Check whether post-resume losses are consistent with pre-resume stats.

        Verification runs **once** after resume, when enough new losses have
        been collected (i.e. ``window_size`` optimizer steps).

        Returns:
            ``True`` if the loss is continuous (or verification is not
            applicable), ``False`` if a discontinuity is detected.
        """
        if self._verification_done or not self._pending_verification:
            return self._verification_result

        self._verification_done = True
        self._pending_verification = False

        if self._pre_mean is None or self._pre_std is None:
            return True

        if not self._post_resume_losses:
            return True

        new_mean = sum(self._post_resume_losses) / len(self._post_resume_losses)
        delta = abs(new_mean - self._pre_mean)

        # Sigma-based check
        sigma_ok = delta <= self._tolerance_sigma * self._pre_std

        # Relative-difference check (fallback for near-zero std)
        if self._pre_mean != 0:
            relative_diff = delta / abs(self._pre_mean)
        else:
            relative_diff = 0.0 if delta == 0 else float("inf")
        relative_ok = relative_diff <= 0.20

        passed = sigma_ok or relative_ok

        if passed:
            logger.info(
                "LossContinuityGuard: verification PASSED "
                "(pre_mean=%.4f, post_mean=%.4f, delta=%.4f, "
                "pre_std=%.4f, rel_diff=%.2f%%)",
                self._pre_mean,
                new_mean,
                delta,
                self._pre_std,
                relative_diff * 100,
            )
        else:
            logger.warning(
                "LossContinuityGuard: Loss discontinuity detected after resume! "
                "pre_mean=%.4f, post_mean=%.4f, delta=%.4f, "
                "threshold=%.4f (%.1f * std=%.4f), rel_diff=%.2f%%. "
                "This may indicate that optimizer state, LR scheduler, or "
                "gradient accumulation was not restored correctly.",
                self._pre_mean,
                new_mean,
                delta,
                self._tolerance_sigma * self._pre_std,
                self._tolerance_sigma,
                self._pre_std,
                relative_diff * 100,
            )

        self._verification_result = passed
        return passed
