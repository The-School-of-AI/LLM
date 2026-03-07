"""
Training State Manager — step counter restoration after checkpoint resume.

Provides a single source of truth for training progress (global step, epoch,
tokens seen, samples seen) and ensures correct logging continuity when
resuming from a checkpoint in distributed (DDP / FSDP / DeepSpeed) setups.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from llm.logger import Logger, Metrics

log = logging.getLogger(__name__)

_STATE_KEY = "training_state"


@dataclass
class TrainingState:
    """Source of truth for training progress."""

    global_step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    samples_seen: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "tokens_seen": self.tokens_seen,
            "samples_seen": self.samples_seen,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingState":
        return cls(
            global_step=int(d.get("global_step", 0)),
            epoch=int(d.get("epoch", 0)),
            tokens_seen=int(d.get("tokens_seen", 0)),
            samples_seen=int(d.get("samples_seen", 0)),
        )


class StepManager:
    """Manages training progress state including checkpoint resume.

    Responsibilities:
        - Restore state from checkpoint ``client_state`` dicts.
        - Synchronize state across distributed workers.
        - Increment global step after each optimizer step.
        - Provide the correct step for logging.
        - Persist training state into checkpoints.

    Usage::

        mgr = StepManager()
        state = mgr.restore(client_state)

        # training loop
        optimizer.step()
        mgr.increment(tokens=batch_tokens, samples=batch_size)

        mgr.log(metrics, logger)

        checkpoint = mgr.inject_state(checkpoint)
    """

    def __init__(self) -> None:
        self._state = TrainingState()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        return self._state.global_step

    @property
    def epoch(self) -> int:
        return self._state.epoch

    @epoch.setter
    def epoch(self, value: int) -> None:
        self._state.epoch = value

    @property
    def tokens_seen(self) -> int:
        return self._state.tokens_seen

    @property
    def samples_seen(self) -> int:
        return self._state.samples_seen

    @property
    def state(self) -> TrainingState:
        return self._state

    # ------------------------------------------------------------------
    # 1. Restore
    # ------------------------------------------------------------------

    def restore(self, client_state: Optional[Dict[str, Any]] = None) -> TrainingState:
        """Restore training state from a checkpoint's ``client_state``.

        Args:
            client_state: The dict returned by the checkpoint manager's
                ``load_checkpoint``.  May be ``None`` (no checkpoint) or may
                lack a ``training_state`` key (legacy checkpoint).

        Returns:
            The restored (or freshly initialised) :class:`TrainingState`.
        """
        if client_state is None:
            self._state = TrainingState()
            self._sync_distributed()
            return self._state

        if _STATE_KEY in client_state:
            try:
                self._state = TrainingState.from_dict(client_state[_STATE_KEY])
            except Exception as exc:
                warnings.warn(
                    f"Corrupted training_state in checkpoint, falling back to "
                    f"defaults: {exc}",
                    stacklevel=2,
                )
                self._state = TrainingState()
        else:
            # Legacy checkpoint — reconstruct from flat keys.
            self._state = TrainingState(
                global_step=int(client_state.get("global_step", 0)),
                epoch=int(client_state.get("epoch", 0)),
                tokens_seen=int(client_state.get("tokens_seen", 0)),
                samples_seen=int(client_state.get("samples_seen", 0)),
            )
            if client_state:
                warnings.warn(
                    "Checkpoint does not contain 'training_state' key. "
                    "Falling back to flat client_state fields.",
                    stacklevel=2,
                )

        self._sync_distributed()
        return self._state

    # ------------------------------------------------------------------
    # 2. Distributed sync
    # ------------------------------------------------------------------

    def _sync_distributed(self) -> None:
        """Broadcast state from rank 0 to all workers."""
        if not (dist.is_available() and dist.is_initialized()):
            return

        tensor = torch.tensor(
            [
                self._state.global_step,
                self._state.epoch,
                self._state.tokens_seen,
                self._state.samples_seen,
            ],
            dtype=torch.long,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dist.broadcast(tensor, src=0)

        self._state.global_step = int(tensor[0].item())
        self._state.epoch = int(tensor[1].item())
        self._state.tokens_seen = int(tensor[2].item())
        self._state.samples_seen = int(tensor[3].item())

    # ------------------------------------------------------------------
    # 3. Increment
    # ------------------------------------------------------------------

    def increment(self, tokens: int = 0, samples: int = 0) -> None:
        """Increment global step (call **after** ``optimizer.step()``).

        Args:
            tokens: Number of tokens processed in this optimizer step.
            samples: Number of samples processed in this optimizer step.
        """
        self._state.global_step += 1
        self._state.tokens_seen += tokens
        self._state.samples_seen += samples

    # ------------------------------------------------------------------
    # 4. Get step
    # ------------------------------------------------------------------

    def get_step(self) -> int:
        """Return the current global training step."""
        return self._state.global_step

    # ------------------------------------------------------------------
    # 5. Logging wrapper
    # ------------------------------------------------------------------

    def log(self, metrics: Metrics, logger: Logger) -> None:
        """Log *metrics* at the current global step.

        Automatically attaches the correct step so that logs continue
        seamlessly after a checkpoint resume.
        """
        logger.log_metrics(self._state.global_step, metrics)

    # ------------------------------------------------------------------
    # 6. Checkpoint persistence
    # ------------------------------------------------------------------

    def inject_state(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Add training state to a checkpoint dict before saving.

        Args:
            checkpoint: The checkpoint dictionary being assembled.

        Returns:
            The same dictionary, with ``training_state`` injected.
        """
        checkpoint[_STATE_KEY] = self._state.to_dict()
        return checkpoint
