"""
IDFT Trainer
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Custom SFTTrainer subclass that replaces standard cross-entropy loss
with the IDFT loss from "Towards On-Policy SFT" (arXiv:2602.12222).
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
from trl import SFTTrainer, SFTConfig

from idft_loss import idft_loss

logger = logging.getLogger(__name__)


class IDFTTrainer(SFTTrainer):
    """
    SFTTrainer with IDFT loss.

    Overrides compute_loss() to use the IDFT reweighted loss and
    logs phi/gamma diagnostics during training.
    """

    def __init__(
        self,
        clip_B: float = 5.0,
        log_diagnostics_every: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_B = clip_B
        self.log_diagnostics_every = log_diagnostics_every
        self._idft_step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to use IDFT loss instead of standard CE."""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Build attention mask for non-padding, non-ignored tokens
        # Labels with -100 are ignored (prompt tokens)
        label_mask = (shift_labels != -100).float()

        # Replace -100 with 0 for gather (won't affect loss due to mask)
        safe_labels = shift_labels.clone()
        safe_labels[shift_labels == -100] = 0

        loss, diagnostics = idft_loss(
            logits=shift_logits,
            labels=safe_labels,
            attention_mask=label_mask,
            clip_B=self.clip_B,
        )

        # Log diagnostics periodically
        self._idft_step_count += 1
        if self._idft_step_count % self.log_diagnostics_every == 0:
            self.log({f"idft/{k}": v for k, v in diagnostics.items()})

        if return_outputs:
            return loss, outputs
        return loss
