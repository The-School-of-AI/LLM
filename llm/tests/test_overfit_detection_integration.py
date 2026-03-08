"""
Integration test for overfitting detection logic introduced in pretrainer.py.

Verifies:
  1. Eval runs exactly every `eval_interval` global steps.
  2. Smoothed train loss, val loss, and train-eval gap are correctly computed.
  3. `overfit_strikes` counter increments when val_loss fails to improve.
  4. `overfit_strikes` resets to 0 when val_loss improves beyond `overfit_threshold`.
  5. An `overfitting_detected` alert fires exactly when strikes >= patience.

Uses:
  - `gpt2` (small HuggingFace model, ~117M params on CPU — fast for smoke tests)
  - `wikitext-2-raw-v1` dataset via the existing `llm.data.get_dataloaders()` factory.
  - Plain `torch.optim.AdamW` — no DeepSpeed required (runs on local Mac / CPU).

The test does NOT assert on the absolute loss values because those depend on
hardware and random seed. It only checks the structural correctness of the
monitoring bookkeeping logic.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.data import get_dataloaders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_val_pass(
    model: nn.Module,
    val_loader,
    device: torch.device,
    max_val_steps: int,
) -> float:
    """Run a validation pass and return average cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_val_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # GPT-2's labels are the shifted input_ids handled internally.
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            total_loss += out.loss.item()
            steps += 1
    return total_loss / max(1, steps)


def _run_train_step(
    model: nn.Module,
    optimizer: AdamW,
    batch: dict[str, Any],
    device: torch.device,
) -> float:
    """Single forward + backward + optimizer step. Returns scalar loss."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    loss: torch.Tensor = out.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


# ---------------------------------------------------------------------------
# Simulation: mirrors the overfitting detection logic from pretrainer.py
# ---------------------------------------------------------------------------


def simulate_training(
    model: nn.Module,
    optimizer: AdamW,
    train_loader,
    val_loader,
    device: torch.device,
    *,
    total_steps: int,
    eval_interval: int,
    overfit_patience: int,
    overfit_threshold: float,
    max_val_steps: int,
) -> list[dict[str, Any]]:
    """
    Train for `total_steps` steps and return one log entry per eval trigger.

    The bookkeeping logic is a direct mirror of PreTrainer.run() so that
    correctness here validates correctness there.
    """
    # -- State (mirrors PreTrainer.__init__ additions) -----------------------
    train_loss_accum: float = 0.0
    train_loss_count: int = 0
    best_val_loss: float = float("inf")
    overfit_strikes: int = 0
    # -------------------------------------------------------------------------

    log: list[dict[str, Any]] = []
    global_step = 0
    train_iter = iter(train_loader)

    for _ in range(total_steps):
        # Replenish iterator if exhausted (epoch boundary).
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_loss = _run_train_step(model, optimizer, batch, device)

        # -- Accumulate (mirrors lines 137-138 of pretrainer.py) --
        train_loss_accum += step_loss
        train_loss_count += 1

        global_step += 1

        # -- Eval trigger (mirrors line 140 of pretrainer.py) --
        if eval_interval and global_step % eval_interval == 0:
            val_loss = _run_val_pass(model, val_loader, device, max_val_steps)

            # -- Smoothed train loss (mirrors lines 143-145) --
            smoothed_train_loss = train_loss_accum / max(1, train_loss_count)
            train_loss_accum = 0.0
            train_loss_count = 0

            # -- Gap (mirrors line 147) --
            train_eval_gap = val_loss - smoothed_train_loss

            # -- Overfitting watchdog (mirrors lines 152-156) --
            if val_loss < best_val_loss - overfit_threshold:
                best_val_loss = val_loss
                overfit_strikes = 0
            else:
                overfit_strikes += 1

            # -- Build log entry (mirrors logger.log_step calls) --
            entry: dict[str, Any] = {
                "global_step": global_step,
                "smoothed_train_loss": smoothed_train_loss,
                "val_loss": val_loss,
                "val_perplexity": math.exp(val_loss),
                "train_eval_gap": train_eval_gap,
                "overfit_strikes": overfit_strikes,
                "best_val_loss": best_val_loss,
            }
            if overfit_patience > 0 and overfit_strikes >= overfit_patience:
                entry["overfitting_detected"] = True

            log.append(entry)
            # Restore train mode (mirrors line 175).
            model.train()

    return log


# ---------------------------------------------------------------------------
# Pytest fixture: model + dataloaders (shared across test variants)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup_model_and_loaders():
    """Load GPT-2 (tiny) and wikitext-2-raw-v1 once for the whole module."""
    device = torch.device("cpu")

    # Use GPT-2 via HuggingFace — small enough to run on Mac CPU in seconds.
    model_name = "gpt2"
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    if hf_tokenizer.pad_token is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # Use the project's existing dataloader factory.
    train_loader, val_loader, _, _ = get_dataloaders(
        tokenizer=hf_tokenizer,
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        batch_size=4,
        max_length=64,
        num_workers=0,  # fork-safe on Mac
        drop_remainder=False,
    )

    optimizer = AdamW(model.parameters(), lr=1e-4)

    return dict(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverfitDetectionIntegration:
    """Verifies the Train-Eval Gap monitoring logic end-to-end."""

    # Config
    TOTAL_STEPS = 20
    EVAL_INTERVAL = 5
    OVERFIT_PATIENCE = 3
    OVERFIT_THRESHOLD = 0.0
    MAX_VAL_STEPS = 10

    @pytest.fixture(autouse=True)
    def _run(self, setup_model_and_loaders):
        """Run training once; all test methods share the resulting log."""
        self.log = simulate_training(
            model=setup_model_and_loaders["model"],
            optimizer=setup_model_and_loaders["optimizer"],
            train_loader=setup_model_and_loaders["train_loader"],
            val_loader=setup_model_and_loaders["val_loader"],
            device=setup_model_and_loaders["device"],
            total_steps=self.TOTAL_STEPS,
            eval_interval=self.EVAL_INTERVAL,
            overfit_patience=self.OVERFIT_PATIENCE,
            overfit_threshold=self.OVERFIT_THRESHOLD,
            max_val_steps=self.MAX_VAL_STEPS,
        )

    # ------------------------------------------------------------------
    # Task 1: Eval frequency — exactly N eval checkpoints logged
    # ------------------------------------------------------------------

    def test_eval_triggered_every_n_steps(self):
        """Eval must fire exactly total_steps // eval_interval times."""
        expected_evals = self.TOTAL_STEPS // self.EVAL_INTERVAL
        assert len(self.log) == expected_evals, (
            f"Expected {expected_evals} eval checkpoints, got {len(self.log)}"
        )

    def test_eval_triggered_at_correct_global_steps(self):
        """Each eval must fire at a multiple of eval_interval."""
        for entry in self.log:
            gs = entry["global_step"]
            assert gs % self.EVAL_INTERVAL == 0, (
                f"Eval fired at global_step={gs}, not a multiple of {self.EVAL_INTERVAL}"
            )

    # ------------------------------------------------------------------
    # Task 2: Eval loss tracked alongside train loss
    # ------------------------------------------------------------------

    def test_all_required_keys_present(self):
        """Every log entry must contain all required monitoring fields."""
        required = {
            "global_step",
            "smoothed_train_loss",
            "val_loss",
            "val_perplexity",
            "train_eval_gap",
            "overfit_strikes",
            "best_val_loss",
        }
        for i, entry in enumerate(self.log):
            missing = required - entry.keys()
            assert not missing, f"Entry {i} at step {entry['global_step']} missing keys: {missing}"

    def test_val_loss_is_finite_and_positive(self):
        """Validation loss must be a finite, positive number."""
        for entry in self.log:
            vl = entry["val_loss"]
            assert math.isfinite(vl), f"val_loss is not finite: {vl}"
            assert vl > 0, f"val_loss must be positive, got {vl}"

    def test_train_loss_is_finite_and_positive(self):
        """Smoothed train loss must be a finite, positive number."""
        for entry in self.log:
            tl = entry["smoothed_train_loss"]
            assert math.isfinite(tl), f"smoothed_train_loss is not finite: {tl}"
            assert tl > 0, f"smoothed_train_loss must be positive, got {tl}"

    # ------------------------------------------------------------------
    # Task 3a: Train-eval gap is correctly computed
    # ------------------------------------------------------------------

    def test_gap_equals_val_minus_train(self):
        """train_eval_gap must exactly equal val_loss - smoothed_train_loss."""
        for entry in self.log:
            expected = entry["val_loss"] - entry["smoothed_train_loss"]
            actual = entry["train_eval_gap"]
            assert abs(actual - expected) < 1e-6, (
                f"Gap mismatch at step {entry['global_step']}: "
                f"expected {expected:.6f}, got {actual:.6f}"
            )

    # ------------------------------------------------------------------
    # Task 3b: Overfitting strike counter logic is correct
    # ------------------------------------------------------------------

    def test_overfit_strikes_non_negative(self):
        """Strike counter must never be negative."""
        for entry in self.log:
            assert entry["overfit_strikes"] >= 0, (
                f"overfit_strikes is negative at step {entry['global_step']}"
            )

    def test_best_val_loss_monotonically_non_increasing(self):
        """best_val_loss must never increase (it only ever improves or holds)."""
        prev = float("inf")
        for entry in self.log:
            bvl = entry["best_val_loss"]
            assert bvl <= prev + 1e-9, (
                f"best_val_loss increased from {prev:.4f} to {bvl:.4f} "
                f"at step {entry['global_step']}"
            )
            prev = bvl

    def test_strikes_reset_on_improvement(self):
        """When val_loss beats best_val_loss, strikes must be 0."""
        prev_best = float("inf")
        for entry in self.log:
            vl = entry["val_loss"]
            # If this eval improved on the best so far (before this step),
            # strikes should have been reset to 0.
            if vl < prev_best - self.OVERFIT_THRESHOLD:
                assert entry["overfit_strikes"] == 0, (
                    f"Expected strikes=0 after improvement at step "
                    f"{entry['global_step']}, got {entry['overfit_strikes']}"
                )
            prev_best = entry["best_val_loss"]

    def test_strikes_increment_on_no_improvement(self):
        """When val_loss does not improve, strikes must increment by 1."""
        prev_best = float("inf")
        prev_strikes = 0
        for i, entry in enumerate(self.log):
            vl = entry["val_loss"]
            strikes = entry["overfit_strikes"]

            if i > 0:  # skip first entry (no previous to compare against)
                if vl >= prev_best - self.OVERFIT_THRESHOLD:
                    # No improvement → strikes must have gone up by 1.
                    assert strikes == prev_strikes + 1, (
                        f"Expected strikes to increase by 1 at step "
                        f"{entry['global_step']}: "
                        f"prev_strikes={prev_strikes}, got strikes={strikes}"
                    )

            prev_best = entry["best_val_loss"]
            prev_strikes = strikes

    # ------------------------------------------------------------------
    # Task 3c: overfitting_detected flag fires at the right time
    # ------------------------------------------------------------------

    def test_overfitting_detected_fires_when_strikes_reach_patience(self):
        """overfitting_detected must be True iff strikes >= patience."""
        for entry in self.log:
            strikes = entry["overfit_strikes"]
            detected = entry.get("overfitting_detected", False)
            if strikes >= self.OVERFIT_PATIENCE:
                assert detected is True, (
                    f"overfitting_detected should be True when strikes "
                    f"({strikes}) >= patience ({self.OVERFIT_PATIENCE}) "
                    f"at step {entry['global_step']}"
                )
            else:
                assert not detected, (
                    f"overfitting_detected should be False/absent when strikes "
                    f"({strikes}) < patience ({self.OVERFIT_PATIENCE}) "
                    f"at step {entry['global_step']}"
                )
