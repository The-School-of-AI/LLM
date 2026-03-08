"""End-to-end test: train a small LLM, checkpoint RNG state, resume, verify losses match.

Strategy: both baseline and resume runs create a NEW DataLoader at the checkpoint
boundary. This ensures the shuffle permutation for phase-2 is determined entirely
by the RNG state at that moment — which is exactly what RNGStateManager restores.
"""

import copy
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.rng_state_manager import RNGStateManager

TOTAL_STEPS = 10
CHECKPOINT_STEP = 5
SEED = 42
BATCH_SIZE = 4
SEQ_LEN = 64


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _prepare_dataset(tokenizer, num_samples: int = 50) -> TensorDataset:
    """Load wikitext-2-raw-v1 and tokenize into a small TensorDataset."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()][:num_samples]
    full_text = " ".join(texts)

    tokens = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=False,
    )["input_ids"].squeeze(0)

    n_chunks = len(tokens) // SEQ_LEN
    tokens = tokens[: n_chunks * SEQ_LEN].reshape(n_chunks, SEQ_LEN)
    return TensorDataset(tokens)


def _train_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    num_steps: int,
) -> list[float]:
    """Train for num_steps and return the loss at each step."""
    model.train()
    losses = []
    step = 0
    while step < num_steps:
        for (batch,) in loader:
            if step >= num_steps:
                break
            labels = batch.clone()
            out = model(input_ids=batch, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
    return losses


def _run_two_phase_training(
    dataset: TensorDataset,
    trash_and_restore: bool,
) -> list[float]:
    """Run training in two phases with a checkpoint boundary between them.

    Phase 1: seed → train CHECKPOINT_STEP steps
    Checkpoint: capture RNG + model + optimizer state
    Phase 2: (optionally trash & restore RNG) → new DataLoader → train remaining steps

    Both baseline (trash_and_restore=False) and resume (trash_and_restore=True)
    create a fresh DataLoader at the checkpoint boundary, so the phase-2 shuffle
    order depends entirely on the RNG state — which is what we're testing.
    """
    _set_seed(SEED)
    model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --- Phase 1 ---
    loader1 = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    _train_steps(model, optimizer, loader1, CHECKPOINT_STEP)

    # --- Checkpoint boundary ---
    rng_snapshot = RNGStateManager.capture()
    model_snapshot = copy.deepcopy(model.state_dict())
    opt_snapshot = copy.deepcopy(optimizer.state_dict())

    if trash_and_restore:
        # Simulate process restart: destroy all RNG state
        _set_seed(999)
        _ = [random.random() for _ in range(100)]
        _ = np.random.rand(100)
        _ = torch.randn(100)

        # Restore from "checkpoint"
        RNGStateManager.restore(rng_snapshot)
        model.load_state_dict(model_snapshot)
        optimizer.load_state_dict(opt_snapshot)

    # --- Phase 2: new DataLoader (shuffle order determined by current RNG) ---
    loader2 = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    phase2_losses = _train_steps(model, optimizer, loader2, TOTAL_STEPS - CHECKPOINT_STEP)
    return phase2_losses


class TestRNGStateE2E:
    """Prove that RNG capture/restore produces identical training after resume."""

    @pytest.fixture(autouse=True, scope="class")
    def _setup(self, request):
        """Load tokenizer and dataset once for all tests in this class."""
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        request.cls.dataset = _prepare_dataset(tokenizer)

    def test_resumed_training_matches_baseline(self):
        """Phase-2 losses must be identical whether or not RNG was trashed and restored."""
        baseline_losses = _run_two_phase_training(self.dataset, trash_and_restore=False)
        resumed_losses = _run_two_phase_training(self.dataset, trash_and_restore=True)

        assert len(baseline_losses) == len(resumed_losses)
        for i, (bl, rl) in enumerate(zip(baseline_losses, resumed_losses)):
            assert bl == pytest.approx(rl, abs=1e-6), (
                f"Step {CHECKPOINT_STEP + i + 1}: baseline loss {bl} != resumed loss {rl}"
            )

    def test_without_rng_restore_losses_diverge(self):
        """Without RNG restore, losses must differ (proving the restore matters)."""
        baseline_losses = _run_two_phase_training(self.dataset, trash_and_restore=False)

        # Run with trashed RNG but NO restore
        _set_seed(SEED)
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        loader1 = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        _train_steps(model, optimizer, loader1, CHECKPOINT_STEP)

        # Trash RNG without restoring
        _set_seed(999)

        loader2 = DataLoader(self.dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        no_restore_losses = _train_steps(model, optimizer, loader2, TOTAL_STEPS - CHECKPOINT_STEP)

        mismatches = sum(
            1 for bl, nl in zip(baseline_losses, no_restore_losses) if abs(bl - nl) > 1e-6
        )
        assert mismatches > 0, "Expected losses to diverge without RNG restore"
