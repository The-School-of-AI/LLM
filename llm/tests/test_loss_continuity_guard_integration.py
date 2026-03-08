"""End-to-end integration test for LossContinuityGuard.

Uses GPT-2 (small) + wikitext-2-raw-v1 to run a real training loop.
No DeepSpeed or S3 required — validates the guard logic directly.

Test plan
---------
test_normal_resume_passes
    Warm-up for WARMUP_STEPS steps (guard silent, loss descends and stabilises)
    Record PRE_STEPS more steps into guard window (stable, low-variance losses)
    Save guard state  →  restore  →  continue training PRE_STEPS + POST_STEPS
    verify()          →  must return True

test_broken_optimizer_detected
    Same warm-up + recording phase → save guard state
    "Break" the model: re-initialise all weights to random (simulates model
    weights restored incorrectly / not at all) → fresh optimizer
    Restore guard state  →  train POST_STEPS steps
    verify()             →  must return False (massive loss jump is detectable)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.loss_continuity_guard import LossContinuityGuard

# ---------------------------------------------------------------------------
# Logging: surface guard INFO / WARNING messages during the test run
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s [%(name)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Constants — tuned for fast CPU execution
# ---------------------------------------------------------------------------
MODEL_NAME = "gpt2"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MAX_LENGTH = 64        # sequence length per sample
BATCH_SIZE = 2         # samples per step
WINDOW_SIZE = 20       # guard window size
# Warm-up steps BEFORE the guard starts recording.
# After ~80 steps with lr=3e-4, GPT-2 loss settles from ~5 → ~3 and std < 0.2.
WARMUP_STEPS = 80
# Steps during which the guard records the pre-resume window (stable losses).
PRE_STEPS = WINDOW_SIZE + 5
# Steps the guard collects after resume to trigger verify().
POST_STEPS = WINDOW_SIZE
DEVICE = torch.device("cpu")
LEARNING_RATE = 3e-4   # higher LR → faster convergence on CPU


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _build_model_and_tokenizer():
    """Load GPT-2 and its tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return model, tokenizer


def _build_batches(tokenizer, n_batches: int) -> list[dict[str, torch.Tensor]]:
    """
    Stream a few pages of wikitext-2 text, tokenize, return ``n_batches``
    batches of shape ``[BATCH_SIZE, MAX_LENGTH]``.
    """
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
    )

    samples: list[str] = []
    for row in dataset:
        text = row["text"].strip()  # type: ignore[index]
        if text:
            samples.append(text)
        if len(samples) >= n_batches * BATCH_SIZE * 2:
            break

    encoded = tokenizer(
        samples,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

    batches: list[dict[str, torch.Tensor]] = []
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    idx = 0
    while len(batches) < n_batches:
        end = idx + BATCH_SIZE
        if end > len(input_ids):
            idx = 0
            end = BATCH_SIZE
        batches.append(
            {
                "input_ids": input_ids[idx:end],
                "attention_mask": attention_mask[idx:end],
            }
        )
        idx = end

    return batches


def _compute_loss(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Forward pass returning a scalar cross-entropy loss."""
    outputs = model(
        input_ids=batch["input_ids"].to(DEVICE),
        attention_mask=batch["attention_mask"].to(DEVICE),
        labels=batch["input_ids"].to(DEVICE),
    )
    return outputs.loss


def _warmup(
    model,
    optimizer: torch.optim.Optimizer,
    batches: list[dict[str, torch.Tensor]],
    n_steps: int,
    start_batch: int = 0,
) -> tuple[int, float]:
    """Run n_steps warm-up steps (guard not involved). Returns (next_batch_idx, final_loss)."""
    model.train()
    batch_idx = start_batch
    last_loss = 0.0
    for _ in range(n_steps):
        batch = batches[batch_idx % len(batches)]
        batch_idx += 1
        optimizer.zero_grad()
        loss = _compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
    return batch_idx, last_loss


def _train_steps(
    model,
    optimizer: torch.optim.Optimizer,
    batches: list[dict[str, torch.Tensor]],
    guard: LossContinuityGuard,
    n_steps: int,
    start_batch: int = 0,
) -> int:
    """
    Run ``n_steps`` optimizer steps, recording each loss into the guard.
    Returns the index of the next batch.
    """
    model.train()
    batch_idx = start_batch
    for _ in range(n_steps):
        batch = batches[batch_idx % len(batches)]
        batch_idx += 1
        optimizer.zero_grad()
        loss = _compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        guard.observe(loss)
    return batch_idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normal_resume_passes(tmp_path: Path):
    """
    A correctly restored guard should report loss continuity as PASS.

    Warm-up (WARMUP_STEPS) → guard records PRE_STEPS stable losses → save state.
    Resume with the **same** model + optimizer → guard records POST_STEPS losses.
    verify() must return True.
    """
    print("\n" + "=" * 60)
    print("  test_normal_resume_passes")
    print("=" * 60)

    model, tokenizer = _build_model_and_tokenizer()
    total_steps = WARMUP_STEPS + PRE_STEPS + POST_STEPS + 10
    batches = _build_batches(tokenizer, total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Warm-up: let loss descend and stabilise ───────────────────────────
    print(f"\n[Warm-up] Running {WARMUP_STEPS} steps (guard silent)…")
    next_batch, warmup_loss = _warmup(model, optimizer, batches, WARMUP_STEPS)
    print(f"  [✓] Warm-up complete. Final loss ≈ {warmup_loss:.4f}")

    # ── Phase 1: record stable pre-resume losses ──────────────────────────
    guard = LossContinuityGuard(window_size=WINDOW_SIZE)
    print(f"\n[Phase 1] Training {PRE_STEPS} steps with guard recording…")
    next_batch = _train_steps(model, optimizer, batches, guard, PRE_STEPS, next_batch)

    guard_state_path = tmp_path / "guard_state.pt"
    torch.save(guard.state_dict(), guard_state_path)
    saved_state = torch.load(guard_state_path, weights_only=True)
    print(
        f"  [✓] Guard state saved → {guard_state_path.name}\n"
        f"      pre_mean={saved_state['loss_mean']:.4f}  "
        f"pre_std={saved_state['loss_std']:.4f}  "
        f"window_len={len(saved_state['loss_window'])}"
    )

    # ── Phase 2: clean resume (same model + optimizer) ────────────────────
    guard2 = LossContinuityGuard(window_size=WINDOW_SIZE)
    guard2.restore(saved_state)
    print(f"\n[Phase 2] Clean resume. Training {POST_STEPS} steps…")
    _train_steps(model, optimizer, batches, guard2, POST_STEPS, next_batch)

    # ── Verification ──────────────────────────────────────────────────────
    result = guard2.verify()
    print(f"\n  [Guard] verify() → {result}")
    assert result is True, (
        "Expected loss continuity PASS after a clean resume, but guard reported FAIL. "
        f"pre_mean={saved_state['loss_mean']:.4f}, pre_std={saved_state['loss_std']:.4f}"
    )
    print("\n  ✅ test_normal_resume_passes PASSED")


def test_broken_optimizer_detected(tmp_path: Path):
    """
    Corrupted training state (model weights re-initialised) must be detected.

    Warm-up → record PRE_STEPS stable losses → save guard state.
    "Break": re-initialise all model weights + create a fresh optimizer
             (simulates model checkpoint not properly restored).
    Restore guard state → train POST_STEPS steps.
    verify() must return False.
    """
    print("\n" + "=" * 60)
    print("  test_broken_optimizer_detected")
    print("=" * 60)

    model, tokenizer = _build_model_and_tokenizer()
    total_steps = WARMUP_STEPS + PRE_STEPS + POST_STEPS + 10
    batches = _build_batches(tokenizer, total_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Warm-up ───────────────────────────────────────────────────────────
    print(f"\n[Warm-up] Running {WARMUP_STEPS} steps (guard silent)…")
    next_batch, warmup_loss = _warmup(model, optimizer, batches, WARMUP_STEPS)
    print(f"  [✓] Warm-up complete. Final loss ≈ {warmup_loss:.4f}")

    # ── Phase 1: record stable pre-resume losses ──────────────────────────
    guard = LossContinuityGuard(window_size=WINDOW_SIZE)
    print(f"\n[Phase 1] Training {PRE_STEPS} steps with guard recording…")
    next_batch = _train_steps(model, optimizer, batches, guard, PRE_STEPS, next_batch)

    guard_state_path = tmp_path / "guard_state.pt"
    torch.save(guard.state_dict(), guard_state_path)
    saved_state = torch.load(guard_state_path, weights_only=True)
    print(
        f"  [✓] Guard state saved.\n"
        f"      pre_mean={saved_state['loss_mean']:.4f}  "
        f"pre_std={saved_state['loss_std']:.4f}"
    )

    # ── Phase 2: broken resume — re-initialise model weights ─────────────
    # Re-initialising all weights mimics an incorrectly restored checkpoint:
    # the model reverts to random initialisation, causing loss to jump back
    # toward log(vocab_size) ≈ 10.8 — far beyond any reasonable tolerance.
    print("\n[BREAK] Re-initialising all model weights (corrupt resume simulation)…")
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    broken_optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    guard2 = LossContinuityGuard(window_size=WINDOW_SIZE)
    guard2.restore(saved_state)
    print(f"\n[Phase 2] BROKEN resume (re-init weights). Training {POST_STEPS} steps…")
    _train_steps(model, broken_optimizer, batches, guard2, POST_STEPS, next_batch)

    # ── Verification ──────────────────────────────────────────────────────
    result = guard2.verify()
    print(f"\n  [Guard] verify() → {result}")
    assert result is False, (
        "Expected loss discontinuity FAIL after weight re-initialisation, "
        "but guard reported PASS. "
        f"pre_mean={saved_state['loss_mean']:.4f}, pre_std={saved_state['loss_std']:.4f}"
    )
    print("\n  ✅ test_broken_optimizer_detected PASSED")


if __name__ == "__main__":
    import sys

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        test_normal_resume_passes(tmp_path)
        test_broken_optimizer_detected(tmp_path)

    print("\n🎉 All integration tests passed!")
    sys.exit(0)
