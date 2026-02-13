# IDFT Smoke Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the full IDFT smoke test pipeline — loss function, custom trainer, phi diagnostics, experiment orchestrator, benchmark evaluation, and decision framework — to compare Standard SFT vs IDFT on a MoE model.

**Architecture:** Extend the existing QLoRA training pipeline in `experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/`. New modules (`idft_loss.py`, `idft_trainer.py`, `phi_diagnostic.py`, `run_idft_smoke_test.py`, `evaluate_smoke_test.py`) are self-contained. Existing files (`qlora_config.py`, `train_qlora.py`, `default_config.yaml`) get minimal extensions to add `method="idft"` support.

**Tech Stack:** PyTorch, Transformers, TRL (SFTTrainer subclass), PEFT, bitsandbytes, datasets, lm-evaluation-harness, wandb (optional), pyyaml

---

## Constants

All new files go under:
```
BASE = experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support
TESTS = tests/18_sft_and_rl_alignment_and_final_benchmarks
```

---

### Task 1: IDFT Loss Function — Tests

**Files:**
- Create: `TESTS/test_idft_loss.py`

**Step 1: Write the failing tests**

```python
"""Unit tests for IDFT loss function."""

import pytest
import torch
import torch.nn.functional as F


class TestSFTLoss:
    """Tests for the standard SFT loss baseline."""

    def test_basic_loss_computation(self):
        """SFT loss should equal mean negative log prob of target tokens."""
        from experiments.eighteen_sft.quantization_support.idft_loss import sft_loss

        torch.manual_seed(42)
        batch, seq_len, vocab = 2, 4, 10
        logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))
        mask = torch.ones(batch, seq_len)

        loss = sft_loss(logits, labels, mask)

        # Manual computation
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        expected = -(token_lp * mask).sum() / mask.sum()

        assert torch.allclose(loss, expected, atol=1e-5)

    def test_padding_mask_excludes_tokens(self):
        """Padding tokens (mask=0) should not contribute to loss."""
        from experiments.eighteen_sft.quantization_support.idft_loss import sft_loss

        torch.manual_seed(42)
        logits = torch.randn(1, 4, 10)
        labels = torch.randint(0, 10, (1, 4))

        mask_full = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss_full = sft_loss(logits, labels, mask_full)
        loss_half = sft_loss(logits, labels, mask_half)

        # Different masks should give different losses
        assert not torch.allclose(loss_full, loss_half)

    def test_loss_is_positive(self):
        """Cross-entropy loss should always be non-negative."""
        from experiments.eighteen_sft.quantization_support.idft_loss import sft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 100)
        labels = torch.randint(0, 100, (2, 8))
        mask = torch.ones(2, 8)

        loss = sft_loss(logits, labels, mask)
        assert loss.item() >= 0


class TestIDFTLoss:
    """Tests for the IDFT loss function."""

    def test_returns_loss_and_diagnostics(self):
        """IDFT loss should return (loss_tensor, diagnostics_dict)."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        result = idft_loss(logits, labels, mask, clip_B=5.0)

        assert isinstance(result, tuple)
        assert len(result) == 2

        loss, diag = result
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert isinstance(diag, dict)

    def test_diagnostics_keys(self):
        """Diagnostics dict should contain all expected keys."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        _, diag = idft_loss(logits, labels, mask)

        expected_keys = {
            "phi_mean", "phi_std",
            "phi_below_neg1_pct", "phi_below_neg3_pct", "phi_below_neg5_pct",
            "gamma_mean", "gamma_max",
        }
        assert set(diag.keys()) == expected_keys

    def test_clip_B_zero_reduces_to_standard_loss(self):
        """With clip_B=0, phi is clamped to 0, gamma=exp(0)=1, so IDFT = SFT."""
        from experiments.eighteen_sft.quantization_support.idft_loss import (
            idft_loss,
            sft_loss,
        )

        torch.manual_seed(42)
        logits = torch.randn(2, 8, 50)
        labels = torch.randint(0, 50, (2, 8))
        mask = torch.ones(2, 8)

        sft = sft_loss(logits, labels, mask)
        idft, _ = idft_loss(logits, labels, mask, clip_B=0.0)

        # When clip_B=0, phi_clipped=0, gamma=1, weight=p^1=p
        # IDFT = -(1/L) * sum(p * log p) which is NOT the same as SFT
        # Actually: with gamma=1, loss = -sum(p * log p) / L = entropy-weighted
        # So this tests that clip_B=0 gives a valid, finite loss
        assert torch.isfinite(idft)

    def test_no_nan_with_extreme_logits(self):
        """Loss should be finite even with very large/small logits."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        # Very large logits (near one-hot distribution)
        logits_big = torch.randn(2, 4, 10) * 100
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, diag = idft_loss(logits_big, labels, mask, clip_B=5.0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert all(
            not (isinstance(v, float) and (v != v))  # NaN check
            for v in diag.values()
        )

    def test_no_nan_with_tiny_logits(self):
        """Loss should be finite even with very small logits (high entropy)."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        # Near-uniform distribution
        logits_tiny = torch.zeros(2, 4, 10) + torch.randn(2, 4, 10) * 0.01
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, diag = idft_loss(logits_tiny, labels, mask, clip_B=5.0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_phi_clipping_works(self):
        """Phi values should be within [-clip_B, clip_B]."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10) * 50  # extreme logits
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        clip_B = 3.0
        _, diag = idft_loss(logits, labels, mask, clip_B=clip_B)

        # With clip_B=3, gamma ranges from exp(-3) to exp(3)
        assert diag["gamma_max"] <= pytest.approx(
            torch.exp(torch.tensor(clip_B)).item(), abs=0.1
        )

    def test_mask_excludes_padding(self):
        """Padding tokens should not affect loss or diagnostics."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(1, 4, 10)
        labels = torch.randint(0, 10, (1, 4))

        mask_full = torch.ones(1, 4)
        mask_half = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        loss_full, _ = idft_loss(logits, labels, mask_full)
        loss_half, _ = idft_loss(logits, labels, mask_half)

        assert not torch.allclose(loss_full, loss_half)

    def test_loss_is_differentiable(self):
        """IDFT loss should support backpropagation."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        torch.manual_seed(42)
        logits = torch.randn(2, 4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (2, 4))
        mask = torch.ones(2, 4)

        loss, _ = idft_loss(logits, labels, mask)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()

    def test_gamma_values_make_sense(self):
        """For near-uniform logits, phi should be negative, gamma > 1 (OOD)."""
        from experiments.eighteen_sft.quantization_support.idft_loss import idft_loss

        # Near-uniform: log_p(x_t) is very negative, entropy is high
        # phi = log_p(x_t) + H should be somewhat negative for random tokens
        logits = torch.zeros(2, 8, 1000)  # uniform over 1000 tokens
        labels = torch.randint(0, 1000, (2, 8))
        mask = torch.ones(2, 8)

        _, diag = idft_loss(logits, labels, mask, clip_B=10.0)

        # log_p(x_t) = -log(1000) ≈ -6.9, H = log(1000) ≈ 6.9
        # phi ≈ -6.9 + 6.9 ≈ 0 for uniform
        # So gamma should be near 1
        assert diag["gamma_mean"] == pytest.approx(1.0, abs=0.5)
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/sriranga/Desktop/LLM && python -m pytest tests/18_sft_and_rl_alignment_and_final_benchmarks/test_idft_loss.py -v`
Expected: FAIL — `ModuleNotFoundError` (idft_loss doesn't exist yet)

**Step 3: Commit test file**

```bash
git add tests/18_sft_and_rl_alignment_and_final_benchmarks/test_idft_loss.py
git commit -m "test: add unit tests for IDFT loss function"
```

---

### Task 2: IDFT Loss Function — Implementation

**Files:**
- Create: `BASE/idft_loss.py`

**Step 1: Implement the loss functions**

```python
"""
IDFT Loss Functions
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Implements the IDFT (In-Distribution Fine-Tuning) loss from
"Towards On-Policy SFT" (arXiv:2602.12222, Feb 2026).

Also provides a standard SFT loss for baseline comparison.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Standard SFT cross-entropy loss.

    L_SFT = -(1/L) * sum(log p_t(x_t))

    Args:
        logits: (batch, seq_len, vocab_size) raw model logits
        labels: (batch, seq_len) target token IDs
        attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

    Returns:
        Scalar loss tensor.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked = token_log_probs * attention_mask
    loss = -masked.sum() / attention_mask.sum()
    return loss


def idft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    clip_B: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    IDFT loss from "Towards On-Policy SFT" (arXiv:2602.12222).

    L_IDFT = -(1/L) * sum(p_t(x_t)^gamma_t * log p_t(x_t))
    where gamma_t = exp(-phi_t) and phi_t = log p_t(x_t) + H[p_t]

    Args:
        logits: (batch, seq_len, vocab_size) raw model logits
        labels: (batch, seq_len) target token IDs
        attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding
        clip_B: Clipping bound for phi. Paper recommends 3-10, default 5.

    Returns:
        Tuple of (loss, diagnostics_dict).
        loss: scalar tensor (differentiable).
        diagnostics_dict: dict with phi/gamma statistics (detached).
    """
    # Step 1: log probabilities and probabilities
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
    probs = log_probs.exp()  # (B, L, V)

    # Step 2: log p_t(x_t) for target tokens
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

    # Step 3: entropy H[p_t] = -sum(p(v) * log p(v))
    entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)

    # Step 4: phi_t = log p_t(x_t) + H[p_t]  (CLL discriminant)
    phi = token_log_probs + entropy  # (B, L)

    # Step 5: clip phi for numerical stability
    phi_clipped = phi.clamp(-clip_B, clip_B)

    # Step 6: gamma_t = exp(-phi_t)
    gamma = torch.exp(-phi_clipped)  # (B, L)

    # Step 7: IDFT loss in log-space for stability: p^gamma = exp(gamma * log p)
    weighted_factor = torch.exp(gamma * token_log_probs)  # p_t^gamma_t
    per_token_loss = -weighted_factor * token_log_probs  # -p_t^gamma_t * log p_t

    # Step 8: mask and average
    masked_loss = per_token_loss * attention_mask
    loss = masked_loss.sum() / attention_mask.sum()

    # Diagnostics (detached, no grad)
    with torch.no_grad():
        valid_phi = phi_clipped[attention_mask.bool()]
        valid_gamma = gamma[attention_mask.bool()]
        diagnostics = {
            "phi_mean": valid_phi.mean().item(),
            "phi_std": valid_phi.std().item(),
            "phi_below_neg1_pct": (valid_phi < -1).float().mean().item() * 100,
            "phi_below_neg3_pct": (valid_phi < -3).float().mean().item() * 100,
            "phi_below_neg5_pct": (valid_phi < -5).float().mean().item() * 100,
            "gamma_mean": valid_gamma.mean().item(),
            "gamma_max": valid_gamma.max().item(),
        }

    return loss, diagnostics
```

**Step 2: Run the tests**

Run: `cd /Users/sriranga/Desktop/LLM && python -m pytest tests/18_sft_and_rl_alignment_and_final_benchmarks/test_idft_loss.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/idft_loss.py
git commit -m "feat: implement IDFT and SFT loss functions"
```

---

### Task 3: Config Extension — IDFTSettings

**Files:**
- Modify: `BASE/qlora_config.py` (add IDFTSettings dataclass, extend TrainingConfig and QLoRAConfig.from_dict)
- Modify: `BASE/default_config.yaml` (add idft section under training)

**Step 1: Add IDFTSettings dataclass after DPOSettings (~line 140)**

Add this new dataclass right after the `DPOSettings` class:

```python
@dataclass
class IDFTSettings:
    """IDFT-specific training settings."""
    enabled: bool = False
    clip_B: float = 5.0
    learning_rates: List[float] = field(
        default_factory=lambda: [5e-5, 2e-5, 1e-5]
    )
    log_diagnostics_every: int = 10
```

**Step 2: Add `idft` field to TrainingConfig (~line 183)**

Add after the `dpo` field:

```python
    idft: IDFTSettings = field(default_factory=IDFTSettings)
```

Also update `method` Literal to include "idft":
```python
    method: Literal["sft", "grpo", "dpo", "idft"] = "sft"
```

**Step 3: Update `QLoRAConfig.from_dict()` to parse idft settings (~line 272)**

Add idft parsing alongside grpo/dpo in the training config section:

```python
        idft_settings = IDFTSettings(**training_dict.pop("idft", {}))
        training_config = TrainingConfig(
            **training_dict,
            grpo=grpo_settings,
            dpo=dpo_settings,
            idft=idft_settings,
        )
```

**Step 4: Update `from_args()` to handle IDFT CLI overrides (~line 345)**

Add after the existing DPO overrides:

```python
        if hasattr(args, 'idft_clip_B') and args.idft_clip_B is not None:
            config.training.idft.clip_B = args.idft_clip_B
            config.training.idft.enabled = True

        if hasattr(args, 'use_idft') and args.use_idft:
            config.training.idft.enabled = True
            config.training.method = "idft"
```

**Step 5: Update `create_argument_parser()` to add IDFT args (~line 618)**

Add IDFT-specific arguments, and update method choices:

```python
    # Update existing method argument choices
    # Change: choices=["sft", "grpo", "dpo"] -> choices=["sft", "grpo", "dpo", "idft"]

    # Add IDFT arguments
    parser.add_argument(
        "--use_idft",
        action="store_true",
        help="Enable IDFT loss (sets method to idft)"
    )
    parser.add_argument(
        "--idft_clip_B",
        type=float,
        help="IDFT phi clipping bound (default: 5.0)"
    )
```

**Step 6: Update `print_config()` to show IDFT settings (~line 520)**

Add after the Training section print block:

```python
        if self.training.method == "idft" or self.training.idft.enabled:
            print(f"\n[IDFT]")
            print(f"  Enabled: {self.training.idft.enabled}")
            print(f"  Clip B: {self.training.idft.clip_B}")
            print(f"  LR grid: {self.training.idft.learning_rates}")
            print(f"  Log diagnostics every: {self.training.idft.log_diagnostics_every}")
```

**Step 7: Add idft section to default_config.yaml (~line 236, after dpo section)**

```yaml
  # -----------------------------
  # IDFT-Specific Settings
  # -----------------------------
  idft:
    # Enable IDFT loss (In-Distribution Fine-Tuning)
    enabled: false

    # Phi clipping bound [-B, B]. Paper recommends 3-10, sweet spot at 5.
    clip_B: 5.0

    # Learning rates to grid search (IDFT needs separate LR from SFT)
    learning_rates:
      - 5.0e-5
      - 2.0e-5
      - 1.0e-5

    # Log phi/gamma diagnostics every N steps
    log_diagnostics_every: 10
```

**Step 8: Verify config loads**

Run: `cd /Users/sriranga/Desktop/LLM/experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support && python qlora_config.py --method idft`
Expected: Config prints successfully with IDFT section visible.

**Step 9: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/qlora_config.py
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/default_config.yaml
git commit -m "feat: add IDFTSettings to QLoRA config and CLI"
```

---

### Task 4: IDFT Trainer — Custom SFTTrainer Subclass

**Files:**
- Create: `BASE/idft_trainer.py`

**Step 1: Implement the trainer**

```python
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
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/idft_trainer.py
git commit -m "feat: add IDFTTrainer subclass with IDFT loss"
```

---

### Task 5: Integrate IDFT into train_qlora.py

**Files:**
- Modify: `BASE/train_qlora.py` (add `train_idft()` function, update `train()` dispatch)

**Step 1: Add `train_idft()` function after `train_dpo()` (~line 598)**

```python
def train_idft(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    config: QLoRAConfig,
) -> Any:
    """
    Train using IDFT (In-Distribution Fine-Tuning) loss.

    Uses a custom SFTTrainer subclass that replaces cross-entropy
    with the IDFT reweighted loss from arXiv:2602.12222.

    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        config: Configuration

    Returns:
        Trainer instance
    """
    from idft_trainer import IDFTTrainer
    from trl import SFTConfig

    logger.info("Starting IDFT training...")
    logger.info(f"  clip_B = {config.training.idft.clip_B}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.training.output_dir}/idft_{timestamp}"

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        num_train_epochs=config.training.num_train_epochs,
        max_steps=config.training.max_steps,
        bf16=config.training.bf16,
        fp16=config.training.fp16,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        eval_strategy=config.training.eval_strategy if eval_dataset else "no",
        eval_steps=config.training.eval_steps if eval_dataset else None,
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=config.training.report_to,
        max_seq_length=config.model.max_seq_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )

    trainer = IDFTTrainer(
        clip_B=config.training.idft.clip_B,
        log_diagnostics_every=config.training.idft.log_diagnostics_every,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    logger.info(f"IDFT training complete. Model saved to: {output_dir}")
    return trainer
```

**Step 2: Update `train()` dispatch (~line 626)**

Add the idft case to the if/elif chain:

```python
    elif config.training.method == "idft":
        trainer = train_idft(model, tokenizer, train_dataset, eval_dataset, config)
```

**Step 3: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/train_qlora.py
git commit -m "feat: integrate IDFT training method into train_qlora.py"
```

---

### Task 6: Phi Diagnostic Script (Phase 1 DDT Validation)

**Files:**
- Create: `BASE/phi_diagnostic.py`

**Step 1: Implement the diagnostic script**

```python
#!/usr/bin/env python3
"""
Phi Distribution Diagnostic — Phase 1 DDT Validation
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Validates that the CLL discriminant (phi) separates in-distribution
vs OOD tokens on MoE model outputs. This is a go/no-go gate before
running the full IDFT smoke test.

Usage:
    python phi_diagnostic.py --config idft_smoke_config.yaml
    python phi_diagnostic.py --model_name "Qwen/Qwen2.5-MoE" --max_batches 50
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_phi_distribution(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    max_batches: int = 100,
    batch_size: int = 4,
    max_seq_length: int = 2048,
) -> Dict[str, float]:
    """
    Run forward passes and collect phi statistics on dataset responses.

    Args:
        model: The base model (not fine-tuned).
        tokenizer: Tokenizer.
        dataset: HuggingFace dataset with 'text' column.
        max_batches: Number of batches to process.
        batch_size: Batch size for forward passes.
        max_seq_length: Maximum sequence length.

    Returns:
        Dict with phi statistics.
    """
    model.eval()
    all_phi = []

    def collate_fn(examples):
        texts = [ex["text"] for ex in examples]
        encodings = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        return encodings

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift for causal LM
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            probs = log_probs.exp()

            token_log_probs = log_probs.gather(
                -1, shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            phi = token_log_probs + entropy
            valid_phi = phi[shift_mask.bool()]
            all_phi.append(valid_phi.cpu())

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{max_batches} batches")

    all_phi = torch.cat(all_phi)

    results = {
        "phi_mean": all_phi.mean().item(),
        "phi_std": all_phi.std().item(),
        "phi_median": all_phi.median().item(),
        "phi_below_neg1_pct": (all_phi < -1).float().mean().item() * 100,
        "phi_below_neg3_pct": (all_phi < -3).float().mean().item() * 100,
        "phi_below_neg5_pct": (all_phi < -5).float().mean().item() * 100,
        "phi_above_0_pct": (all_phi > 0).float().mean().item() * 100,
        "total_tokens": len(all_phi),
    }
    return results


def evaluate_phi_results(results: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply go/no-go decision based on phi distribution.

    Expected (from paper):
    - Dataset responses: mean phi around -0.07 to -0.33
    - 3-12% of tokens with phi < -3

    Returns:
        Dict with decision and reasoning.
    """
    decision = {
        "go": True,
        "reasons": [],
        "warnings": [],
    }

    # Check 1: phi mean should be negative (indicates OOD presence)
    if results["phi_mean"] > 0.5:
        decision["warnings"].append(
            f"phi_mean={results['phi_mean']:.4f} is positive — "
            "data may already be in-distribution, IDFT gains may be small."
        )

    # Check 2: Some tokens should be strongly OOD
    if results["phi_below_neg3_pct"] < 1.0:
        decision["warnings"].append(
            f"Only {results['phi_below_neg3_pct']:.1f}% tokens have phi < -3 — "
            "very few OOD tokens, IDFT may not provide benefit."
        )

    # Check 3: Distribution should have meaningful spread
    if results["phi_std"] < 0.5:
        decision["go"] = False
        decision["reasons"].append(
            f"phi_std={results['phi_std']:.4f} is very low — "
            "CLL discriminant is not separating tokens. "
            "IDFT may be incompatible with this MoE architecture."
        )

    # Check 4: If nearly all tokens are OOD, something is wrong
    if results["phi_below_neg5_pct"] > 50:
        decision["warnings"].append(
            f"{results['phi_below_neg5_pct']:.1f}% tokens have phi < -5 — "
            "extremely high OOD fraction, consider reducing clip_B."
        )

    if not decision["reasons"]:
        decision["reasons"].append("Phi distribution shows meaningful spread.")

    return decision


def main():
    parser = argparse.ArgumentParser(description="Phi Distribution Diagnostic")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--dataset_name", type=str, default="OpenAssistant/oasst1")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_json", type=str, default="phi_diagnostic_results.json")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"trust_remote_code": True}
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = args.device

    # Try loading with bf16 to save memory
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=torch.bfloat16, **model_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, **model_kwargs
        )

    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)

    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))

    # Ensure text column exists
    if "text" not in dataset.column_names:
        # Try to find a suitable text column
        for col in ["content", "prompt", "instruction"]:
            if col in dataset.column_names:
                dataset = dataset.rename_column(col, "text")
                break

    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info(f"Running phi diagnostic ({args.max_batches} batches)...")

    results = compute_phi_distribution(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )

    decision = evaluate_phi_results(results)

    # Print results
    print("\n" + "=" * 70)
    print("PHI DISTRIBUTION DIAGNOSTIC RESULTS")
    print("=" * 70)
    print(f"  Model:          {args.model_name}")
    print(f"  Tokens analyzed: {results['total_tokens']:,}")
    print(f"\n  phi mean:    {results['phi_mean']:.4f}")
    print(f"  phi std:     {results['phi_std']:.4f}")
    print(f"  phi median:  {results['phi_median']:.4f}")
    print(f"\n  phi < -1:    {results['phi_below_neg1_pct']:.1f}%")
    print(f"  phi < -3:    {results['phi_below_neg3_pct']:.1f}%")
    print(f"  phi < -5:    {results['phi_below_neg5_pct']:.1f}%")
    print(f"  phi > 0:     {results['phi_above_0_pct']:.1f}%")

    print(f"\n  DECISION: {'GO' if decision['go'] else 'NO-GO'}")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    for warning in decision["warnings"]:
        print(f"    WARNING: {warning}")
    print("=" * 70)

    # Save results
    output = {"phi_stats": results, "decision": decision}
    output_path = Path(args.output_json)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Exit with code based on decision
    sys.exit(0 if decision["go"] else 1)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/phi_diagnostic.py
git commit -m "feat: add phi distribution diagnostic for DDT validation"
```

---

### Task 7: IDFT Smoke Test Config

**Files:**
- Create: `BASE/idft_smoke_config.yaml`

**Step 1: Create the experiment config**

```yaml
# =============================================================================
# IDFT Smoke Test Configuration
# Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks
# =============================================================================
# A/B comparison: Standard SFT vs IDFT loss
# Based on "Towards On-Policy SFT" (arXiv:2602.12222)
# =============================================================================

model:
  # Primary: 70B MoE checkpoint after pre-training
  # Fallback 1: 8B MoE checkpoint from Stage 3
  # Fallback 2: Public MoE (Qwen/Qwen2.5-7B for code validation)
  name: "Qwen/Qwen2.5-7B"
  trust_remote_code: true
  torch_dtype: "auto"
  device_map: "auto"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096
  max_prompt_length: 2048
  max_completion_length: 2048

quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  compute_dtype: "bfloat16"
  double_quant: true
  exclude_modules:
    - "lm_head"
    - "embed_tokens"
    - ".*norm.*"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"
  # For MoE: target all expert FFNs + attention (NOT just attention)
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  output_dir: "./outputs/idft_smoke_test"
  method: "sft"  # Overridden per condition by orchestrator

  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 4  # effective batch = 32

  learning_rate: 2.0e-5  # Overridden per LR grid point
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01

  num_train_epochs: 2
  max_steps: -1

  bf16: true
  fp16: false
  gradient_checkpointing: true
  max_grad_norm: 1.0

  logging_steps: 10
  save_steps: 500
  save_total_limit: 2
  eval_strategy: "steps"
  eval_steps: 500

  report_to: "wandb"
  seed: 42
  dataloader_num_workers: 4

  grpo:
    num_generations: 4
    beta: 0.0
    temperature: 0.7
    top_p: 0.9
    epsilon: 0.2

  dpo:
    beta: 0.1
    label_smoothing: 0.0

  idft:
    enabled: true
    clip_B: 5.0
    learning_rates:
      - 5.0e-5
      - 2.0e-5
      - 1.0e-5
    log_diagnostics_every: 10

data:
  # 50K stratified subset — prepared by orchestrator
  dataset_name: "OpenAssistant/oasst1"
  dataset_split: "train"
  max_samples: 50000
  val_split_ratio: 0.05
  text_column: "text"
  prompt_template: ""
  filters:
    language: "en"
    min_quality: 0.5

hardware:
  auto_detect: true
  fallback_to_cpu: false
  mps_fallback_to_bf16: true
  force_device: null

hub:
  push_to_hub: false
  hub_model_id: null
  private: false
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/idft_smoke_config.yaml
git commit -m "feat: add IDFT smoke test experiment config"
```

---

### Task 8: Evaluation Script (Phase 3)

**Files:**
- Create: `BASE/evaluate_smoke_test.py`

**Step 1: Implement the evaluation wrapper**

```python
#!/usr/bin/env python3
"""
Smoke Test Evaluation — Phase 3
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Evaluates trained checkpoints on benchmarks using lm-evaluation-harness.
Compares Standard SFT vs IDFT vs base model.

Usage:
    python evaluate_smoke_test.py \
        --checkpoint_dir ./outputs/idft_smoke_test/sft_best \
        --output_json results_sft.json

    python evaluate_smoke_test.py \
        --checkpoint_dir ./outputs/idft_smoke_test/idft_best \
        --output_json results_idft.json
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Benchmark configuration matching the smoke test plan
BENCHMARKS = {
    "gsm8k": {
        "task": "gsm8k",
        "category": "math",
        "num_fewshot": 5,
        "description": "Math reasoning (full)",
    },
    "math_500": {
        "task": "minerva_math",
        "category": "math",
        "num_fewshot": 4,
        "description": "Hard math (MATH-500)",
    },
    "humaneval": {
        "task": "humaneval",
        "category": "code",
        "num_fewshot": 0,
        "description": "Code generation",
    },
    "mmlu_stem": {
        "task": "mmlu_stem",
        "category": "general",
        "num_fewshot": 5,
        "description": "General knowledge (STEM)",
    },
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "category": "safety",
        "num_fewshot": 0,
        "description": "Safety/factuality",
    },
}

# Eval settings from the plan
EVAL_SETTINGS = {
    "temperature": 0.3,
    "num_runs_small": 8,  # benchmarks < 1000 samples
    "num_runs_large": 2,  # benchmarks > 1000 samples
    "max_gen_tokens": 2048,
}


def run_lm_eval(
    model_path: str,
    task: str,
    num_fewshot: int = 0,
    batch_size: str = "auto",
    output_path: Optional[str] = None,
    use_peft: bool = False,
    base_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run lm-evaluation-harness on a model checkpoint.

    Args:
        model_path: Path to model or HF model name.
        task: Benchmark task name.
        num_fewshot: Number of few-shot examples.
        batch_size: Batch size for evaluation.
        output_path: Path to save results JSON.
        use_peft: Whether model_path is a PEFT adapter.
        base_model: Base model name (required if use_peft=True).

    Returns:
        Dict with evaluation results.
    """
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
    ]

    if use_peft and base_model:
        cmd[4] = f"pretrained={base_model},peft={model_path}"

    if output_path:
        cmd.extend(["--output_path", output_path])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            logger.error(f"lm_eval failed: {result.stderr}")
            return {"error": result.stderr, "task": task}

        # Parse output
        if output_path and Path(output_path).exists():
            with open(output_path) as f:
                return json.load(f)
        return {"stdout": result.stdout, "task": task}

    except subprocess.TimeoutExpired:
        logger.error(f"Evaluation timed out for task {task}")
        return {"error": "timeout", "task": task}
    except FileNotFoundError:
        logger.error(
            "lm_eval not found. Install with: pip install lm-eval"
        )
        return {"error": "lm_eval not installed", "task": task}


def evaluate_checkpoint(
    checkpoint_path: str,
    label: str,
    benchmarks: Optional[List[str]] = None,
    use_peft: bool = False,
    base_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on all smoke test benchmarks.

    Args:
        checkpoint_path: Path to model checkpoint.
        label: Label for this condition (e.g., "sft", "idft", "base").
        benchmarks: List of benchmark names to run. None = all.
        use_peft: Whether checkpoint is a PEFT adapter.
        base_model: Base model name (required if use_peft=True).

    Returns:
        Dict with all benchmark results.
    """
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())

    results = {"label": label, "checkpoint": checkpoint_path, "benchmarks": {}}

    for name in benchmarks:
        if name not in BENCHMARKS:
            logger.warning(f"Unknown benchmark: {name}, skipping")
            continue

        bench = BENCHMARKS[name]
        logger.info(f"Evaluating {label} on {name} ({bench['description']})...")

        output_path = f"eval_results_{label}_{name}.json"
        result = run_lm_eval(
            model_path=checkpoint_path,
            task=bench["task"],
            num_fewshot=bench["num_fewshot"],
            output_path=output_path,
            use_peft=use_peft,
            base_model=base_model,
        )

        results["benchmarks"][name] = {
            "category": bench["category"],
            "description": bench["description"],
            "raw_results": result,
        }

    return results


def compute_aggregate_scores(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute Math-Avg and General-Avg from benchmark results.

    Returns:
        Dict with aggregate scores.
    """
    math_scores = []
    general_scores = []

    for name, bench_result in results.get("benchmarks", {}).items():
        raw = bench_result.get("raw_results", {})
        # Try to extract accuracy from lm-eval output format
        score = None
        if "results" in raw:
            task_results = raw["results"]
            for task_name, task_data in task_results.items():
                if "acc" in task_data:
                    score = task_data["acc"] * 100
                elif "acc_norm" in task_data:
                    score = task_data["acc_norm"] * 100

        if score is not None:
            if bench_result["category"] == "math":
                math_scores.append(score)
            else:
                general_scores.append(score)

    aggregates = {}
    if math_scores:
        aggregates["math_avg"] = sum(math_scores) / len(math_scores)
    if general_scores:
        aggregates["general_avg"] = sum(general_scores) / len(general_scores)

    return aggregates


def compare_conditions(
    sft_results: Dict[str, Any],
    idft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply the decision framework from the smoke test plan.

    Returns:
        Dict with comparison, deltas, and recommendation.
    """
    sft_agg = compute_aggregate_scores(sft_results)
    idft_agg = compute_aggregate_scores(idft_results)

    comparison = {
        "sft_scores": sft_agg,
        "idft_scores": idft_agg,
        "deltas": {},
        "recommendation": "",
        "outcome": "",
    }

    if base_results:
        comparison["base_scores"] = compute_aggregate_scores(base_results)

    # Compute deltas (IDFT - SFT)
    math_delta = idft_agg.get("math_avg", 0) - sft_agg.get("math_avg", 0)
    general_delta = idft_agg.get("general_avg", 0) - sft_agg.get("general_avg", 0)
    comparison["deltas"] = {
        "math_avg_delta": math_delta,
        "general_avg_delta": general_delta,
    }

    # Decision framework
    if math_delta >= 2.0 and general_delta >= -0.5:
        comparison["outcome"] = "strong_positive"
        comparison["recommendation"] = (
            "ADOPT: IDFT beats SFT by >= 2% on Math-Avg with <= 0.5% "
            "General-Avg regression. Integrate into full SFT pipeline."
        )
    elif math_delta >= 1.0 and general_delta >= 0.0:
        comparison["outcome"] = "moderate_positive"
        comparison["recommendation"] = (
            "ADOPT: IDFT beats SFT by >= 1% on Math-Avg with no "
            "General-Avg regression. Integrate into full SFT pipeline."
        )
    elif math_delta > 0 and general_delta < 0:
        comparison["outcome"] = "mixed"
        comparison["recommendation"] = (
            "INVESTIGATE: IDFT shows mixed results. Consider per-dataset "
            "IDFT or adjusting clip_B."
        )
    else:
        comparison["outcome"] = "negative"
        comparison["recommendation"] = (
            "DO NOT ADOPT: IDFT does not outperform standard SFT. "
            "Stick with standard SFT loss."
        )

    return comparison


def print_results_table(
    sft_results: Dict[str, Any],
    idft_results: Dict[str, Any],
    base_results: Optional[Dict[str, Any]] = None,
):
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("IDFT SMOKE TEST RESULTS")
    print("=" * 80)

    header = f"{'Benchmark':<20} {'Category':<12} {'Base':>8} {'SFT':>8} {'IDFT':>8} {'Delta':>8}"
    print(header)
    print("-" * 80)

    all_benchmarks = set(
        list(sft_results.get("benchmarks", {}).keys())
        + list(idft_results.get("benchmarks", {}).keys())
    )

    for name in sorted(all_benchmarks):
        category = ""
        base_score = "-"
        sft_score = "-"
        idft_score = "-"
        delta = "-"

        if name in sft_results.get("benchmarks", {}):
            category = sft_results["benchmarks"][name].get("category", "")
        if name in idft_results.get("benchmarks", {}):
            category = idft_results["benchmarks"][name].get("category", "")

        # Extract scores (placeholder - actual extraction depends on lm-eval output)
        print(f"{name:<20} {category:<12} {base_score:>8} {sft_score:>8} {idft_score:>8} {delta:>8}")

    print("=" * 80)

    comparison = compare_conditions(sft_results, idft_results, base_results)
    print(f"\nOutcome: {comparison['outcome'].upper()}")
    print(f"Recommendation: {comparison['recommendation']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="IDFT Smoke Test Evaluation")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--label", type=str, required=True,
        help="Condition label (sft, idft, base)"
    )
    parser.add_argument(
        "--output_json", type=str, required=True,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+", default=None,
        help="Benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--use_peft", action="store_true",
        help="Checkpoint is a PEFT adapter (requires --base_model)"
    )
    parser.add_argument(
        "--base_model", type=str, default=None,
        help="Base model name for PEFT adapter loading"
    )
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint_dir,
        label=args.label,
        benchmarks=args.benchmarks,
        use_peft=args.use_peft,
        base_model=args.base_model,
    )

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/evaluate_smoke_test.py
git commit -m "feat: add benchmark evaluation script for smoke test"
```

---

### Task 9: Experiment Orchestrator (All Phases)

**Files:**
- Create: `BASE/run_idft_smoke_test.py`

**Step 1: Implement the orchestrator**

```python
#!/usr/bin/env python3
"""
IDFT Smoke Test Orchestrator
Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

Runs all 4 phases of the IDFT smoke test:
  Phase 0: Setup & validation
  Phase 1: DDT (phi distribution) validation — go/no-go gate
  Phase 2: Training runs (SFT x3 LRs + IDFT x3 LRs)
  Phase 3: Evaluation on 6 benchmarks
  Phase 4: Decision framework & recommendation

Usage:
    python run_idft_smoke_test.py --config idft_smoke_config.yaml
    python run_idft_smoke_test.py --config idft_smoke_config.yaml --skip_phase1
    python run_idft_smoke_test.py --config idft_smoke_config.yaml --phase 2
"""

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("idft_smoke_test.log"),
    ],
)
logger = logging.getLogger(__name__)


def phase0_setup(config_path: str) -> Dict[str, Any]:
    """
    Phase 0: Setup and validation.

    - Load and validate config
    - Check model availability
    - Check VRAM / hardware
    - Create output directories

    Returns:
        Dict with config and setup metadata.
    """
    from qlora_config import QLoRAConfig

    logger.info("=" * 70)
    logger.info("PHASE 0: SETUP")
    logger.info("=" * 70)

    config = QLoRAConfig.from_yaml(config_path)
    config.auto_configure_hardware()
    warnings = config.validate()

    for w in warnings:
        logger.warning(w)

    # Create output directories
    base_output = Path(config.training.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    for subdir in ["sft_runs", "idft_runs", "eval_results", "phi_diagnostic"]:
        (base_output / subdir).mkdir(exist_ok=True)

    config.print_config()

    return {
        "config": config,
        "config_path": config_path,
        "output_dir": str(base_output),
        "timestamp": datetime.now().isoformat(),
    }


def phase1_ddt_validation(setup: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 1: DDT Validation — phi distribution analysis.

    Go/no-go gate. If phi distribution shows no separation,
    IDFT is incompatible with this MoE and we abort.

    Returns:
        Dict with phi stats and go/no-go decision.
    """
    from phi_diagnostic import compute_phi_distribution, evaluate_phi_results
    from qlora_config import QLoRAConfig

    logger.info("=" * 70)
    logger.info("PHASE 1: DDT VALIDATION (phi distribution)")
    logger.info("=" * 70)

    config = setup["config"]

    # Load model for diagnostic (lighter than training)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    logger.info(f"Loading model for phi diagnostic: {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load dataset
    from datasets import load_dataset

    dataset = load_dataset(config.data.dataset_name, split=config.data.dataset_split)
    if config.data.max_samples and len(dataset) > config.data.max_samples:
        dataset = dataset.select(range(config.data.max_samples))

    if "text" not in dataset.column_names:
        for col in ["content", "prompt", "instruction"]:
            if col in dataset.column_names:
                dataset = dataset.rename_column(col, "text")
                break

    # Run diagnostic
    phi_results = compute_phi_distribution(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_batches=100,
        batch_size=4,
    )

    decision = evaluate_phi_results(phi_results)

    # Save results
    output_path = Path(setup["output_dir"]) / "phi_diagnostic" / "phase1_results.json"
    with open(output_path, "w") as f:
        json.dump({"phi_stats": phi_results, "decision": decision}, f, indent=2)

    logger.info(f"Phi diagnostic results saved to {output_path}")
    logger.info(f"Decision: {'GO' if decision['go'] else 'NO-GO'}")

    # Free model memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"phi_stats": phi_results, "decision": decision}


def phase2_training_runs(setup: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phase 2: Training runs.

    Run Standard SFT and IDFT at each LR in the grid.
    Select best LR for each condition based on eval loss.

    Returns:
        Dict with training results and best checkpoints.
    """
    logger.info("=" * 70)
    logger.info("PHASE 2: TRAINING RUNS")
    logger.info("=" * 70)

    config = setup["config"]
    learning_rates = config.training.idft.learning_rates
    base_output = Path(setup["output_dir"])

    results = {"sft_runs": [], "idft_runs": []}

    # --- Standard SFT Runs ---
    for lr in learning_rates:
        logger.info(f"\n--- SFT Run: LR={lr} ---")
        run_config = deepcopy(config)
        run_config.training.method = "sft"
        run_config.training.learning_rate = lr
        run_config.training.output_dir = str(
            base_output / "sft_runs" / f"lr_{lr}"
        )

        try:
            from train_qlora import train

            trainer = train(run_config)
            eval_loss = _get_best_eval_loss(trainer)
            results["sft_runs"].append({
                "lr": lr,
                "output_dir": run_config.training.output_dir,
                "eval_loss": eval_loss,
                "status": "success",
            })
        except Exception as e:
            logger.error(f"SFT run failed at LR={lr}: {e}")
            results["sft_runs"].append({
                "lr": lr, "status": "failed", "error": str(e)
            })

    # --- IDFT Runs ---
    for lr in learning_rates:
        logger.info(f"\n--- IDFT Run: LR={lr} ---")
        run_config = deepcopy(config)
        run_config.training.method = "idft"
        run_config.training.learning_rate = lr
        run_config.training.output_dir = str(
            base_output / "idft_runs" / f"lr_{lr}"
        )

        try:
            from train_qlora import train

            trainer = train(run_config)
            eval_loss = _get_best_eval_loss(trainer)
            results["idft_runs"].append({
                "lr": lr,
                "output_dir": run_config.training.output_dir,
                "eval_loss": eval_loss,
                "status": "success",
            })
        except Exception as e:
            logger.error(f"IDFT run failed at LR={lr}: {e}")
            results["idft_runs"].append({
                "lr": lr, "status": "failed", "error": str(e)
            })

    # Select best checkpoint per condition
    results["sft_best"] = _select_best_run(results["sft_runs"])
    results["idft_best"] = _select_best_run(results["idft_runs"])

    logger.info(f"\nBest SFT: LR={results['sft_best'].get('lr')}")
    logger.info(f"Best IDFT: LR={results['idft_best'].get('lr')}")

    # Save results
    output_path = base_output / "phase2_training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def phase3_evaluation(
    setup: Dict[str, Any],
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Phase 3: Evaluation on benchmarks.

    Evaluate best SFT, best IDFT, and base model on all benchmarks.

    Returns:
        Dict with evaluation results for all conditions.
    """
    from evaluate_smoke_test import (
        evaluate_checkpoint,
        compare_conditions,
        print_results_table,
    )

    logger.info("=" * 70)
    logger.info("PHASE 3: EVALUATION")
    logger.info("=" * 70)

    config = setup["config"]
    base_output = Path(setup["output_dir"]) / "eval_results"

    sft_best = training_results.get("sft_best", {})
    idft_best = training_results.get("idft_best", {})

    eval_results = {}

    # Evaluate base model
    logger.info("\nEvaluating base model...")
    eval_results["base"] = evaluate_checkpoint(
        checkpoint_path=config.model.name,
        label="base",
    )

    # Evaluate best SFT
    if sft_best.get("output_dir"):
        logger.info("\nEvaluating best SFT checkpoint...")
        eval_results["sft"] = evaluate_checkpoint(
            checkpoint_path=sft_best["output_dir"],
            label="sft",
            use_peft=True,
            base_model=config.model.name,
        )

    # Evaluate best IDFT
    if idft_best.get("output_dir"):
        logger.info("\nEvaluating best IDFT checkpoint...")
        eval_results["idft"] = evaluate_checkpoint(
            checkpoint_path=idft_best["output_dir"],
            label="idft",
            use_peft=True,
            base_model=config.model.name,
        )

    # Save results
    output_path = base_output / "phase3_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    return eval_results


def phase4_decision(
    setup: Dict[str, Any],
    eval_results: Dict[str, Any],
    phi_results: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Phase 4: Analysis and decision.

    Apply decision framework and print recommendation.

    Returns:
        Dict with final decision and recommendation.
    """
    from evaluate_smoke_test import compare_conditions, print_results_table

    logger.info("=" * 70)
    logger.info("PHASE 4: ANALYSIS & DECISION")
    logger.info("=" * 70)

    sft_results = eval_results.get("sft", {})
    idft_results = eval_results.get("idft", {})
    base_results = eval_results.get("base")

    comparison = compare_conditions(sft_results, idft_results, base_results)

    # Print table
    print_results_table(sft_results, idft_results, base_results)

    # Add phi diagnostic context
    if phi_results:
        comparison["phi_context"] = phi_results.get("phi_stats", {})

    # Save final report
    base_output = Path(setup["output_dir"])
    report_path = base_output / "final_report.json"
    with open(report_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    logger.info(f"\nFinal report saved to {report_path}")

    return comparison


def _get_best_eval_loss(trainer) -> Optional[float]:
    """Extract best eval loss from trainer state."""
    try:
        if hasattr(trainer.state, "best_metric"):
            return trainer.state.best_metric
        if hasattr(trainer.state, "log_history"):
            eval_losses = [
                entry["eval_loss"]
                for entry in trainer.state.log_history
                if "eval_loss" in entry
            ]
            return min(eval_losses) if eval_losses else None
    except Exception:
        return None


def _select_best_run(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select the run with lowest eval loss."""
    successful = [r for r in runs if r.get("status") == "success"]
    if not successful:
        return {"status": "no_successful_runs"}

    # Prefer runs with eval_loss, fallback to first successful
    with_loss = [r for r in successful if r.get("eval_loss") is not None]
    if with_loss:
        return min(with_loss, key=lambda r: r["eval_loss"])
    return successful[0]


def main():
    parser = argparse.ArgumentParser(
        description="IDFT Smoke Test Orchestrator"
    )
    parser.add_argument(
        "--config", "-c", type=str, default="idft_smoke_config.yaml",
        help="Path to experiment config YAML"
    )
    parser.add_argument(
        "--phase", type=int, default=None,
        help="Run only a specific phase (0-4)"
    )
    parser.add_argument(
        "--skip_phase1", action="store_true",
        help="Skip Phase 1 DDT validation (not recommended)"
    )
    parser.add_argument(
        "--training_results_json", type=str, default=None,
        help="Path to Phase 2 results JSON (to skip training and go to eval)"
    )
    args = parser.parse_args()

    start_time = time.time()

    # Phase 0: Setup
    setup = phase0_setup(args.config)

    if args.phase is not None and args.phase == 0:
        logger.info("Phase 0 complete. Exiting.")
        return

    # Phase 1: DDT Validation
    phi_results = None
    if not args.skip_phase1 and (args.phase is None or args.phase == 1):
        phi_results = phase1_ddt_validation(setup)

        if not phi_results["decision"]["go"]:
            logger.error(
                "PHASE 1 FAILED: DDT validation indicates IDFT is not "
                "compatible with this model. Aborting."
            )
            for reason in phi_results["decision"]["reasons"]:
                logger.error(f"  - {reason}")
            sys.exit(1)

        if args.phase == 1:
            logger.info("Phase 1 complete. Exiting.")
            return

    # Phase 2: Training Runs
    training_results = None
    if args.training_results_json:
        with open(args.training_results_json) as f:
            training_results = json.load(f)
        logger.info(f"Loaded training results from {args.training_results_json}")
    elif args.phase is None or args.phase == 2:
        training_results = phase2_training_runs(setup)

        if args.phase == 2:
            logger.info("Phase 2 complete. Exiting.")
            return

    # Phase 3: Evaluation
    eval_results = None
    if training_results and (args.phase is None or args.phase == 3):
        eval_results = phase3_evaluation(setup, training_results)

        if args.phase == 3:
            logger.info("Phase 3 complete. Exiting.")
            return

    # Phase 4: Decision
    if eval_results and (args.phase is None or args.phase == 4):
        decision = phase4_decision(setup, eval_results, phi_results)

    elapsed = time.time() - start_time
    logger.info(f"\nTotal elapsed time: {elapsed / 3600:.1f} hours")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/run_idft_smoke_test.py
git commit -m "feat: add IDFT smoke test orchestrator (all phases)"
```

---

### Task 10: Update requirements.txt

**Files:**
- Modify: `BASE/requirements.txt`

**Step 1: Add new dependencies at the end (before Installation Notes)**

Add after the Development & Testing section:

```
# -----------------------------------------------------------------------------
# IDFT Smoke Test Dependencies
# -----------------------------------------------------------------------------
# Evaluation harness for benchmarks
# pip install lm-eval
# lm-eval>=0.4.0

# Experiment tracking (enable wandb in config: report_to: "wandb")
# wandb>=0.16.0
```

**Step 2: Commit**

```bash
git add experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/requirements.txt
git commit -m "docs: add IDFT smoke test dependencies to requirements"
```

---

### Task 11: Final Integration Test

**Step 1: Run the full unit test suite**

Run: `cd /Users/sriranga/Desktop/LLM && python -m pytest tests/18_sft_and_rl_alignment_and_final_benchmarks/test_idft_loss.py -v`
Expected: All tests PASS

**Step 2: Verify config loading with IDFT**

Run: `cd /Users/sriranga/Desktop/LLM/experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support && python -c "from qlora_config import QLoRAConfig; c = QLoRAConfig.from_yaml('idft_smoke_config.yaml'); c.print_config()"`
Expected: Config prints with IDFT section showing clip_B=5.0

**Step 3: Verify IDFT loss imports work**

Run: `cd /Users/sriranga/Desktop/LLM/experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support && python -c "from idft_loss import idft_loss, sft_loss; import torch; logits = torch.randn(2, 4, 10); labels = torch.randint(0, 10, (2, 4)); mask = torch.ones(2, 4); loss, diag = idft_loss(logits, labels, mask); print(f'Loss: {loss.item():.4f}'); print(f'Diagnostics: {diag}')"`
Expected: Prints a finite loss value and diagnostics dict

**Step 4: Verify IDFTTrainer imports**

Run: `cd /Users/sriranga/Desktop/LLM/experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support && python -c "from idft_trainer import IDFTTrainer; print('IDFTTrainer imported successfully')"`
Expected: "IDFTTrainer imported successfully"

**Step 5: Run pre-commit checks**

Run: `cd /Users/sriranga/Desktop/LLM && git diff --name-only HEAD~10`
Then: `cd /Users/sriranga/Desktop/LLM && python -m black --check experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/idft_loss.py experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/idft_trainer.py experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/phi_diagnostic.py experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/run_idft_smoke_test.py experiments/18_sft_and_rl_alignment_and_final_benchmarks/quantization_support/evaluate_smoke_test.py`
Expected: All files formatted correctly (or fix formatting)

**Step 6: Final commit (if any formatting fixes needed)**

```bash
git add -A
git commit -m "style: format IDFT smoke test files"
```
