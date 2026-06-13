# IDFT Smoke Test Design — Team 18

**Date:** 2026-02-13
**Status:** Approved
**Branch:** `p18/feat/qlora-quantization-approach-333`

## Goal

Validate whether IDFT (In-Distribution Fine-Tuning) from "Towards On-Policy SFT" (arXiv:2602.12222) improves SFT quality on a 70B MoE LLM (~7B active params). This is an A/B comparison: Standard SFT vs IDFT, same data, same model, independent LR search.

## Architecture

Extend the existing QLoRA training pipeline with IDFT support. New files are self-contained modules; existing files get minimal extensions.

### File Structure

```
experiments/18_.../quantization_support/
├── idft_loss.py              # IDFT + standard SFT loss functions
├── idft_trainer.py           # SFTTrainer subclass with IDFT loss
├── phi_diagnostic.py         # Phase 1: DDT validation script
├── run_idft_smoke_test.py    # Orchestrator for all phases
├── evaluate_smoke_test.py    # Phase 3: benchmark evaluation
├── idft_smoke_config.yaml    # Experiment-specific config
├── qlora_config.py           # Extended with IDFTSettings
└── train_qlora.py            # Extended with method="idft"

tests/18_.../
└── test_idft_loss.py         # Unit tests for loss correctness
```

## Component Design

### 1. `idft_loss.py` — Loss Functions

Two functions:

- `sft_loss(logits, labels, attention_mask) -> loss` — Standard cross-entropy baseline
- `idft_loss(logits, labels, attention_mask, clip_B=5.0) -> (loss, diagnostics)` — IDFT loss

The IDFT loss computes:
- `phi_t = log p_t(x_t) + H[p_t]` (CLL discriminant)
- `gamma_t = exp(-clip(phi_t, -B, B))` (modulation coefficient)
- `L_IDFT = -(1/L) * sum(p_t^gamma_t * log p_t)` (reweighted loss)

All exponentiation done in log-space for numerical stability: `p^gamma = exp(gamma * log_p)`.

Diagnostics dict includes: `phi_mean`, `phi_std`, `phi_below_neg1_pct`, `phi_below_neg3_pct`, `phi_below_neg5_pct`, `gamma_mean`, `gamma_max`.

### 2. `idft_trainer.py` — Custom Trainer

Subclasses TRL's `SFTTrainer`. Overrides `compute_loss()` to:
- Call `idft_loss()` instead of standard CE
- Log phi diagnostics to wandb/console every N steps
- Return scalar loss for backprop

### 3. `qlora_config.py` — Config Extension

New dataclass:
```python
@dataclass
class IDFTSettings:
    enabled: bool = False
    clip_B: float = 5.0
    learning_rates: List[float] = field(default_factory=lambda: [5e-5, 2e-5, 1e-5])
    log_diagnostics_every: int = 10
```

Added to `TrainingConfig` alongside `grpo` and `dpo`. Training method "idft" routes to IDFT trainer.

### 4. `phi_diagnostic.py` — Phase 1 DDT Validation

Standalone script. Loads the base model, runs forward passes over 100 batches from the dataset, computes phi distribution for dataset tokens. Reports:
- Mean, std of phi
- % tokens with phi < -1, -3, -5
- Go/no-go decision based on whether distribution shows OOD separation

If phi shows no separation (mean near 0, low std), prints a warning that IDFT may not help on MoE.

### 5. `run_idft_smoke_test.py` — Experiment Orchestrator

Runs all 4 phases sequentially:

**Phase 0 (Setup):** Validates config, checks model availability, prepares data subset.

**Phase 1 (DDT Validation):** Calls `phi_diagnostic.py` logic. Aborts if no phi separation.

**Phase 2 (Training Runs):** For each condition (SFT, IDFT) x each LR in grid: launches training, saves best checkpoint per condition based on validation loss.

**Phase 3 (Evaluation):** Runs lm-evaluation-harness on both best checkpoints + base model across 6 benchmarks (GSM8K, MATH-500, HumanEval, MMLU-STEM, TruthfulQA, IndicGLUE subset).

**Phase 4 (Analysis):** Computes Math-Avg, General-Avg, applies decision framework, prints recommendation.

### 6. `evaluate_smoke_test.py` — Benchmark Evaluation

Wrapper around `lm-evaluation-harness`. Runs benchmarks with settings from the plan (temp=0.3, multiple runs for small benchmarks). Outputs results as JSON for comparison.

### 7. `test_idft_loss.py` — Unit Tests

- Test that `idft_loss` with `clip_B=0` reduces to standard SFT (gamma=1 for all tokens)
- Test phi computation is correct for known logits
- Test clipping works (phi values outside [-B, B] are clamped)
- Test numerical stability (no NaN/Inf for extreme inputs)
- Test attention mask correctly excludes padding
- Test diagnostics dict contains expected keys

## Data Flow

```
Base Model → phi_diagnostic.py (Phase 1: go/no-go)
                    ↓ (if go)
    ┌───────────────┴───────────────┐
    │                               │
Standard SFT (3 LRs)         IDFT (3 LRs)
    │                               │
    └──── Best of each ─────────────┘
                    ↓
         evaluate_smoke_test.py
                    ↓
         Decision Framework (Phase 4)
```

## Config: `idft_smoke_config.yaml`

Extends `default_config.yaml` with:
- Model: 70B MoE checkpoint (or 8B fallback, or Qwen2.5-MoE public fallback)
- Data: 50K stratified subset
- Training: QLoRA, rank=64, alpha=128, frozen router, 2 epochs, max_seq=4096
- IDFT: clip_B=5.0, LR grid [5e-5, 2e-5, 1e-5]
- Eval: 6 benchmarks, temp=0.3

## Success Criteria (from plan)

| Outcome | Criteria | Decision |
|---------|----------|----------|
| Strong positive | IDFT >= +2% Math-Avg, <= 0.5% General-Avg regression | Adopt |
| Moderate positive | IDFT >= +1% Math-Avg, no General-Avg regression | Adopt |
| Mixed | Wins some, loses some | Investigate further |
| Negative | IDFT <= SFT or >1% General-Avg regression | Do not adopt |
| DDT fails | No phi separation | Incompatible with MoE |
