# MMLU-Pro Evaluation Results

## Model
**`google/gemma-3-1b-pt`** (Pre-trained base checkpoint)

## Run Command

```bash
olmes --model google/gemma-3-1b-pt --task mmlu_pro:mc::none --output-dir gemma-mmlu-pro
```

---

## Overview

Evaluation of the model on the [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) benchmark — a challenging multiple-choice dataset spanning 14 academic disciplines with 10-option questions designed to emphasize reasoning over recall. All tasks were evaluated using **5-shot** prompting on the **test** split.

- **Overall Score (micro):** `0.1151` (11.51%)
- **Overall Score (macro):** `0.1166` (11.66%)
- **Total Instances:** 12,032
- **Primary Metric:** Raw Accuracy (micro-averaged)

> ⚠️ These scores are near random-chance for a 10-option multiple-choice task (random baseline ~10%), which is **expected behavior for the pre-trained base model** — it has not undergone instruction-tuning or task-specific alignment.

---

## Per-Domain Results

| Domain | Accuracy | Instances |
|---|---|---|
| 🏆 Philosophy | **14.43%** | 499 |
| 🥈 Economics | **13.51%** | 844 |
| 🥉 Other | **12.66%** | 924 |
| Computer Science | 11.95% | 410 |
| Psychology | 11.90% | 798 |
| Physics | 11.70% | 1,299 |
| Math | 11.55% | 1,351 |
| History | 11.29% | 381 |
| Biology | 11.16% | 717 |
| Business | 11.15% | 789 |
| Engineering | 11.04% | 969 |
| Health | 11.00% | 818 |
| Law | 10.63% | 1,101 |
| ⬇️ Chemistry | **9.28%** | 1,132 |

---

## Comparison with Official Gemma 3 Benchmarks

Google's Gemma 3 technical report reports MMLU-Pro only for **instruction-tuned (IT) models**, not for the pre-trained (PT) base. The table below places this run in context against published and available reference points:

| Source | Model | Variant | MMLU-Pro |
|---|---|---|---|
| **This run (OLMo eval harness)** | `gemma-3-1b` | **PT (base)** | **11.51%** |
| Gemma 3 Technical Report (Table 6) | `gemma-3-4b` | IT | 43.6% |
| Gemma 3 Technical Report (Table 6) | `gemma-3-27b` | IT | 67.5% |
| Gemini 1.5 Pro (reference) | Gemini 1.5 Pro | IT | 75.8% |
| Random baseline (10 options) | — | — | ~10.0% |

> ⚠️ **Important caveat:** Google does not publish an MMLU-Pro score for `gemma-3-1b-pt` in the technical report. The 1B is the smallest model in the family and was primarily intended as an on-device, efficiency-focused checkpoint — Google's published MMLU-Pro numbers start at the 4B-IT level.

---

## Validation & Analysis

### ✅ Score is consistent with a raw PT checkpoint
An overall accuracy of **11.51%** for a 1B pre-trained base model on a 10-option MCQ benchmark is entirely expected. MMLU-Pro is specifically designed to require multi-step reasoning beyond surface-level recall — this is extremely difficult for a base model that has seen no instruction-following data. The result is just slightly above the random-chance floor of ~10%.

### ⚠️ PT vs IT gap is large — and intentional
The gap between this PT score (~11.5%) and the 4B-IT score (~43.6%) reflects the enormous impact of instruction-tuning and RLHF alignment, not just parameter count. Even `gemma-3-1b-it` would be expected to score meaningfully higher than this PT baseline.  If the goal is to compare against Google's published benchmarks, the evaluation should be re-run on `gemma-3-1b-it`.

### ⚠️ 1B is absent from Google's MMLU-Pro reporting
Google deliberately omits the 1B model from its MMLU-Pro tables in the technical report, likely because the base model's near-random performance isn't informative for production comparisons. This makes direct validation against official numbers impossible — the PT score here serves more as a **pre-training baseline** than a deployed-model benchmark.

### 📌 Domain variance is narrow — a sign of no signal
All 14 domains cluster tightly between 9.3% and 14.4%, a range of just ~5 points. This near-uniform distribution across very different subjects (Chemistry vs. Philosophy vs. Law) is characteristic of a model responding with minimal task understanding — essentially probability mass spread across 10 choices without domain-specific knowledge driving differentiation.

### 📌 Chemistry scores below random (9.28%)
Chemistry is the only domain to fall below the 10% random baseline. This is likely a statistical artifact of the small relative score differences at near-chance levels rather than the model being specifically bad at chemistry.

### 📌 Eval harness note
This run uses the OLMo evaluation harness (`olmo3:heldout` regime is also recorded alongside `mc::none`). Google's internal MMLU-Pro evaluations use their own harness with potentially different prompt formats and shot configurations, making scores non-directly comparable even if the model were the same variant.

---

## Evaluation Configuration

| Parameter | Value |
|---|---|
| Dataset | `TIGER-Lab/MMLU-Pro` |
| Split | `test` |
| Few-shot | 5 |
| Primary Metric | `acc_raw` (micro) |
| Regime | OLMo-style harness |
| No-answer rate | 0.0 (all questions answered) |
| Total Processing Time | ~35 seconds |

---

## Next Steps

1. **Evaluate `gemma-3-1b-it`** to get a score that is directly comparable to Google's published IT benchmarks
2. **Instruction fine-tuning (SFT)** on a general-purpose instruction dataset to move beyond the PT baseline
3. **Domain-specific fine-tuning** for high-stakes areas (Chemistry, Law, Engineering) which are both practically important and currently weakest
4. **RLHF / DPO alignment** to further improve reasoning and multi-step problem-solving
5. **Re-evaluate post fine-tuning** to track progression toward the 4B-IT (~43.6%) and 27B-IT (~67.5%) reference points
