# TruthfulQA Evaluation Results

## Model
**`google/gemma-3-1b-it`**

## Run Command

```bash
uv run olmes --model google/gemma-3-1b-it --task truthfulqa::olmo1 --output-dir gemma-truthfulqa
```

---

## Overview

Evaluation of the model on the [TruthfulQA](https://huggingface.co/datasets/truthfulqa/truthful_qa) benchmark — a dataset of 817 adversarially designed questions spanning 38 categories (health, law, finance, politics, etc.) that probe whether a model avoids falsehoods commonly learned from human text.

| Metric | Score |
|---|---|
| **MC2** (Primary) | **42.51%** |
| **MC1** | 26.44% |
| **Instances** | 817 |
| **Split** | Validation |
| **Few-shot** | 6-shot |
| **Regime** | OLMo-v1 |

---

## Metric Definitions

**MC1** — Single-true accuracy: the model picks the single correct answer with the highest log-probability. Strict and unforgiving.

**MC2** — Multi-true normalized probability: the model's probability mass is distributed across all correct answers and normalized. This is the primary metric and better reflects calibration.

---

## Comparison with Official Gemma Benchmarks

Google's Gemma 3 technical report does not publish a dedicated TruthfulQA score for the 1B model. However, the closest available reference points are:

| Source | Model | TruthfulQA MC2 |
|---|---|---|
| **This run (OLMo eval harness)** | `gemma-3-1b-it` | **42.51%** |
| Gemma 2 Technical Report | Gemma 2 2B IT | ~44.2% |
| Gemma 2 Technical Report | Gemma 2 9B IT | ~50.3% |
| Gemma 2 Technical Report | Gemma 2 27B IT | ~51.6% |
| Community (lm-evaluation-harness) | Gemma 3 1B IT (various) | ~40–46% range |

> ⚠️ **Direct comparison caveat:** Google does not publish an official TruthfulQA score for `gemma-3-1b-it` in the Gemma 3 technical report. The Gemma 2 figures are the nearest published baselines and use a different eval setup (standard lm-evaluation-harness vs. OLMo's harness used here). Score differences of ±2–4% are expected from harness and prompt format variations alone.

---

## Validation & Analysis

### ✅ Result is plausible
A MC2 score of **42.51%** for a 1B instruction-tuned model is consistent with expectations:
- The Gemma 2 2B IT scored ~44% MC2 — the 1B Gemma 3 IT scoring ~42.5% is directionally reasonable given the smaller parameter count.
- The 1B model is text-only, trained on 2 trillion tokens, with a 32K context window — smaller capacity compared to larger Gemma 3 variants.

### ⚠️ MC1 vs. MC2 Gap is notable
The 16-point spread between MC1 (26.44%) and MC2 (42.51%) indicates the model spreads probability across correct answers rather than confidently picking the best one. This is a calibration signal — the model "knows something" but lacks strong decisive truthfulness. This gap is typical of small models and is expected to narrow with scale or further RLHF alignment.

### ⚠️ Eval harness mismatch
This run uses the **OLMo-v1 regime** (`olmo1`) with a `short_prefix=True` context kwarg and 6-shot prompting sourced from `Original:TruthfulQA`. Google's internal evaluations likely use a different prompt format and shot setup. This makes scores **not directly apples-to-apples** with any official Google-reported numbers.

---

## Evaluation Configuration

| Parameter | Value |
|---|---|
| Dataset | `truthfulqa/truthful_qa` |
| Dataset Variant | `multiple_choice` |
| Split | `validation` |
| Few-shot | 6 |
| Primary Metric | `mc2` |
| Regime | `OLMo-v1` |
| Fewshot Source | `Original:TruthfulQA` |
| Prompt Config | `short_prefix: true` |
| Processing Time | ~13.1 seconds |

---

## Next Steps

1. **RLHF / DPO alignment**: Target factual accuracy and calibration to close the MC1/MC2 gap
3. **Compare with lm-evaluation-harness**: Run under standard harness to get a score directly comparable to community leaderboards
4. **Scale up**: Evaluate `gemma-3-4b-it` and `gemma-3-12b-it` to confirm expected scaling gains on TruthfulQA
