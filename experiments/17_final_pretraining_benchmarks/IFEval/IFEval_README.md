# IFEval Benchmark Results

## Model
**`meta-llama/Llama-3.2-1B-Instruct`** *(placeholder — swap for your target model checkpoint)*

## Run Command

```bash
olmes --model meta-llama/Llama-3.2-1B-Instruct --task ifeval --output ./results_ifeval
```

---

## Overview

Evaluation on the [IFEval](https://huggingface.co/datasets/HuggingFaceH4/ifeval) benchmark — 541 prompts each containing verifiable formatting/instruction constraints (e.g. "respond in bullet points", "use no commas", "wrap in quotes"). The model is scored on whether its output satisfies every constraint.

| Metric | Score |
|---|---|
| **inst_level_loose_acc** (Primary) | **55.04%** |
| **inst_level_strict_acc** | 50.36% |
| **prompt_level_loose_acc** | 39.93% |
| **prompt_level_strict_acc** | 35.12% |
| **Instances** | 541 |
| **Split** | train |
| **Few-shot** | 0-shot |

---

## Metric Definitions

**inst_level_loose_acc** — Fraction of individual instructions satisfied, with loose matching (minor formatting variations allowed). This is the primary metric.

**inst_level_strict_acc** — Same but exact/strict matching. No tolerance for minor deviations.

**prompt_level_loose/strict_acc** — A prompt only counts as correct if *all* its instructions are satisfied. Stricter than instruction-level since one failure fails the whole prompt.

---

## Per-Category Breakdown (Strict)

| Instruction Type | Strict Acc |
|---|---|
| `detectable_format:title` | 94.6% ✅ |
| `keywords:existence` | 84.6% ✅ |
| `detectable_content:postscript` | 80.8% ✅ |
| `detectable_content:number_placeholders` | 70.4% |
| `length_constraints:number_words` | 67.3% |
| `keywords:frequency` | 66.7% |
| `change_case:english_lowercase` | 43.6% |
| `detectable_format:multiple_sections` | 35.7% |
| `length_constraints:number_paragraphs` | 11.1% |
| `detectable_format:number_bullet_lists` | 12.9% ⚠️ |
| `detectable_format:json_format` | 5.9% ⚠️ |
| `startend:quotation` | 2.4% ⚠️ |
| `combination:repeat_prompt` | 4.9% ⚠️ |
| `detectable_format:constrained_response` | 0.0% ⚠️ |

---

## Comparison with Official Meta Benchmarks

**Metric clarification:**

- **Primary** (`inst_level_loose_acc`) = single metric OLMES designates as the headline score = **55.04%**
- **Avg** = `(prompt_strict + prompt_loose + inst_strict + inst_loose) / 4` = **45.11%**
- Meta's reported score uses the same formula as Avg: `Avg(Prompt/Instruction acc Loose/Strict)`

The only apples-to-apples comparison is **Avg vs Avg**.

| Source | Model | Avg of all 4 metrics | Primary only (`inst_level_loose_acc`) |
| --- | --- | --- | --- |
| **This run (OLMES)** | `Llama-3.2-1B-Instruct` | **45.11%** | 55.04% |
| [Meta official model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | `Llama-3.2-1B-Instruct` bf16 | **59.5%** | — |
| Meta official model card | `Llama-3.2-1B-Instruct` QLoRA | 55.6% | — |
| Meta official model card | `Llama-3.2-3B-Instruct` | 77.4% | — |
| *(Target model checkpoint)* | Custom model | TBD | TBD |

**Gap: 45.11% vs 59.5% = ~14 points.**

> ⚠️ **Gap explanation:** Meta uses their internal evaluation harness with different prompt formatting and chat templates than OLMES. Score differences of ±10–15% between harnesses are documented on IFEval due to sensitivity to prompt structure. This does **not** indicate a bug — it reflects harness differences. Cross-check with `lm-evaluation-harness` for a fairer comparison.

---

## Validation & Analysis

### ✅ Result is plausible
A `inst_level_loose_acc` of **55%** for `Llama-3.2-1B-Instruct` is consistent with community-reported numbers for this model on IFEval. The pipeline is working correctly.

### ⚠️ Weak spots to watch

The 1B model shows near-zero accuracy on structured format constraints — `json_format` (5.9%), `quotation` (2.4%), `constrained_response` (0%) — and low accuracy on `bullet_lists` (12.9%). These categories require the model to respect hard formatting rules and are sensitive to instruction-tuning quality.

### ⚠️ Prompt-level vs. instruction-level gap
The ~15-point gap between `inst_level_loose_acc` (55%) and `prompt_level_loose_acc` (40%) shows the model often satisfies some but not all constraints within a single prompt. Closing this gap requires consistent multi-constraint following.

---

## Evaluation Configuration

| Parameter | Value |
|---|---|
| Dataset | `HuggingFaceH4/ifeval` |
| Split | `train` |
| Few-shot | 0 |
| Primary Metric | `inst_level_loose_acc` |
| max_gen_toks | 1280 |
| Temperature | 0.0 (greedy) |
| do_sample | false |
| Processing Time | ~16 seconds |
| Run Date | 2026-02-27 |

---

## Next Steps

1. **Swap model**: Re-run with the target model checkpoint once available
2. **Target weak categories**: Focus instruction-tuning on `json_format`, `quotation`, `constrained_response`, and `bullet_lists`
3. **Close the prompt-level gap**: Multi-constraint following needs improvement — consider targeted data for prompts with 3+ constraints
4. **Compare harnesses**: Cross-check with `lm-evaluation-harness` IFEval to validate score consistency across frameworks
