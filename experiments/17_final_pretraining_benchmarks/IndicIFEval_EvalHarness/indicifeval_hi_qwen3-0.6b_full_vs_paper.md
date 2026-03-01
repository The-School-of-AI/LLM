# IndicIFEval Hindi (hi) — Full Report

- Model: Qwen/Qwen3-0.6B
- Generated: 2026-03-01T14:22:17Z
- Updated: 2026-03-02
- Trans run dir: C:/Users/sidhe/TSAIV4/capstone/team17/benchmarkscripts/IndicIFEval_EvalHarness/scripts/../results/hf/Qwen_Qwen3-0.6B/hi_trans_full
- Ground run dir: C:/Users/sidhe/TSAIV4/capstone/team17/benchmarkscripts/IndicIFEval_EvalHarness/scripts/../results/hf/Qwen_Qwen3-0.6B/hi_ground_full

## Results

| Task | Split | prompt_level_strict_acc | inst_level_strict_acc | prompt_level_loose_acc | inst_level_loose_acc |
|---|---|---:|---:|---:|---:|
| indicifeval_trans_hi | model | 0.1449 | 0.2759 | 0.1592 | 0.2894 |
| indicifeval_trans_hi | paper | N/A | N/A | 0.3050 | N/A |
| indicifeval_ground_hi | model | 0.2757 | 0.2757 | 0.3042 | 0.3042 |
| indicifeval_ground_hi | paper | N/A | N/A | 0.5690 | N/A |
| **avg(trans, ground)** | model | 0.2103 | 0.2758 | 0.2317 | 0.2968 |

## Paper comparison

Paper reference: https://arxiv.org/html/2602.22125v1 (Table 1 = Trans, Table 2 = Ground).

| Task | Metric | Model (%) | Paper (%) | Δ (pp) |
|---|---|---:|---:|---:|
| indicifeval_trans_hi | prompt_level_loose_acc | 15.9 | 30.5 | -14.6 |
| indicifeval_ground_hi | prompt_level_loose_acc | 30.4 | 56.9 | -26.5 |

Notes:
- The paper reports prompt-level *loose* accuracy as percentages; this report compares against `prompt_level_loose_acc * 100`.
- Table 1 (Trans) uses a curated common subset of 321 prompts per language; see the investigation notes below for how this maps to the HF dataset.
- Table 2 (Ground) is not directly comparable to Trans (different prompts, no English baseline).

## Investigation notes (why local accuracy is much lower than the paper)

These checks were run to understand the large gaps (Trans: -14.6 pp, Ground: -26.5 pp) in the table above.

### 1) What does the paper’s “321 common subset” correspond to?

On the current `ai4bharat/IndicIFEval` HF dataset for Hindi Trans:

- Total rows in `indicifeval-trans` / `hi`: **490**
- Rows tagged `parallel`: **321**

So the paper’s Table 1 “common subset of 321 prompts per language” corresponds exactly to the dataset’s `parallel` tag.

### 2) Does evaluating 490 vs 321 explain the lower Trans score?

No. Using the saved per-sample outputs from the full run and recomputing `prompt_level_loose_acc` on subsets:

| Subset (indicifeval-trans/hi) | n | prompt_level_loose_acc (%) |
|---|---:|---:|
| ALL (full split) | 490 | 15.92 |
| tag:`parallel` (paper subset size) | 321 | 14.64 |

This means “wrong subset size” is not the reason the local Trans accuracy is below the paper; the 321 subset is slightly harder (for this run) than the full 490.

### 3) Prompt formatting likely differs (chat template vs raw prompt)

`Qwen/Qwen3-0.6B` is a chat-tuned model and its tokenizer exposes a chat template (with sentinel tokens like `<|im_start|>` / `<|im_end|>`). The original runs in this report used raw `doc["prompt"]` without chat formatting.

To test whether this matters, a minimal Qwen chat wrapper was tried:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

…and generation was stopped on `<|im_end|>`.

Quick sanity runs (`--limit 20`, *not* statistically stable) show substantial improvements vs raw prompting:

| Task | Prompting | n | prompt_level_loose_acc (%) |
|---|---|---:|---:|
| indicifeval_trans_hi | raw prompt (this report) | 490 | 15.92 |
| indicifeval_trans_hi_chat | Qwen chat wrapper | 20 | 20.00 |
| indicifeval_ground_hi | raw prompt (this report) | 457 | 30.42 |
| indicifeval_ground_hi_chat | Qwen chat wrapper | 20 | 45.00 |

This makes “chat formatting mismatch” the strongest remaining explanation for the paper vs local accuracy gap.

### Recommended next step to match the paper more closely

- Re-run full Hindi using chat-formatted tasks (and optionally restrict Trans to tag:`parallel`):
- `indicifeval_trans_hi_chat`
- `indicifeval_ground_hi_chat`

Then regenerate this report pointing `--trans_dir` and `--ground_dir` at those new run directories.

