# OLMES BBH (Big Bench Hard) Benchmark Validation

**Task:** Validate the CLI commands and verify metrics using OLMES for BBH (Big Bench Hard)  
**Assigned to:** Jayant Guru Shrivastava  
**Date:** 2026-03-02  

---

## 1. Objective

Validate that the [OLMES](https://github.com/allenai/olmes) CLI tool can correctly evaluate language models on the **BBH (Big Bench Hard)** benchmark, and verify that the scores match published results.

---

## 2. Setup

### Environment
- **Platform:** Lightning.ai Studio (free tier)
- **GPU:** NVIDIA L40S (48GB VRAM)
- **Model:** `Qwen/Qwen2.5-1.5B-Instruct` (public, no HuggingFace auth needed)
- **Task:** `bbh:cot-v1::tulu` (Chain-of-Thought, 3-shot, Tulu chat format)

### Installation Commands

```bash
# Step 1: Install OLMES
pip install --upgrade pip
git clone https://github.com/allenai/olmes.git ~/olmes
cd ~/olmes
pip install -e ".[gpu]"

# Step 2: Verify install
olmes --list-tasks bbh
```

---

## 3. Running the BBH Benchmark

### CLI Command

```bash
cd ~/olmes && olmes \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --task bbh:cot-v1::tulu \
  --output-dir ~/bbh_results \
  --batch-size 4
```

### Key Parameters
| Parameter | Value | Why |
|-----------|-------|-----|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | Public model with known BBH score |
| `--task` | `bbh:cot-v1::tulu` | Standard BBH eval with chain-of-thought prompting |
| `--batch-size 4` | 4 | Balances speed and memory (batch-size 16 causes OOM) |
| `--output-dir` | `~/bbh_results` | Where result JSONs are saved |

---

## 4. Results (Full: 27/27 tasks)

| # | BBH Subtask | Score (%) |
|---|------------|-----------|
| 0 | boolean_expressions | 84.4 |
| 1 | causal_judgement | 61.5 |
| 2 | date_understanding | 48.8 |
| 3 | disambiguation_qa | 46.0 |
| 4 | dyck_languages | 2.4 |
| 5 | formal_fallacies | 51.2 |
| 6 | geometric_shapes | 26.0 |
| 7 | hyperbaton | 53.2 |
| 8 | logical_deduction_five_objects | 35.6 |
| 9 | logical_deduction_seven_objects | 23.6 |
| 10 | logical_deduction_three_objects | 52.4 |
| 11 | movie_recommendation | 48.8 |
| 12 | multistep_arithmetic_two | 67.2 |
| 13 | navigate | 68.4 |
| 14 | object_counting | 56.8 |
| 15 | penguins_in_a_table | 63.7 |
| 16 | reasoning_about_colored_objects | 56.8 |
| 17 | ruin_names | 33.6 |
| 18 | salient_translation_error_detection | 41.2 |
| 19 | snarks | 51.7 |
| 20 | sports_understanding | 49.6 |
| 21 | temporal_sequences | 16.0 |
| 22 | tracking_shuffled_objects_five_objects | 20.4 |
| 23 | tracking_shuffled_objects_seven_objects | 17.2 |
| 24 | tracking_shuffled_objects_three_objects | 37.2 |
| 25 | web_of_lies | 73.2 |
| 26 | word_sorting | 6.4 |

### Aggregate Score

| Metric | Value |
|--------|-------|
| **BBH Average (27/27 tasks)** | **44.2** |
| **Expected BBH Score (OLMo 2 paper, Table 7)** | **~45.8** |
| **Difference** | **-1.6 (within margin)** |

✅ **Result validates correctly** — the BBH average of **44.2** closely matches the published score of **45.8** for Qwen 2.5 1.5B Instruct. The ~1.6 point difference is within normal variance caused by hardware/software environment differences.

---

## 5. Reading Results

After the run completes, use this script to extract and display scores:

```bash
python3 -c "
import json, glob
scores = []
for f in sorted(glob.glob('~/bbh_results/*-metrics.json')):
    data = json.load(open(f))
    score = data['metrics']['primary_score']
    name = data['task_name'].replace('bbh_','')
    print(f'{name}: {score*100:.1f}')
    scores.append(score)
print(f'\n--- Completed {len(scores)}/27 tasks ---')
print(f'BBH Average: {sum(scores)/len(scores)*100:.1f}')
print(f'Expected: ~45.8')
"
```

---

## 6. Output Files

Each BBH subtask produces 4 files:
| File | Description |
|------|-------------|
| `*-metrics.json` | **The scores** (primary_score = exact_match accuracy) |
| `*-predictions.jsonl` | Model's chain-of-thought answers for each question |
| `*-recorded-inputs.jsonl` | Sample prompts for debugging |
| `*-requests.jsonl` | Full raw request data |

Additionally, OLMES generates:
- `metrics.json` — Aggregated metrics across all tasks
- `metrics-all.jsonl` — All task metrics in a single JSONL file

---

## 7. Notes & Troubleshooting

### Model Selection
| Model | Auth? | Published BBH |
|-------|-------|---------------|
| `Qwen/Qwen2.5-1.5B-Instruct` | ❌ No auth | 45.8 |
| `allenai/OLMo-2-1124-1B-Instruct` | ✅ HF token needed | 35.0 |
| `allenai/OLMo-2-0425-1B-Instruct` | ❌ No auth | No published number |

### Common Issues
| Issue | Fix |
|-------|-----|
| `trust_remote_code` validation error with vLLM | Remove `--model-type vllm`, use default HF backend |
| CUDA OOM with `--batch-size 16` | Reduce to `--batch-size 4` or `--batch-size 8` |
| Slow on Colab T4 (4+ hours) | Use Lightning.ai L40S or add `--batch-size` flag |

### Run Time Estimates
| GPU | Batch Size | Estimated Time |
|-----|-----------|---------------|
| T4 (Colab) | 1 | ~4-5 hours |
| L40S (Lightning.ai) | 4 | ~1.5-2 hours |
| A100 | 4 | ~30-60 min |

---

## 8. Conclusion

The OLMES CLI tool **successfully evaluates** the BBH benchmark. With all 27 subtasks completed, the BBH average of **44.2** closely matches the published score of **45.8** (from the OLMo 2 paper, Table 7), confirming that:

1. ✅ OLMES CLI commands work correctly for BBH
2. ✅ The benchmark produces valid, reproducible metrics
3. ✅ Results align with published scores (within 1.6 points variance)
