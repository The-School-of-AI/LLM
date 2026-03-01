# RiddleBench Evaluation

> **Dataset:** [ai4bharat/RiddleBench](https://huggingface.co/datasets/ai4bharat/RiddleBench)
> **Paper:** [arXiv 2510.24932](https://arxiv.org/abs/2510.24932)

## About the Benchmark

RiddleBench is a reasoning benchmark from AI4Bharat containing **1,737 puzzle-style questions** across four task categories. Each question embeds its own instructions (e.g. *"only reply with the answer"*) and expects a short free-text answer — typically a number, letter, or short phrase.

The dataset ships as a single HuggingFace `train` split with the following schema:

| Column     | Type   | Description                                              |
|------------|--------|----------------------------------------------------------|
| `id`       | int64  | Unique question identifier                               |
| `type`     | string | Task category (one of 4 values, see below)               |
| `question` | string | Full question text, already includes task instructions    |
| `answer`   | string | Gold answer (free-text, e.g. `"5435"`, `"B"`, `"126"`)   |

### Task Categories

| Category                  | # Items | Description                                                       |
|---------------------------|--------:|-------------------------------------------------------------------|
| **sequence tasks**        |     990 | Find the missing term in a numerical sequence                     |
| **seating task**          |     432 | Determine positions or relations in seating arrangements          |
| **coding and decoding sum** |   169 | Decode encoded values and compute the result                      |
| **blood relations**       |     146 | Infer family relationships from a set of clues                    |

## What the Script Does

`eval.py` loads the dataset, runs each question through a HuggingFace causal-LM, extracts the model's answer, and scores it against the gold label with exact-match (after normalisation).

### Answer Extraction

The script applies a cascade of heuristics to pull the final answer from the model's raw output:

1. Regex match for `Answer: X` / `Result: X` patterns
2. Regex match for a trailing `= <number>` expression
3. Fallback: last non-empty line of the output

### Answer Normalisation

Both predicted and gold answers are lowercased, stripped, and — if numeric — converted to a canonical form (e.g. `"5435.0"` → `"5435"`, `"1417.50"` → `"1417.5"`). Matching is then exact string comparison.

## Output Metrics

The script produces a JSON results file and a console summary with:

| Metric           | Description                                                   |
|------------------|---------------------------------------------------------------|
| **accuracy**     | Overall exact-match accuracy (%)                              |
| **correct**      | Total number of correctly answered items                      |
| **total_items**  | Number of items evaluated                                     |
| **avg_latency_s**| Mean wall-clock seconds per item                              |
| **per_type**     | Breakdown of `correct`, `total`, and `accuracy` per category  |

The per-item JSON array additionally records `predicted`, `gold`, `raw_output`, `correct` (bool), and `latency_s` for every question.

## Prerequisites

```bash
pip install transformers datasets torch accelerate
```

## Usage

```bash
# Run with the default model (Qwen/Qwen2.5-0.5B-Instruct), full dataset
python eval.py

# Specify a model explicitly
python eval.py --model Qwen/Qwen2.5-0.5B-Instruct

# Quick smoke test: first 50 items, verbose output
python eval.py --limit 50 --verbose

# Evaluate a single category
python eval.py --type "sequence tasks" --limit 20

# Custom output path and generation settings
python eval.py --output my_results.json --max-new-tokens 128 --device cuda
```

### CLI Options

| Flag                | Default                           | Description                                    |
|---------------------|-----------------------------------|------------------------------------------------|
| `--model`           | `Qwen/Qwen2.5-0.5B-Instruct`     | HuggingFace model name or local path           |
| `--type`            | _(all)_                           | Filter to a single task category                |
| `--limit`           | _(all)_                           | Cap the number of items evaluated               |
| `--verbose`         | `false`                           | Print question and raw output for every item    |
| `--output`          | `results.json`                    | Path for the JSON results file                  |
| `--device`          | `auto`                            | Device map (`auto`, `cpu`, `cuda`, etc.)        |
| `--max-new-tokens`  | `64`                              | Max tokens to generate per answer               |

## Validated Results — Qwen/Qwen2.5-0.5B-Instruct

The script has been validated end-to-end on **Qwen/Qwen2.5-0.5B-Instruct** across all 1,737 items. Results file: `results_qwen0.5b.json`.

```
==============================================================
  RiddleBench -- Qwen/Qwen2.5-0.5B-Instruct
  1737 items  |  26 correct  |  1.5% accuracy
  Avg latency: 1.09 s / item
==============================================================
  Type                         Correct   Total      Acc
  ---------------------------- -------  ------  -------
  blood relations                    0     146     0.0%
  coding and decoding sum            3     169     1.8%
  seating task                       0     432     0.0%
  sequence tasks                    23     990     2.3%
==============================================================
```

**Observations:**
- The 0.5B model achieves **1.5% overall accuracy**, indicating that RiddleBench is extremely challenging for small language models.
- **Sequence tasks** are the only category where the model scores above zero beyond random noise (2.3%, 23/990).
- **Blood relations** and **seating tasks** are completely unsolved (0.0%) — these require multi-step relational reasoning that is well beyond this model's capacity.
- **Coding and decoding sum** sees only 3 correct answers out of 169 (1.8%).
- Average latency is ~1.09 s per item (measured on the hardware used during validation).
