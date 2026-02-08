# OLMES Reporting Pipeline: Pre-training & SFT Benchmarking

The **OLMES Reporting Pipeline** is a robust, production-ready framework designed to automate the evaluation of Large Language Models across various training stages (1B to 70B+). It integrates the **allenai/OLMES** evaluation standard, providing a unified interface for tracking model progress through pre-training and SFT.

## 🌟 OLMES Capabilities

The OLMES engine brings several critical capabilities to this pipeline:
- **Comprehensive Task Suites**: Built-in support for standardized suites like MMLU, ARC, and HellaSwag, optimized for the OLMES regime.
- **Evaluation Paradigms**: Supports multiple evaluation styles including Multiple Choice (MC), Rank Classification (RC), and Chain-of-Thought (CoT).
- **Metric Consistency**: Ensures metrics are computed using standardized filters (e.g., `acc_per_char`, `bits_per_byte_corr`) for fair cross-model comparison.
- **Automated Setup**: Integrated auto-downloader and installer ensure the OLMES engine is always available and up-to-date.

## 🚀 Quick Start

### 1. Installation
The evaluator automatically detects and uses a local `.venv` if present.
```bash
# Recommended: create and setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r 02-OLMES-v1/requirements.txt
```

#### ⚠️ Environment Requirement (Mac/OSX)
The **allenai/olmes** internal runner expects a `python` command to be available in the path. If your environment only provides `python3`, you must create an alias or symlink:
```bash
# Example fix for active shell
alias python=python3

# Or symlink inside your venv
ln -s $(which python3) .venv/bin/python
```

### 2. Full Execution
Run the benchmarking suite on your checkpoint:
```bash
python3 02-OLMES-v1/src/eval_runner.py \
    --config 02-OLMES-v1/configs/stage_1b.yaml \
    --phase pretraining \
    --model_args "pretrained=path/to/checkpoint" \
    --device "cuda:0" \
    --batch_size "1"
```

> [!TIP]
> To run a quick test, add `--limit 5` to the command above. 
> **Note**: Avoid using `--limit 1`, as OLMES interprets it as 1.0 (100% of the dataset). Use integers like `2` or higher for sampling.

### ☁️ Google Colab Execution
Run the pipeline in a Colab notebook with a GPU (T4, L4, or A100):

1. **Setup Repo**:
   ```bash
   !git clone <your-repo-url>
   %cd <repo-folder>/experiments/17_final_pretraining_benchmarks
   ```

2. **Install Dependencies**:
   ```bash
   !pip install -r 02-OLMES-v1/requirements.txt
   ```

3. **Run Pipeline**:
   ```python
   # In a code cell
   !python3 02-OLMES-v1/src/eval_runner.py \
       --config 02-OLMES-v1/configs/stage_1b.yaml \
       --phase pretraining \
       --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" \
       --device "cuda" \
       --batch_size 8 \
       --limit 5
   ```
   *Note: The script automatically handles the `python` command shim and OLMES vendor setup on Colab.*

## ⚡ Batch Size Recommendations

The `--batch_size` parameter has a major impact on total evaluation time. **We recommend using a fixed value over `auto`** to avoid the slow per-task auto-detection phase.

| Mode | Value | Recommended For | Rationale |
| :--- | :--- | :--- | :--- |
| **Standard** | `1` | **Default / Robustness** | Skips slow auto-detection. Safest for all hardware and model sizes. |
| **High Perf** | `32`, `64` | **Optimized Runs** | Fastest execution if you know your GPU's VRAM limits. |
| **Auto** | `auto` | *Discouraged* | Convenient but adds significant overhead by searching for batch size on every task. |

---

## 📂 Output Structure
Every execution creates a unique, timestamped directory to prevent data loss and ensure clean logs:

```text
benchmark-results/
└── [stage]/                 (e.g., 1b, 8b)
    └── [phase]/             (e.g., pretraining, sft)
        └── [timestamp]/     (e.g., 20240201_123000)
            ├── incremental_results.json  <-- Saved after every task
            ├── logs/
            │   └── execution.log         <-- Full stdout/stderr capture
            ├── reports/
            │   └── summary_report.md     <-- Human-readable Markdown
            └── harness_raw/              <-- Raw JSON from lm-eval per task
                ├── mmlu.json
                └── gsm8k.json
```

## 🛠 Features

### 1. Granular Reporting
The YAML configuration supports specific subjects and subsets. The evaluator will automatically:
- Expand `subjects` (e.g., MMLU subjects) into individual harness tasks.
- Aggregate these sub-tasks into a parent benchmark score.
- Generate a nested Markdown report showing both the **Aggregate** and **↳ Sub-task** scores.

### 2. Incremental Saving
Results are saved to `incremental_results.json` immediately after each benchmark completes. If a 24-hour run crashes at hour 23, you still have 23 hours of data.

### 3. Intelligent Execution
- **Venv Detection**: Automatically uses `.venv/bin/python3` if available.
- **Robust Parsing**: Extracts metrics from `lm-eval` regardless of the filter used (e.g., `acc`, `exact_match`, `acc_norm`).
- **Conflict Resolution**: Strips potential `device` conflicts between CLI arguments and `model_args`.

## ⚙️ Configuration (YAML)
Each benchmark in `configs/*.yaml` can specify:
- `harness_task`: Base task name in `lm-evaluation-harness`.
- `subjects`: List of MMLU-style subjects to run specifically.
- `tasks`: List of specific subsets (e.g., for BLiMP).
- `subset`: Specific dataset subset (e.g., for TriviaQA).
- `shots`: Number of few-shot examples.
- `phases`: `[pretraining]` or `[sft]`.

---