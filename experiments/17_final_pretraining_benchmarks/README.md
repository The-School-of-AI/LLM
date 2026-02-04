# Pre-training & SFT Benchmarking Suite

This module provides a robust, configurable framework to run specific benchmarks based on the model training stage (1B, 3B, 8B, 70B) and phase (Pre-training/SFT). It leverages the `lm-evaluation-harness` for standard tasks and supports custom scripts for advanced evaluations.

## 🚀 Quick Start

### 1. Installation
The evaluator automatically detects and uses a local `.venv` if present.
```bash
# Recommended: create and setup venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r 01-EleutherAI-v1/requirements.txt
```

### 2. Run a Trial (Quick Verification)
Run a limited execution (5 samples per task) to ensure weights and metrics load correctly:
```bash
python3 01-EleutherAI-v1/src/eval_runner.py \
    --config 01-EleutherAI-v1/configs/stage_1b.yaml \
    --phase pretraining \
    --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" \
    --device "cpu" \
    --trial
```

### 3. Full Execution (Production)
Run the full benchmarking suite on your checkpoint:
```bash
python3 01-EleutherAI-v1/src/eval_runner.py \
    --config 01-EleutherAI-v1/configs/stage_1b.yaml \
    --phase pretraining \
    --model_args "pretrained=path/to/checkpoint" \
    --device "cuda:0" \
    --batch_size "1"
```

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
results/
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

## 📊 Benchmark Analysis (NEW!)

Before finalizing your evaluation pipeline, use the **Benchmark Analysis Toolkit** to investigate datasets:

```bash
cd benchmark_analysis
./run_analysis.sh
```

This toolkit helps you:
- **Count tokens** in test datasets (for FLOPs calculation)
- **Identify evaluation metrics** (accuracy, F1, exact match, etc.)
- **Estimate computational requirements** (GPU-hours, FLOPs)
- **Generate reports** (CSV, JSON, Markdown)

📖 **Documentation**: See `benchmark_analysis/INDEX.md` for complete guide

**Quick Links**:
- `benchmark_analysis/QUICKSTART.md` - Get started in 5 minutes
- `benchmark_analysis/OVERVIEW.md` - Understand the toolkit
- `benchmark_analysis/WORKFLOW.md` - Step-by-step process

---

## 🚧 Future Work & TODOs

- [ ] **Custom Scripts**: Implement logic for placeholders in `scripts/` (AIME 2025, IndicGLUE, SWE-bench, etc.).
- [x] **Benchmarks List Review**: ✅ Use `benchmark_analysis/` toolkit to analyze all 25 benchmarks
- [ ] **YAML Config Audit**: Comprehensive review of all task parameters (shots, mode, paradigm) in `1b`, `3b`, `8b`, `70b`, and `sft` YAMLs vs `Paradigms.md`.
- [ ] **Multi-GPU Support**: Add orchestration for native model parallelism and FSDP via `accelerate`.
- [ ] **MSGS Benchmark**: Investigate missing MSGS tasks in `v0.4.x` and re-enable or find substitutes.
- [ ] **Visualizer**: Create a simple utility to compare `incremental_results.json` files across different model stages.

---
*Refer to `benchmarks-list.txt` for the full list of 25 supported benchmarks.*
*Use `benchmark_analysis/` to investigate token counts and metrics before finalizing configs.*
*Refer to `01-EleutherAI-v1/benchmarks-list.txt` for the full list of 25 supported benchmarks.*
