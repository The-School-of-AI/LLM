# OLMES Evaluation Pipeline (v1)

A unified benchmarking orchestrator designed to track model progress from early pre-training (1B) to high-scale alignment (70B+). This pipeline automates the execution of **OLMES**, **LM-Evaluation-Harness**, and **Custom Indic scripts** through a single declarative configuration.

## 🏗 Pipeline Architecture

The pipeline is split into three layers:
1.  **Orchestrator (`pipeline_runner.py`)**: Loads the stage configurations, resolves task types, and manages the run lifecycle.
2.  **Core Executive (`eval_runner.py`)**: A shared library that handles environment setup, vendor patching (for macOS/HPC), and execution of specific engines.
3.  **Config Layer (`benchmark-config.yaml`)**: Defines what benchmarks run at which training stage, along with capability groupings (buckets) and comparative baselines.

---

## 📜 The Two-Pipeline Strategy

To balance **research speed** with **scientific rigor**, this project maintains two distinct benchmarking configurations.

### 1. Developer Feedback Pipeline (`benchmark-config.yaml`)
*   **Purpose**: Rapid iteration and regression testing during active pre-training.
*   **Philosophy**: "Directional correctness over absolute precision."
*   **Optimizations**: Uses `:easy` task subsets and fast **RC (Completion)** formulations.
*   **When to use**: Hourly or nightly runs to ensure the model isn't "collapsing."

### 2. Industry Standard Pipeline (`industry-benchmarks.yaml`)
*   **Purpose**: Formal benchmarking for model cards and technical reports.
*   **Philosophy**: "Maximum comparability and statistical significance."
*   **Rigors**: Uses **N=1,000** samples or **Full** datasets, and the **Best of MC/RC** methodology (Best-of-both-worlds).
*   **When to use**: Major milestones (e.g., end of 1B, 8B, 70B stages) or before a public release.

For a deeper dive into the exact academic targets (Meta, Alibaba, AI2) and SOTA scores for each tier, see:
👉 **[INDUSTRY_BENCHMARKS.md](./INDUSTRY_BENCHMARKS.md)**

---

## 🚀 Execution Guide

### 1. Environment Setup
The pipeline requires a stable Python environment (**Python 3.12 recommended**). Avoid experimental versions like 3.14 for now.

```bash
cd 02-OLMES-v1/
# Recommended: Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies using the venv's python
./.venv/bin/python3 -m pip install -r requirements.txt
```

### Gated Datasets (Indic-Bias, etc.)
Some benchmarks require access to gated HuggingFace datasets. To enable them:
1. **Accept the dataset terms** on HuggingFace (e.g., visit [ai4bharat/Indic-Bias](https://huggingface.co/datasets/ai4bharat/Indic-Bias) and click "Accept").
2. **Set your token** before running the pipeline:
```bash
export HF_TOKEN="hf_your_token_here"
```
The pipeline will confirm detection at startup with a `🔑 HF_TOKEN detected` message.

# To evaluate a model checkpoint against a specific stage (e.g., pretrain_1b or pretrain_70b):
```bash
python3 src/pipeline_runner.py \
    --config configs/benchmark-config.yaml \
    --stage pretrain_1b \
    --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" \
    --device "cpu"
```

### 3. Smoke Testing (Quick Verification)
Use the included helper script to run a "one of everything" test with a tiny sample limit (2 samples/task) across all stages:
```bash
# From 17_final_pretraining_benchmarks/02-OLMES-v1

# Run with defaults (all stages, SmolLM2-135M, cpu)
tests/run_smoke_tests.sh

# Custom model and specific stages
tests/run_smoke_tests.sh \
    --model "your-org/your-model" \
    --stages "pretrain_1b,pretrain_3b"

# Full customization
tests/run_smoke_tests.sh \
    --config configs/benchmark-config.yaml \
    --model "your-org/your-model" \
    --stages "pretrain_1b,ci_breadth" \
    --device "cuda"
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `-c, --config` | `configs/benchmark-config.yaml` | Path to benchmark config YAML |
| `-m, --model` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model name |
| `-s, --stages` | `pretrain_1b,pretrain_3b,...,ci_breadth` | Comma-separated list of stages |
| `-d, --device` | `cpu` | Execution device (`cpu`, `cuda`, `mps`) |
| `-t, --hf-token` | *(env `HF_TOKEN`)* | HuggingFace API token for gated datasets |

---

## � Supported Training Stages

The pipeline defines specialized stages to match the model's maturity:

| Stage | Focus | Key Coverage |
| :--- | :--- | :--- |
| **`ci_breadth`** | **CI Smoke Test** | Fast representative sample (English, Indic, CoT, Code Completion). |
| **`pretrain_1b`** | **Early Signal** | Fast RC/BPB + Indic NLU trends. |
| **`pretrain_3b`** | **Enhanced Signal** | Pretrain 1B + Core Knowledge & QA. |
| **`pretrain_8b`** | **Full Milestone** | Full Base Suite (STEM, Math, Code Completion) + MMLU. + Extended Indic.|
| **`pretrain_70b`** | **Reasoning Ready** | Full Base + `bbh:cot` + Extended Indic. |
| **`sft`** | **Agency & Chat** | IFEval, AlpacaEval, Factuality (SimpleQA/PopQA). |

---

## �📊 Understanding Results

Every run creates a unique timestamped directory in `benchmark-results/`.

### 1. Human-Readable Reports
Navigate to `benchmark-results/[stage]/[timestamp]/reports/summary_report.md`.
- **Executive Summary**: Overall pass rate and completion status.
- **Capability Mapping**: Benchmarks grouped into logical buckets (Reasoning, Code Completion, Indic, etc.).
- **Granular Details**: Collapsible sections showing every individual sub-task score (e.g., individual MMLU subjects).

### 2. Machine-Comparable Data
For automated analysis or cross-run comparisons, use `reports/final_results.json`:
- **`aggregates`**: Flat mapping of top-level benchmark suite scores.
- **`granular`**: Flat mapping of **all** underlying sub-tasks and their scores across all suites.
- **`metadata`**: Full context of the run (model args, device, limit, etc.).

---

## 🚧 Status & Roadmap

The following benchmarks have known limitations or are currently under active development.

| Feature / Benchmark | Status / Limitation | Next Steps / Required Setup |
| :--- | :--- | :--- |
| **Indic-Bias** | **Gated Access** | Requires access to `ai4bharat/Indic-Bias`. ✅ `HF_TOKEN` injection implemented — see [Environment Setup](#-execution-guide). |
| **olmo3:adapt** | **Base Model Noise** | Unstable on base models. **TODO**: Validate on first instruction-aligned milestones. |
| **MMLU-Pro** | **High Resource** | Very slow execution. **TODO**: Integrate into milestones after optimization. |
| **HELM Safety** | **Network Fragility** | `RealToxicityPrompts` downloads are flaky. Recommendation: Pre-cache datasets locally. |

---

## �🔍 Detailed Benchmark Catalog

For a complete breakdown of every task, dataset, and sub-task included in each suite, refer to:
👉 **[BENCHMARK_DETAILS.md](./BENCHMARK_DETAILS.md)**

This document explains the "Signal of Life" strategy, the progression from Bits-Per-Byte (BPB) to generative evaluation, and the specific composition of complex suites like `bbh:cot` and `olmo3:adapt`.
