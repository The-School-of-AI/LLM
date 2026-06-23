# 🎯 LLM Pre-training & Evaluation Suite

Welcome to the central hub for the ERA-V4 Final Pre-training Benchmarks. This directory orchestrates a multi-tier evaluation system designed to validate model performance across different scales (1B to 70B parameters) and training phases.

---

## 📂 Directory Structure

| Component | Description |
| :--- | :--- |
| [**`02-OLMES-v1/`**](./02-OLMES-v1/) | **🚀 Primary Orchestrator**. The main entry point. Use this to run unified pipelines that combine OLMES, Harness, and Custom scripts. |
| [**`01-EleutherAI-v1/`**](./01-EleutherAI-v1/) | Core evaluation engines. Contains the `lm-evaluation-harness` integrations and custom benchmark scripts. |
| [**`token_count_analysis/`**](./token_count_analysis/) | **📊 Analysis Toolkit**. Tools for counting tokens, identifying dataset metrics, and estimating compute requirements (FLOPs). |
| [`benchmark-results/`](./benchmark-results/) | Automatically managed output directory containing timestamped logs, JSON results, and human-readable Markdown reports. |

---

## 🚦 High-Level Guidance

To get started with benchmarking, you should almost always work from the **`02-OLMES-v1`** directory. It acts as the "brain" that coordinates all other modules.

### 1. Setup Your Environment
```bash
cd 02-OLMES-v1/
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run a Unified Pipeline
The orchestrator allows you to execute complex evaluation "Stages" (defined in YAML) with a single command:
```bash
# Example: Running the CI Smoke Test stage
python3 src/pipeline_runner.py \
    --config configs/benchmark-config.yaml \
    --stage ci_breadth \
    --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" \
    --device "cuda"
```

### 3. Analyze Your Results
Once a run completes, explore the generated reports:
```bash
# Find the latest report
ls -ltr benchmark-results/
```

---

## 🛠 Which Module Should I Use?

- **Running Benchmarks?** Use `02-OLMES-v1/`.
- **Debugging a Custom Script?** Look into `01-EleutherAI-v1/src/custom-scripts/`.
- **Investigating Dataset Statistics?** Use `token_count_analysis/`.
- **Viewing Scores?** Check `benchmark-results/`.

---
> [!TIP]
> Always check the individual READMEs within each subdirectory for detailed, module-specific documentation.
