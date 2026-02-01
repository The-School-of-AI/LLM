# Pre-training & SFT Benchmarking Suite

This module provides a configurable framework to run specific benchmarks based on the model training stage (1B, 3B, 8B, 70B) and phase (Pre-training/SFT). It primarily uses the `lm-evaluation-harness` for standard tasks and supports custom scripts for others.

## Directory Structure
- `configs/`: YAML configurations for each model stage (Harness tasks + Custom specs).
- `src/`: Core evaluation logic (`eval_runner.py`).
- `scripts/`: Placeholder/Actual custom evaluation scripts (e.g., AIME, SWE-bench).
- `results/`: Standardized JSON output storage.

## Usage

### 1. Pre-training Evaluation (1B -> 70B)
To verify the base model capabilities at any stage of growth:
```bash
python3 src/eval_runner.py --config configs/stage_1b.yaml --phase pretraining --test
```

### 2. SFT/Alignment Evaluation (Post-70B)
SFT benchmarks are handled via a dedicated configuration designed for the final model state:
```bash
python3 src/eval_runner.py --config configs/sft_stage.yaml --phase sft --test
```

### 3. Real Execution (using lm-evaluation-harness)
Ensure `lm-eval` is installed. Provide model arguments compatible with the harness.
```bash
python3 src/eval_runner.py \
    --config configs/stage_70b.yaml \
    --phase pretraining \
    --model_args "pretrained=path/to/checkpoint,dtype=bfloat16"
```

## Configuration (YAML)
Each benchmark in the YAML file specifies:
- `harness_task`: The task name in `lm-evaluation-harness` (e.g., `mmlu`, `gsm8k`).
- `custom_script`: Path to a python script for benchmarks not in the harness.
- `shots`: Number of few-shot examples.
- `mode/paradigm`: Stage-specific running modes (e.g., MC1 vs MC2, Zero-Shot vs CoT) as defined in `Paradigms.md`.
- `tasks/subjects`: Curated subsets relevant to the model's capacity stage.
- `phases`: List containing `pretraining` and/or `sft`.
- `enabled`: Boolean to toggle the benchmark for that specific model stage.

## Benchmarks Covered
The suite covers all 25 benchmarks listed in `benchmarks-list.txt`, including:
- **Linguistic**: BLiMP, HellaSwag, Winogrande, MSGS.
- **Reasoning**: GSM8K, MATH, ARC, BBH, GPQA Diamond, AIME 2025.
- **Knowledge**: MMLU, TriviaQA, SimpleQA.
- **Code**: HumanEval, APPS, SWE-bench.
- **Alignment/Safety**: IFEval, TruthfulQA, HELM Safety.
- **Multilingual (Indic)**: IndicGLUE, IndicQA, Indic-Bias.
- **Long Context**: L-Eval, RULER.

All configurations are tuned to focus on the specific metrics and risks identified in `Paradigms.md` for each growth stage (1B -> 70B).
