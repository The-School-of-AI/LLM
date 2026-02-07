# Test Execution Suite

This folder contains the configuration and results for verifying the benchmarking pipeline on small models.

## Structure
- `test_config.yaml`: A minimal configuration running a single, fast benchmark (`HellaSwag` with 0-shot).
- `results/`: Directory for test execution outputs.

## Verified Staging
all YAML files in the `configs/` directory have been verified for syntax and compatibility with the `eval_runner.py` script.

## Running a Test Execution
To run a real evaluation on a small model (e.g., `SmolLM2-135M`):

```bash
python3 01-EleutherAI-v1/src/eval_runner.py \
    --config 01-EleutherAI-v1/tests/test_config.yaml \
    --phase pretraining \
    --model_args "pretrained=HuggingFaceTB/SmolLM2-135M" \
    --device "cpu" \
    --trial \
    --output_dir 01-EleutherAI-v1/tests/results
```
