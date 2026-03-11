# 03_evaluation — Evaluation Scripts

Post-training evaluation: quantization validation, benchmark evaluation, and IDFT smoke test.

## Files

| File | Purpose |
|------|---------|
| `validate_quantization.py` | Validates end-to-end quantization support (Issue #333). Run before training to confirm hardware setup. |
| `evaluate_smoke_test.py` | Runs lm-evaluation-harness on a checkpoint. Used in the IDFT smoke test. |
| `phi_diagnostic.py` | Computes phi distribution on base model outputs. Go/no-go gate for IDFT. |
| `run_idft_smoke_test.py` | Full IDFT A/B smoke test orchestrator (phases 0-4). |

## Imports

All evaluation scripts import from `../02_sft_training/` (added to `sys.path` at runtime).
No manual `PYTHONPATH` changes needed — just run from this directory.

## Usage

```bash
cd 03_evaluation

# Validate quantization before training
python validate_quantization.py --quick
python validate_quantization.py --config ../02_sft_training/default_config.yaml

# Run full IDFT smoke test
python run_idft_smoke_test.py

# Evaluate a specific checkpoint
python evaluate_smoke_test.py \
    --checkpoint_dir /path/to/checkpoint \
    --label sft \
    --output_json results_sft.json \
    --use_peft \
    --base_model Qwen/Qwen2.5-7B

# Run phi diagnostic on base model (IDFT go/no-go)
python phi_diagnostic.py --model_name Qwen/Qwen2.5-7B --max_batches 100
```
