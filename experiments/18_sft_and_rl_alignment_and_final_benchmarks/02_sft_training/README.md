# 02_sft_training — QLoRA Training

QLoRA-based training for SFT, GRPO, DPO, and IDFT methods.

## Files

| File | Purpose |
|------|---------|
| `train_qlora.py` | Main training entry point. Handles all four methods. |
| `qlora_config.py` | Type-safe config (dataclasses). Loads from YAML + CLI overrides. |
| `idft_trainer.py` | Custom `SFTTrainer` subclass that uses IDFT loss. |
| `idft_loss.py` | IDFT reweighted loss from arXiv:2602.12222. |
| `default_config.yaml` | Default training config (used when no `--config` is given). |
| `idft_smoke_config.yaml` | Config for the IDFT A/B smoke test (SFT vs IDFT). |
| `compare_generations.py` | Standalone A/B tool to compare base vs SFT checkpoint. |
| `requirements.txt` | Python dependencies. |

## Quick Start

```bash
pip install -r requirements.txt

# SFT with local data
python train_qlora.py --local_dataset_path ../01_sft_data/outputs/train.jsonl

# GRPO alignment
python train_qlora.py --method grpo

# IDFT (on-policy SFT loss)
python train_qlora.py --method idft --idft_clip_B 5.0

# Disable quantization (Apple Silicon / CPU)
python train_qlora.py --no_quantization

# Side-by-Side Generation Comparison
# IMPORTANT: Ensure 'base_model_path' and 'adapter_path' are set in default_config.yaml
python compare_generations.py --config default_config.yaml
```

## Generation & Comparison

The pipeline includes an automated `GenerationComparisonCallback` that runs during training. To use the standalone comparison tool or the automated callback:

1.  Open `default_config.yaml`.
2.  Under the `generation:` section, ensure:
    - `base_model_path` points to your base model (e.g., `"microsoft/phi-2"`).
    - `adapter_path` (optional for training, required for standalone) points to your LoRA checkpoint.
    - Your `prompts` list is populated.

## Config Priority

CLI args > custom `--config` YAML > `default_config.yaml`

All important hyperparameters (model name, LR, LoRA rank, quantization bits, etc.)
can be overridden from the command line. See `python train_qlora.py --help`.
