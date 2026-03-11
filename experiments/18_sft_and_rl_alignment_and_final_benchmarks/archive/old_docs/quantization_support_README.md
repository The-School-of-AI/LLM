# Team 18: SFT, RL-Style Alignment & Final Post-Training Benchmarks

This folder contains the official implementation for Team 18's post-training pipeline, focusing on QLoRA-based SFT and RL alignment with full quantization support.

## Overview

| Component | Description |
|-----------|-------------|
| **Category** | Model (Post-Training / Alignment) |
| **Objective** | Extract benchmark-worthy capability gains via instruction tuning and alignment |
| **Method** | LoRA / QLoRA-only (no full fine-tuning) |
| **Focus** | Quantization support end-to-end (Issue #333) |

## Project Structure

```
team18/
├── README.md                           # This file
├── QLORA_QUANTIZATION_APPROACH.md      # Comprehensive quantization approach document
├── default_config.yaml                 # Default configuration file
├── qlora_config.py                     # Type-safe configuration with dataclasses
├── train_qlora.py                      # Main training script (SFT/GRPO/DPO)
├── validate_quantization.py            # End-to-end quantization validation
└── requirements.txt                    # Python dependencies
```

## Quick Start

### 1. Installation

```bash
cd team18
pip install -r requirements.txt
```

### 2. Validate Quantization Support (Issue #333)

Before training, verify that quantization works end-to-end on your hardware:

```bash
# Full validation suite
python validate_quantization.py --config default_config.yaml

# Quick validation (memory + inference only)
python validate_quantization.py --quick
```

### 3. Run Training

```bash
# Basic SFT training with default config
python train_qlora.py

# Custom configuration
python train_qlora.py --config my_config.yaml

# Override specific parameters
python train_qlora.py \
    --model_name "microsoft/phi-2" \
    --quantization_bits 4 \
    --lora_r 64 \
    --method sft

# GRPO training (RL-style alignment)
python train_qlora.py --method grpo --num_generations 4

# Disable quantization (for Apple Silicon MPS)
python train_qlora.py --no_quantization
```

## Configuration

The pipeline uses a layered configuration system:

1. **Defaults** (`default_config.yaml`) - Base configuration
2. **Custom YAML** (`--config`) - Override defaults
3. **CLI Arguments** - Override everything

See `default_config.yaml` for all available options.

## Hardware Support

| Hardware | Quantization | Notes |
|----------|--------------|-------|
| NVIDIA CUDA (Ampere+) | 4-bit NF4 | Full support, recommended |
| NVIDIA CUDA (Pre-Ampere) | 8-bit INT8 | Limited 4-bit support |
| Apple Silicon (MPS) | None (BF16) | bitsandbytes not supported |
| Google Colab T4 | 4-bit NF4 | Works well with double quant |
| CPU Only | None | Use GGUF for inference |

## Key Documents

- **[QLORA_QUANTIZATION_APPROACH.md](QLORA_QUANTIZATION_APPROACH.md)** - Detailed approach document covering:
  - Quantization formats (NF4, FP4, INT8)
  - Layer-specific strategies
  - Training pipeline stages
  - Hardware considerations
  - Validation checklist

## Team Responsibilities

Per the team charter:

1. **Instruction tuning (SFT)** - Using approved datasets only
2. **Safety alignment** - Where applicable
3. **Benchmark-aligned post-training** - No benchmark gaming
4. **LoRA/QLoRA adapter training** - No full fine-tuning
5. **Final aligned model artifact** - Production-grade

## Constraints (Non-Negotiable)

- **LoRA-only or QLoRA-only** - Full-weight fine-tuning is disallowed
- **No dataset generation** - Team 18 does not create datasets
- **No benchmark tuning** - Run benchmarks for evaluation only
- **Reproducibility required** - Everything must be logged and reproducible

## Dependencies

- Teams 12, 5, 16, 17, 14 (blocking prerequisites)
- Downstream: Team 17 (final deliverable), Team 20 (results/publication)

## Timeline

| Phase | Dates | Activities |
|-------|-------|------------|
| Setup/Preparation | Feb 9-10 | SOTA study, observation mode, pipeline testing |
| Peak Execution | Feb 10-19 | SFT + alignment, stability validation, benchmark execution |
| Monitor/Support | Feb 20-23 | Final polish, regression checks, documentation |
| Done/Frozen | Feb 23 | Aligned model locked, results frozen |

## References

- [QLoRA Paper](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
- [DeepSeekMath GRPO](https://arxiv.org/abs/2402.03300)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
