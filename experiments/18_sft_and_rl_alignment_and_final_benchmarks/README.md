# Experiment 18: SFT and RL Alignment — Post-Training Pipeline

Team 18 QLoRA-based SFT pipeline for the 70B MoE model. Covers data preparation,
QLoRA training (SFT, GRPO, DPO, IDFT), and post-training evaluation.

## Folder Structure

```
18_sft_and_rl_alignment_and_final_benchmarks/
├── 01_sft_data/           Data pipeline: source → standardize → decontaminate
│   │                      → chat template → train/val split
│   ├── scripts/           Pipeline scripts (run in order per scripts/README.md)
│   ├── FINAL_DATASETS_AND_PIPELINE.md
│   ├── DATASET_SOURCING_STRATEGY.md
│   ├── DATASET_BENCHMARK_COVERAGE_MATRIX.md
│   ├── CHAT_TEMPLATE.md
│   └── Benchmark-Datasets-V01.xlsx
│
├── 02_sft_training/       Training code: QLoRA SFT/GRPO/DPO/IDFT
│   ├── train_qlora.py     Main training entry point
│   ├── qlora_config.py    Type-safe config (dataclasses + YAML + CLI)
│   ├── idft_trainer.py    IDFT custom SFTTrainer subclass
│   ├── idft_loss.py       IDFT loss implementation (arXiv:2602.12222)
│   ├── default_config.yaml     Default training config
│   ├── idft_smoke_config.yaml  Config for IDFT A/B smoke test
│   └── requirements.txt
│
├── 03_evaluation/         Evaluation: quantization validation + benchmarks
│   ├── validate_quantization.py   End-to-end quantization check (Issue #333)
│   ├── evaluate_smoke_test.py     lm-eval-harness benchmark runner
│   ├── phi_diagnostic.py          IDFT phi distribution go/no-go diagnostic
│   └── run_idft_smoke_test.py     Full IDFT smoke test orchestrator (phases 0-4)
│
└── archive/               Previous contributors' planning artifacts (not active)
    ├── observation_mode/  Observation-mode runbook, sample data, completion report
    └── old_docs/          Verbose reference docs (QLORA_QUANTIZATION_APPROACH.md, etc.)
```

## Pipeline: Data → Train → Evaluate

### Step 1: Prepare SFT Data (`01_sft_data/`)

See `01_sft_data/scripts/README.md` for the full pipeline:

```bash
cd 01_sft_data/scripts
python standardize_conversation_format.py input.jsonl standardized.jsonl --format alpaca
python decontaminate_against_benchmarks.py standardized.jsonl decontaminated.jsonl --benchmark-hashes-dir benchmark_hashes/
python apply_chat_template.py decontaminated.jsonl templated.jsonl --template chatml
python train_val_split.py templated.jsonl --train-out train.jsonl --val-out val.jsonl
python verify_loss_masking.py train.jsonl --tokenizer path/to/model
```

### Step 2: Train (`02_sft_training/`)

```bash
cd 02_sft_training

# SFT with local data (output of Step 1)
python train_qlora.py --local_dataset_path ../01_sft_data/outputs/train.jsonl

# Custom config
python train_qlora.py --config default_config.yaml --method sft

# GRPO alignment
python train_qlora.py --method grpo

# IDFT (on-policy SFT loss, arXiv:2602.12222)
python train_qlora.py --method idft --idft_clip_B 5.0
```

### Step 3: Evaluate (`03_evaluation/`)

```bash
cd 03_evaluation

# Validate quantization works end-to-end (run before training)
python validate_quantization.py --quick

# Full IDFT A/B smoke test (SFT vs IDFT across 3 LRs, 5 benchmarks)
python run_idft_smoke_test.py

# Evaluate a specific checkpoint on benchmarks
python evaluate_smoke_test.py --checkpoint_dir /path/to/checkpoint --label sft --output_json results.json
```

## Hardware Requirements

| Hardware | Quantization | Notes |
|----------|-------------|-------|
| NVIDIA CUDA (Ampere+) | 4-bit NF4 | Full support, recommended |
| NVIDIA CUDA (pre-Ampere) | 8-bit INT8 | Use `--quantization_bits 8` |
| Apple Silicon (MPS) | None (BF16) | Use `--no_quantization` |
| CPU only | None | Use `--no_quantization` |

## Constraints

- LoRA/QLoRA only — no full fine-tuning
- No dataset generation — source from approved datasets only
- No benchmark tuning — benchmarks are for evaluation only
- All runs must be reproducible (seed=42 default)
