# 1B Recurrence Model Training (Optimized Version)

Optimized training pipeline for the **1.47B parameter Dense Recurrence Model** with Triton kernel acceleration, bf16 mixed precision, fused optimizer, and gradient accumulation.

## Optimizations over Base Version

| Feature | Base | Optimized |
|---|---|---|
| **Precision** | float32 | bf16 (auto on CUDA) |
| **Optimizer** | AdamW | Fused AdamW (`fused=True` on CUDA) |
| **Gradient Accumulation** | ✗ | ✓ (`--grad-accum N`) |
| **Triton Kernels** | ✗ | ✓ (RMSNorm, Sinkhorn, Sparse Attn) |
| **FLA Kernels** | ✗ | ✓ (Fused Gated DeltaNet) |
| **Auto Batch Size** | Fixed | Per-GPU memory (A100→8, T4→2) |
| **CLI Arguments** | Hardcoded | Full argparse |
| **CUDA Memory Tracking** | ✗ | ✓ (current + peak per step) |
| **Pin Memory / Workers** | Disabled | Auto-tuned per device |

## Architecture

| Component | Details |
|---|---|
| **Parameters** | ~1.47B (100% dense, no MoE) |
| **Hidden Size** | 4096 |
| **Layers** | 8 (6 DeltaNet + 2 GSA) |
| **Context Target** | 262,144 tokens (YARN RoPE scaling) |
| **Embeddings** | Byte-level Kronecker product (256 × 32 = 8192 dims) |
| **Multi-Token Prediction** | 2 predictions (NTP + MTP) |

## Project Structure

```
endGame/
├── recurrence_model_1b.py      # 1B model architecture (DeltaNet + GSA + Kronecker)
├── train_recurrence_1b.py      # Optimized training script with CLI
├── data_utils.py               # SYNTH dataset streaming with deterministic resume
├── kernels/                    # Triton kernel implementations
├── tokenizer.json              # BPE tokenizer
├── pyproject.toml              # Project dependencies (includes flash-linear-attention)
└── README.md
```

## Prerequisites

- Python ≥ 3.11
- CUDA-capable GPU (recommended: A100/H100 for bf16 + Triton)
- [uv](https://docs.astral.sh/uv/) package manager
- SYNTH dataset downloaded locally at `../synth_local_en`

## Setup

### 1. Install `uv` (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
cd experiments/10_slm_training/optimized_version/endGame

# Create venv and install all dependencies from pyproject.toml
uv venv
uv sync
```

### 3. Download the SYNTH dataset (if not already available)

```bash
uv run python download_mini_synth.py
```

## Training

### Basic run (auto-detects device and optimal settings)

```bash
uv run python train_recurrence_1b.py
```

### CLI arguments

```bash
uv run python train_recurrence_1b.py \
    --seq-length 2048 \
    --batch-size 4 \
    --max-steps 1000 \
```

### All available options

| Argument | Default | Description |
|---|---|---|
| `--seq-length` | `512` | Sequence length (try 2048/4096 for A100) |
| `--batch-size` | Auto | Auto-selected per GPU memory if not set |
| `--max-steps` | `100` | Number of training steps |
| `--lr` | `1e-4` | Learning rate |
| `--grad-accum` | `1` | Gradient accumulation steps |
| `--no-bf16` | `False` | Disable bf16 mixed precision |
| `--dataset-path` | `../synth_local_en` | Path to local SYNTH dataset |
| `--tokenizer` | `tokenizer.json` | Path to tokenizer file |
| `--log-interval` | `1` | Log every N steps |
| `--num-workers` | Auto | DataLoader workers (auto per device) |


### Example Training output

```
step    0 | loss_ntp: 10.8542 | loss_mtp: 10.8327 | aux: 0.0000 | dt: 1234.5ms | tok/s: 828 (avg: 828) | mem: 5.2/8.1 GB
```

- **loss_ntp** — Next-token prediction loss
- **loss_mtp** — Multi-token prediction loss (weighted 0.3×)
- **aux** — Auxiliary loss
- **dt** — Step time in milliseconds
- **tok/s** — Current step throughput (and running average)
- **mem** — CUDA memory: current / peak (GB)

## Dependencies

Defined in `pyproject.toml`:

```
datasets >= 4.5.0
flash-linear-attention >= 0.4.1    # Fused DeltaNet kernels
torch >= 2.10.0
transformers >= 5.1.0
```

> **Note:** `flash-linear-attention` (`fla`) provides fused Triton kernels for the Gated DeltaNet layers. Training will fall back to PyTorch if not available, but with significantly lower throughput on GPU.
