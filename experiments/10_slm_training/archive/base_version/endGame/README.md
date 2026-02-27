# 1B Recurrence Model Training (Base Version)

Training pipeline for the **1.47B parameter Dense Recurrence Model** with Hybrid Gated DeltaNet + Gated Sparse Attention (GSA) and Kronecker embeddings.

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
├── train_recurrence_1b.py      # Training script
├── data_utils.py               # SYNTH dataset streaming with deterministic resume
├── tokenizer.json              # BPE tokenizer
├── pyproject.toml              # Project dependencies
└── README.md
```

## Prerequisites

- Python ≥ 3.11
- CUDA-capable GPU (or Apple Silicon with MPS)
- [uv](https://docs.astral.sh/uv/) package manager
- SYNTH dataset downloaded locally at `../synth_local_en`

## Setup

### 1. Install `uv` (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create virtual environment and install dependencies

```bash
cd experiments/10_slm_training/base_version/endGame

# Create venv and install all dependencies from pyproject.toml
uv venv
uv sync
```

### 3. Download the SYNTH dataset (if not already available)

```bash
uv run python download_mini_synth.py
```

## Training

### Run training

```bash
uv run python train_recurrence_1b.py
```

The training script will automatically:
1. Load the tokenizer from `tokenizer.json`
2. Setup Kronecker embeddings (256 × 32 = 8192 dims)
3. Create the 1B model and move to the best available device (CUDA → MPS → CPU)
4. Stream the SYNTH dataset with deterministic resume
5. Run the training loop (100 steps by default)

### Training configuration

Key parameters can be modified directly in `train_recurrence_1b.py` → `main()`:

| Parameter | Default | Location |
|---|---|---|
| `seq_len` | 1024 | `SYNTHStream(seq_len=...)` |
| `batch_size` | 1 | `SYNTHStream(batch_size=...)` and `DataLoader(batch_size=...)` |
| `num_steps` | 100 | `simple_training_loop(num_steps=...)` |
| `lr` | 1e-4 | `simple_training_loop` → `AdamW(lr=...)` |
| `grad_clip` | 1.0 | `clip_grad_norm_(max_norm=...)` |

### Training output

Each step logs:
```
step   0 | loss_ntp: 10.8542 | loss_mtp: 10.8327 | aux: 0.0000 | dt: 1234.5ms | tok/sec: 828.3
```

- **loss_ntp** — Next-token prediction loss
- **loss_mtp** — Multi-token prediction loss (weighted 0.3×)
- **aux** — Auxiliary loss
- **dt** — Step time in milliseconds
- **tok/sec** — Training throughput (tokens per second)

## Dependencies

Defined in `pyproject.toml`:

```
datasets >= 4.5.0
torch >= 2.10.0
transformers >= 5.1.0
```
