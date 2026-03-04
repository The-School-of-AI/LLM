# RULER Benchmark — NVIDIA L4 Setup Guide

> Run the [NVIDIA RULER benchmark](https://github.com/NVIDIA/RULER) on any HuggingFace model on an NVIDIA L4 GPU (24 GB VRAM).  
> **No Docker required.** The `RULER/` scripts directory is already included in this repo.  
> **Default model:** `google/gemma-3-1b-pt` (1B base pretrained, fits L4 at all seq lengths)

---

## Environment Requirements

| Item | Requirement |
|------|------------|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA L4 (24 GB) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Disk | ~25 GB free (deps + datasets + model) |

```bash
nvidia-smi       # verify GPU
nvcc --version   # verify CUDA
```

---

## Quick-Start

```bash
# From this directory on the L4:
cd ~/LLM/experiments/17_final_pretraining_benchmarks/ruler

# 1. Create and activate virtual environment
uv venv && source .venv/bin/activate

# 2. Install dependencies (uv reads pyproject.toml)
uv sync

# 3. Download datasets (one-time)
cd RULER/scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
cd ../../../../..

# 4. Download model
huggingface-cli login
huggingface-cli download google/gemma-3-1b-pt --local-dir ./models/google/gemma-3-1b-pt

# 5. Patch RULER configs to register Gemma models
bash patch_configs.sh

# 6. Run the benchmark
bash run_ruler.sh gemma-3-1b-pt synthetic
```

---

## Supported Models (registered in `patch_configs.sh`)

| Key for `run_ruler.sh` | HF Model ID | Template |
|---|---|---|
| `gemma-1b` | google/gemma-1b | base |
| `gemma-2b` | google/gemma-2b | base |
| `gemma-3b` | google/gemma-3b | base |
| `gemma-3-1b-pt` | google/gemma-3-1b-pt | base (pretrained) |
| `gemma-1b-it` | google/gemma-1b-it | gemma (instruct) |
| `gemma-2b-it` | google/gemma-2b-it | gemma (instruct) |
| `gemma-3b-it` | google/gemma-3b-it | gemma (instruct) |

To add any other HF model, add a case block to `RULER/scripts/config_models.sh` (see Step 7 below).

---

## Configuration Reference

Edit the top of `run_ruler.sh` to change any setting:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gemma-3-1b-pt` | Key in `config_models.sh` |
| `MODEL_DIR` | `./models` | Parent dir of downloaded models |
| `ROOT_DIR` | `./results` | Where outputs are written |
| `GPUS` | `1` | Number of GPUs (L4 = 1) |
| `BATCH_SIZE` | `32` | Inference batch size |
| `SEQ_LENGTHS` | `(4096 8192)` | Sequence lengths to evaluate |
| `NUM_SAMPLES` | `500` | Samples per task; use `10` for smoke test |

---

## Step-by-Step Manual Setup

### Step 1 — Install Python Dependencies

**Option A — uv (recommended, fast):**
```bash
uv venv && source .venv/bin/activate
uv sync
```

**Option B — pip into conda base (if vllm already installed):**
```bash
pip install transformers accelerate huggingface-hub nltk rouge-score \
            tqdm pyyaml requests pandas>=2.2.0 scipy>=1.13.0 scikit-learn>=1.4.0
```

> **Note:** If using the conda base env, ignore pip conflict warnings about GCP packages (`dataproc-jupyter-plugin`, `bigframes`, etc.) — these don't affect RULER.

### Step 2 — Download Datasets (one-time)

```bash
cd RULER/scripts/data/synthetic/json/
python download_paulgraham_essay.py   # Paul Graham essays (NIAH haystack)
bash download_qa_dataset.sh           # SQuAD + HotpotQA (QA tasks)
cd ../../../../..
```

### Step 3 — Download Model

```bash
huggingface-cli login   # required for gated models (Gemma)
huggingface-cli download google/gemma-3b-it --local-dir ./models/google/gemma-3b-it
```

### Step 4 — (Optional) Extend Context Window

Gemma 3B-IT natively supports 8K tokens. To test at 16K/32K, patch `config.json`:

```json
{
  "max_position_embeddings": 32768,
  "rope_scaling": { "type": "linear", "factor": 4.0 }
}
```

Then add `16384 32768` to `SEQ_LENGTHS` in `run_ruler.sh`.

### Step 5 — Register Gemma in RULER configs

```bash
bash patch_configs.sh
```

This adds gemma-1b/2b/3b and their `-it` variants to `RULER/scripts/config_models.sh`. Safe to re-run — skips already-registered models.

### Step 6 — Add a Custom Model

Add a case block inside `MODEL_SELECT()` in `RULER/scripts/config_models.sh`:

```bash
my-model)
    MODEL_PATH="${MODEL_DIR}/org/my-model"
    MODEL_TEMPLATE_TYPE="base"        # or "gemma", "meta-chat", etc.
    MODEL_FRAMEWORK="vllm"
    TOKENIZER_PATH="${MODEL_DIR}/org/my-model"
    TOKENIZER_TYPE="hf"
    ;;
```

Then run: `bash run_ruler.sh my-model synthetic`

---

## Troubleshooting

**Port 5000 already in use:**
```bash
lsof -i :5000 && kill -9 <PID>
```

**Out of GPU memory:**
```bash
# In run_ruler.sh, reduce:
BATCH_SIZE=8
SEQ_LENGTHS=(4096)
```

**Disk full during install:**
```bash
pip cache purge
rm -rf ~/.cache/uv/
df -h /
```

**Gemma 401 / gated model error:**
```bash
huggingface-cli login
```

**numpy/scipy binary incompatibility (`numpy.dtype size changed`):**
```bash
pip install --force-reinstall --no-cache-dir "pandas>=2.2.0" "scipy>=1.13.0" "scikit-learn>=1.4.0"
```

---

## Expected Runtime on L4

| Sequence Length | Tasks | Samples | Est. Time |
|---|---|---|---|
| 4K | 13 | 500 | ~30 min |
| 8K | 13 | 500 | ~1 hr |
| 16K | 13 | 500 | ~2 hr |
| 32K | 13 | 500 | ~4 hr |

> Run inside `tmux` to survive SSH disconnects:
> ```bash
> tmux new -s ruler
> bash run_ruler.sh gemma-3b-it synthetic
> # Ctrl+B then D to detach
> ```

---

## Output Structure

```
results/
└── gemma-3-1b-pt/
    └── synthetic/
        ├── 4096/
        │   ├── data/   # validation.jsonl (generated inputs)
        │   └── pred/   # validation.jsonl (model predictions + scores)
        └── 8192/
            ├── data/
            └── pred/
```

---

## References

- [RULER GitHub](https://github.com/NVIDIA/RULER)
- [RULER Paper (arXiv:2404.06654)](https://arxiv.org/abs/2404.06654)
- [Gemma 3-1B-PT on HuggingFace](https://huggingface.co/google/gemma-3-1b-pt)
