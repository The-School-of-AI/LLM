# RULER Benchmark — NVIDIA L4 Setup Guide

> Run the [NVIDIA RULER benchmark](https://github.com/NVIDIA/RULER) on any HuggingFace model on an NVIDIA L4 GPU (24 GB VRAM).  
> **No Docker required** — direct pip install.

---

## Environment Requirements

| Item | Requirement |
|------|------------|
| OS | Ubuntu 22.04+ |
| GPU | NVIDIA L4 (24 GB) |
| CUDA | 12.1+ |
| Python | 3.10+ |
| Disk | ~20 GB free (datasets + model) |

Verify your GPU and CUDA before starting:
```bash
nvidia-smi
nvcc --version
```

---

## Quick-Start (Recommended)

Use the pre-configured wrapper script included in this folder:

```bash
# 1. Clone RULER into this folder
git clone https://github.com/NVIDIA/RULER.git RULER

# 2. Apply all patches (configures model, tasks, paths)
bash patch_configs.sh

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download datasets
cd RULER/scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
cd ../../../../..

# 5. Download the model (adjust MODEL and LOCAL_DIR as needed)
huggingface-cli download google/gemma-2b --local-dir ./models/google/gemma-2b

# 6. Run the benchmark
bash run_ruler.sh
```

---

## Step-by-Step Manual Setup

### Step 1 — Clone RULER

```bash
git clone https://github.com/NVIDIA/RULER.git RULER
cd RULER
```

### Step 2 — Install Python Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install vllm transformers accelerate
pip install nltk rouge_score tqdm pyyaml requests
```

Or use the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 3 — Download Datasets

From inside `RULER/`:
```bash
cd scripts/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
cd ../../../..
```

> Downloads Paul Graham essays (NIAH haystack) and SQuAD + HotpotQA (QA tasks).

### Step 4 — Download Your Model

```bash
# Example: Gemma 2B (fits L4 at all sequence lengths)
huggingface-cli download google/gemma-2b --local-dir ./models/google/gemma-2b

# Example: Gemma 1B
huggingface-cli download google/gemma-1b --local-dir ./models/google/gemma-1b
```

> **Note:** Gemma requires accepting the license on HuggingFace first.  
> Log in first: `huggingface-cli login`

### Step 5 — (Optional) Extend Context Window

Gemma 1B/2B natively supports 8K tokens. To test at 16K/32K, edit the model's `config.json`:

```json
{
  "max_position_embeddings": 32768,
  "rope_scaling": {
    "type": "linear",
    "factor": 4.0
  }
}
```

> Skip if only testing at 4K and 8K.

### Step 6 — Configure Paths in `RULER/scripts/run.sh`

Edit the top of `RULER/scripts/run.sh`:

```bash
GPUS="1"
ROOT_DIR="$(pwd)/results"        # output directory
MODEL_DIR="$(pwd)/models"        # parent dir of model folders
ENGINE_DIR=""                     # leave empty (TensorRT-LLM only)
BATCH_SIZE=32                     # L4 handles 32 comfortably for 1-2B models
```

### Step 7 — Register Your Model in `RULER/scripts/config_models.sh`

Add a new case block inside the `MODEL_SELECT()` function:

```bash
gemma-2b)
    MODEL_PATH="${MODEL_DIR}/google/gemma-2b"
    MODEL_TEMPLATE_TYPE="base"
    MODEL_FRAMEWORK="vllm"
    TOKENIZER_PATH="${MODEL_DIR}/google/gemma-2b"
    TOKENIZER_TYPE="hf"
    ;;

gemma-1b)
    MODEL_PATH="${MODEL_DIR}/google/gemma-1b"
    MODEL_TEMPLATE_TYPE="base"
    MODEL_FRAMEWORK="vllm"
    TOKENIZER_PATH="${MODEL_DIR}/google/gemma-1b"
    TOKENIZER_TYPE="hf"
    ;;
```

For Gemma Instruct variants, use `MODEL_TEMPLATE_TYPE="gemma"` and add to `RULER/scripts/data/template.py`:
```python
"gemma": (
    "<start_of_turn>user\n{context}\n\n{query}<end_of_turn>\n"
    "<start_of_turn>model\n"
),
```

### Step 8 — Configure Sequence Lengths in `RULER/scripts/config_tasks.sh`

Update `SEQ_LENGTHS` and `NUM_SAMPLES`:

```bash
SEQ_LENGTHS=(4096 8192)      # add 16384 32768 if context was extended in Step 5
NUM_SAMPLES=500               # set to 10 for a quick smoke test
```

### Step 9 — Run the Benchmark

```bash
cd RULER/scripts
bash run.sh gemma-2b synthetic
```

Or use the wrapper from this folder:
```bash
bash run_ruler.sh
```

---

## Configuration Reference

All settings in `run_ruler.sh` (wrapper):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `gemma-2b` | Name key used in config_models.sh |
| `HF_MODEL_ID` | `google/gemma-2b` | HuggingFace model ID for download |
| `MODEL_DIR` | `./models` | Where models are downloaded |
| `ROOT_DIR` | `./results` | Where outputs are saved |
| `BATCH_SIZE` | `32` | Inference batch size |
| `SEQ_LENGTHS` | `4096 8192` | Space-separated sequence lengths |
| `NUM_SAMPLES` | `500` | Samples per task (10 for smoke test) |

---

## Troubleshooting

**Port 5000 already in use:**
```bash
lsof -i :5000
kill -9 <PID>
```

**Out of GPU memory:**
```bash
# In run_ruler.sh or run.sh, reduce:
BATCH_SIZE=8
SEQ_LENGTHS=(4096)     # Start small
```

**Gemma model gated / 401 error:**
```bash
huggingface-cli login   # paste your HF token
```

**Model not found:**
- Verify `MODEL_DIR` matches where the model was downloaded
- Check the path: `ls ${MODEL_DIR}/google/gemma-2b/config.json`

**NLTK data missing:**
```python
import nltk; nltk.download('punkt')
```

---

## Expected Runtime on L4

| Sequence Length | Tasks | Samples | Est. Time |
|---|---|---|---|
| 4K | 13 | 500 | ~30 min |
| 8K | 13 | 500 | ~1 hr |
| 16K | 13 | 500 | ~2 hr |
| 32K | 13 | 500 | ~4 hr |

> **Tip:** Run inside `tmux` to survive SSH disconnects:
> ```bash
> tmux new -s ruler
> bash run_ruler.sh
> # Ctrl+B then D to detach
> ```

---

## Output Structure

```
results/
└── <model_name>/
    └── synthetic/
        └── <seq_len>/
            ├── data/
            │   └── validation.jsonl    # generated inputs
            └── pred/
                └── validation.jsonl    # model predictions + scores
```

---

## References

- [RULER GitHub](https://github.com/NVIDIA/RULER)
- [RULER Paper (arXiv:2404.06654)](https://arxiv.org/abs/2404.06654)
- [Gemma on HuggingFace](https://huggingface.co/google/gemma-2b)
