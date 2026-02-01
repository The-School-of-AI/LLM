# DeepSpeed MoE Training Template

>  DeepSpeed training template for Mixture-of-Experts (MoE) models using GPT-2–style architectures with ZeRO-2 and ZeRO-3 optimization.

## Overview

This template provides a ready-to-use training pipeline that:

- Trains a **GPT-2 language model** with a **Mixture-of-Experts (MoE)** layer replacing one transformer MLP block
- Leverages **DeepSpeed ZeRO-2 / ZeRO-3** to drastically reduce GPU memory consumption
- Supports both **GPU and CPU** training modes
- Logs all metrics to **TensorBoard** in real time
- Operates fully **offline** after a one-time model and dataset download
- Manages dependencies cleanly via **uv + pyproject.toml**

---

## Repository Structure

```
deepspeed_template/
├── assets/
│   └── images/                         # Reference screenshots
│
├── config/
│   └── deepspeed/
│       ├── zero-2-moe.json             # ZeRO-2 + MoE  ← recommended starting point
│       ├── zero-2.json                 # ZeRO-2 (dense model)
│       ├── zero-3-moe.json             # ZeRO-3 + MoE  (advanced)
│       └── zero-3.json                 # ZeRO-3 (dense model)
│
├── src/
│   ├── models/
│   │   ├── moe_gpt2.py                 # GPT-2 model with MoE layer
│   │   └── __init__.py
│   ├── data.py                         # Dataset loading & tokenization
│   ├── moe_utils.py                    # MoE helper utilities
│   ├── train.py                        # Training & evaluation loops
│   ├── utils.py                        # Seeding & misc helpers
│   └── __init__.py
│
├── test/
│   ├── test_moe.py                     # MoE layer unit tests
│   ├── test_training_cpu.py            # CPU training tests
│   ├── test_training_gpu.py            # GPU training tests
│   └── __init__.py
│
├── main.py                             # Entry point — run this file
├── pyproject.toml                      # Project metadata & dependencies
├── requirements.txt                    # Fallback dependency list
└── README.md                           # This file
```

---

## System Requirements

| Requirement | Minimum Version |
|---|---|
| Operating System | Ubuntu 20.04+ |
| Python | 3.11+ |
| GPU | NVIDIA with CUDA support (CPU mode also available) |
| Internet | Required **once** for initial cache download only |

---

## Install uv

`uv` is the **only supported** environment manager for this project.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your shell, then verify the installation:

```bash
uv --version
```

---

## Create & Activate Environment

Run all commands from the **repository root**.

**1. Enter the project directory:**

```bash
cd deepspeed_template
```

**2. Create the virtual environment:**

```bash
uv venv
```

**3. Activate it:**

```bash
source .venv/bin/activate
```

**4. Install all dependencies from `pyproject.toml`:**

```bash
uv pip install -e .
```

This installs: `torch`, `deepspeed`, `transformers`, `datasets`, `pytest`, and all other listed dependencies.

---

## Verify DeepSpeed Installation

```bash
deepspeed --version
```

> ⚠️ **If this command fails, do not continue.** Resolve the installation issue before proceeding.

---

## HuggingFace Cache Setup

This project is designed to run **offline**. You must download the model and dataset **once** and cache them locally.

### Step 1 — Download Model & Dataset (one-time)

```bash
python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

model = "distilgpt2"

AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
load_dataset("wikitext", "wikitext-2-raw-v1")

print("HuggingFace assets downloaded successfully")
EOF
```

### Step 2 — Enable Offline Mode

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

After this point, **no internet connection is required**.

### Step 3 — Verify Cache Exists

```bash
ls ~/.cache/huggingface
```

You should see both directories listed:

```
datasets/
hub/
```

If either is missing, repeat Step 1.

---

## Running Training

### Step 1 — Create a Run ID

This generates a unique, timestamped identifier for your run:

```bash
export RUN_ID=$(date +%Y%m%d_%H%M%S)
```

### Step 2 — Run MoE Training with ZeRO-2

```bash
deepspeed --num_gpus=2 main.py \
  --deepspeed_config config/deepspeed/zero-2-moe.json \
  --num_epochs 1 \
  --batch_size 8 \
  --num_experts 8 \
  --top_k 1 \
  --moe_layer_idx 0 \
  --log_interval 10 \
  --run_name moe8_zero2_${RUN_ID} \
  2>&1 | tee logs/moe_zero2_${RUN_ID}.log
```

---

## Expected Output

**During training:**

```
[train] epoch=0 step=10 loss=...
tok/s=...
mem_alloc=XXXMB
```

**After training completes:**

```
[test] avg_loss=...
[test] ppl=...
```

**Generation sample:**

```
Prompt: Hi
Generated text: ...
```

---

## TensorBoard Monitoring

### Start TensorBoard on the Server

```bash
tensorboard --logdir tb_logs --port 6006 --bind_all
```

### Forward the Port to Your Local Machine

```bash
ssh -i /path/to/key.pem -L 6006:localhost:6006 ubuntu@<SERVER_IP>
```

### Open in Browser

```
http://localhost:6006
```

TensorBoard will display: **training loss**, **learning rate**, **throughput**, and **GPU memory** usage in real time.

---

## Live Monitoring Commands

**GPU usage (refreshes every second):**

```bash
watch -n 1 nvidia-smi
```

**Disk usage:**

```bash
df -h
```

**Follow the training log in real time:**

```bash
tail -f logs/moe_zero2_${RUN_ID}.log
```

---

## Running Tests

**CPU tests:**

```bash
pytest test/test_training_cpu.py
```

**GPU tests:**

```bash
pytest test/test_training_gpu.py
```

**MoE layer tests:**

```bash
pytest test/test_moe.py
```
