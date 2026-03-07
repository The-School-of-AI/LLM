# OLMES Evaluation Runner

A single shell script that clones [OLMES](https://github.com/allenai/olmes), sets up the environment with GPU support, and runs LLM evaluation tasks — storing results locally or to S3.

---

## Prerequisites

| Tool | Notes |
|------|-------|
| `git` | To clone OLMES |
| `python` + `pip` | uv is auto-installed if missing |
| NVIDIA GPU + CUDA | Required for vLLM backend (default) |
| Huggingface Token | Required for models requiring permission. export HF_TOKEN=<YOUR TOKEN> |
| AWS CLI (optional) | Only needed if using S3 output (`aws configure`) |

---

## Quick Start

```bash
chmod +x run_eval.sh
./run_eval.sh
```

This runs with all defaults:
- **Model**: `google/gemma-3-1b-it` (fetched from HuggingFace)
- **Tasks**: `mmlu:mc::olmes`, `mmlu_pro:mc::none`, `triviaqa::olmes`, `arc_challenge::olmes`
- **Output**: `./results/`

---

## Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-m, --model` | `MODEL` | `google/gemma-3-1b-it` | HuggingFace model ID or `s3://` path |
| `-t, --tasks` | `TASKS` | *(four tasks above)* | Space-separated OLMES task names |
| `-o, --output-dir` | `OUTPUT_DIR` | `./results` | Local directory for results |
| `-r, --remote-output` | `REMOTE_OUTPUT_DIR` | *(none)* | S3 URI for remote copy of results |
| `-a, --model-args` | `MODEL_ARGS` | *(none)* | JSON string of extra model args |
| `--model-type` | `MODEL_TYPE` | `vllm` | Backend: `vllm`, `hf`, or `litellm` |
| `--limit` | `LIMIT` | *(none)* | Max instances per task (for quick tests) |
| `--dry-run` | `DRY_RUN=true` | `false` | Print command without running |
| `-h, --help` | — | — | Show usage |

Options can be passed as flags **or** environment variables interchangeably.

---

## Examples

### 1. Default run (HuggingFace model, local output)

```bash
./run_eval.sh
```

### 2. Different HuggingFace model

```bash
./run_eval.sh --model meta-llama/Llama-3.2-1B-Instruct --output-dir ./llama-results
```

### 3. Model from S3

Models stored on S3 (e.g., fine-tuned checkpoints) are passed directly as the model path. OLMES passes this to vLLM which can load from S3 if the AWS credentials are configured.

```bash
./run_eval.sh \
  --model s3://my-bucket/models/my-finetuned-model \
  --output-dir ./results
```

> **Note**: Ensure `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_DEFAULT_REGION` are set (or use an IAM instance role).

### 4. Results to S3

```bash
./run_eval.sh \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output-dir ./results \
  --remote-output s3://my-bucket/eval-results/qwen-run1
```

Results are saved locally first, then OLMES copies them to S3 via `--remote-output-dir`.

### 5. Model from S3 + results to S3

```bash
export MODEL="s3://my-bucket/models/my-model"
export REMOTE_OUTPUT_DIR="s3://my-bucket/eval-results/my-model-run"
export OUTPUT_DIR="./results"

./run_eval.sh
```

### 6. Custom task list

Task names follow the OLMES format: `<benchmark>:<format>::<fewshot_config>`

```bash
./run_eval.sh --tasks "hellaswag::olmes winogrande::olmes piqa::olmes"
```

### 7. Quick test with instance limit

```bash
./run_eval.sh --limit 10 --dry-run   # preview command
./run_eval.sh --limit 10             # run with 10 instances per task
```

### 8. Using `hf` backend (no GPU / CPU-only)

```bash
./run_eval.sh --model-type hf --model google/gemma-3-1b-it
```

### 9. Model requiring trust_remote_code

```bash
./run_eval.sh \
  --model mistralai/Mistral-7B-v0.1 \
  --model-args '{"trust_remote_code": true}'
```

---

## How It Works

```
run_eval.sh
  │
  ├── 1. Clone or pull allenai/olmes into ./olmes/
  │
  ├── 2. Install uv (if missing), then:
  │        uv sync --group gpu
  │      (installs vLLM + all OLMES dependencies)
  │
  ├── 3. Build olmes command from config
  │
  └── 4. Execute:
           uv run olmes \
             --model <model> \
             --model-type <backend> \
             --task <task1> --task <task2> ... \
             --output-dir <local-dir> \
             [--remote-output-dir <s3-uri>]
```

---

## Output Structure

```
results/
  <task-name>/
    metrics.json        # aggregated scores
    predictions.jsonl   # per-instance outputs
    config.json         # run configuration
```

---

## Default Tasks Reference

| Task | Format | Few-shot config |
|------|--------|----------------|
| `mmlu:mc::olmes` | Multiple choice | OLMES standard |
| `mmlu_pro:mc::none` | Multiple choice | Zero-shot |
| `triviaqa::olmes` | Open generation | OLMES standard |
| `arc_challenge::olmes` | Multiple choice | OLMES standard |
| `gsm8k::olmes` | Open generation (CoT) | OLMES standard |
| `minerva_math_500::olmo3:midtrain` | Open generation (CoT) | OLMo3 midtrain |
| `truthfulqa::olmo1` | Multiple choice | OLMo1 |
| `bbh:qa::none` | QA (open generation) | Zero-shot |

---

## Troubleshooting

**`uv` not found after auto-install**
> Re-run the script; pip installs to the user path which may require a shell reload.

**CUDA out of memory**
> Use `--model-type hf` for a lower-memory backend, or add `--model-args '{"gpu_memory_utilization": 0.8}'` for vLLM.

**S3 access denied**
> Run `aws sts get-caller-identity` to verify credentials are configured correctly.

**Task not found**
> Run `uv run olmes --list-tasks` inside the `olmes/` directory to see all available task names.


---

## Running OLMES on a Custom TSAI Model

In addition to running evaluations on HuggingFace-hosted models, this runner can also evaluate **locally trained models** (e.g., models trained with DeepSpeed).

This section documents the steps required to run OLMES on the TSAI model used in our experiments.

---

### 1. Model Conversion Pipeline

Training produced a **DeepSpeed ZeRO checkpoint**, which must be converted to a format that HuggingFace can load.

```
DeepSpeed ZeRO checkpoint
        ↓
zero_to_fp32.py (optional)
        ↓
pytorch_model.bin / model.safetensors
        ↓
HuggingFace model directory
        ↓
OLMES evaluation
```

If the checkpoint is already consolidated, the conversion step is unnecessary.

Example conversion:

```bash
python zero_to_fp32.py checkpoint_dir pytorch_model.bin
```

If a consolidated checkpoint file is given, convert it to a .bin or a .safetensors file using the convert_model.py file.       

---

### 2. HuggingFace Model Directory

Create a directory containing:

```
tsai_model/
  pytorch_model.bin  (or model.safetensors)
  config.json
  modeling_tsai.py

  recurrence_model_1b.py
  reversible_ops_midpoint.py
  liger_ops.py

  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
```

All model code must live in the same directory so that HuggingFace can load it with:

```
trust_remote_code=True
```

---

### 3. Code Fixes - 
#### ONLY for testing if benchmarking scripts are working properly on our checkpoint files and tokenizer


The original training code relied on internal dependencies and custom kernels that are not present in the evaluation environment. Several small patches were required.

---

#### 3.1. Profiler Fallback

The model references an internal profiler module:
```llm.profiler.time_region```  
Add a fallback implementation.

---

#### 3.2. Fix Relative Imports

Dynamic module loading in HuggingFace does not support relative imports.

Change imports in `recurrence_model_1b.py`.

Example:
```
from .reversible_ops_midpoint import ReversibleMidpointStack
..
from .liger_ops import ...
```

---

#### 3.3. Kernel Dependencies

The model expects several custom kernels.
Install required dependencies:

```bash
pip install triton
pip install fla
```

These provide kernels required for:
* DeltaNet attention
* Flash linear attention

---

#### 3.4. Sinkhorn Fallback

If Triton kernels are unavailable, add a PyTorch fallback implementation of Sinkhorn normalization.

---

#### 3.5. Sparse Attention Fallback

The Gated Sparse Attention (GSA) block originally requires Triton kernels.
During evaluation we allow a fallback to dense attention. This allows inference to proceed without sparse kernels.

---

#### 3.6. Indexer Kernel Fallback

The fused routing kernel (`fused_indexer_topk`) may also be unavailable.
Add a simple fallback that returns placeholder routing indices.
This bypasses sparse routing during benchmarking.

---

**Note**:
HuggingFace dynamically copies model code into a cache directory.
After modifying model files, clear the cache:

```bash
rm -rf ~/.cache/huggingface/modules
```

Otherwise HuggingFace will continue using stale versions of the code.

---

### 4. Validate Model Loading

Before running benchmarks, verify that the model loads correctly.

Example test script:

```
python test_model.py
```

Recommended checks:

* Model loads successfully
* Tokenizer loads successfully
* Forward pass produces logits
* Logits have reasonable statistics

Example:

```
logits mean ≈ 0
logits std  ≈ 1–10
```

Also verify that most parameters loaded from the checkpoint:

```
fraction loaded ≈ 0.99+
```

---

### 5. Running OLMES with the TSAI Model

Once the model loads correctly, run the evaluation:

```bash
./run_eval.sh \
 --model tsai_model \
 --model-type hf \
 --model-args '{"trust_remote_code": true}'
```

---

### 6. Quick Test Run

To test the pipeline before running the full benchmark suite:

```bash
./run_eval.sh \
 --model tsai_model \
 --model-type hf \
 --limit 20
```

This evaluates a small subset of instances for each task.

---

### 7. Summary

Key steps required to run OLMES on the TSAI model:

* Convert DeepSpeed checkpoint to HuggingFace format
* Package model code and tokenizer in a single directory
* Patch missing internal dependencies
* Add fallbacks for unavailable kernels
* Verify model loading before running benchmarks

Once these steps are completed, the TSAI model runs successfully under the OLMES evaluation harness.
