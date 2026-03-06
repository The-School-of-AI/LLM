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
