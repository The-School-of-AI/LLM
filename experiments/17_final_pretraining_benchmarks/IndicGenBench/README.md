# IndicGenBench Benchmark

Evaluation harness for [IndicGenBench](https://arxiv.org/abs/2404.16816) (Google Research, ACL 2024) — a multilingual benchmark for evaluating LLM generation capabilities across **29 Indic languages**, **13 scripts**, and **4 language families**.

There is no official evaluation harness for this benchmark. This package implements the evaluation protocol described in the paper, loading data directly from the [official HuggingFace datasets](https://huggingface.co/collections/google/indicgenbench).

## Overview

### Tasks & Metrics

| Task | Description | Input | Output | Metrics |
|------|-------------|-------|--------|---------|
| **CrossSum-IN** | Cross-lingual summarization | English article | Summary in Indic language | ROUGE-1/2/L, METEOR |
| **Flores-IN** | Machine translation | English sentence | Translation in Indic language | BLEU, chrF, METEOR |
| **XQuAD-IN** | Extractive QA | Indic question + Indic passage | Answer span | Exact Match, Token F1 |
| **XorQA-IN** | Cross-lingual QA | Indic question + English passage | Answer span | Exact Match, Token F1 |

### Dataset Scale

| Task | Languages | Train | Dev | Test |
|------|-----------|-------|-----|------|
| CrossSum-IN | 29 | 2.9K | 2.9K | 14.5K |
| Flores-IN | 29 | -- | 28.9K | 29.3K |
| XQuAD-IN | 12 | 1.2K | 1.2K | 14.2K |
| XorQA-IN | 28 | 2.8K | 14K | 15.1K |

### Supported Languages

Assamese, Awadhi, Bengali, Bhojpuri, Bodo, Chhattisgarhi, Garhwali, Gujarati, Haryanvi, Hindi, Kannada, Konkani, Maithili, Malayalam, Malvi, Manipuri, Marathi, Marwari, Nepali, Odia, Pashto, Punjabi, Rajasthani, Sanskrit, Santali, Tamil, Telugu, Tibetan, Urdu.

> XQuAD-IN covers a subset of 12 languages: Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, Urdu.

## Quick Start

```bash
cd experiments/17_final_pretraining_benchmarks/IndicGenBench

# Install (use a virtual environment)
uv venv .venv && source .venv/bin/activate
uv pip install -e .

# 1. Smoke test — dummy model, no GPU, verifies pipeline end-to-end
./scripts/run_verify.sh

# 2. Dev run — real model, small subset
MODEL=Qwen/Qwen2.5-0.5B-Instruct DEVICE=mps ./scripts/run_dev.sh

# 3. Full evaluation — all languages, test split (requires GPU)
MODEL=google/gemma-3-1b-it DEVICE=cuda ./scripts/run_test.sh
```

> For gated models (Gemma, Llama, etc.), log in first: `huggingface-cli login`

## CLI Reference

```bash
python -m benchmark_indicgenbench [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | None | Path to YAML config file |
| `--split` | `dev` | Dataset split: `dev` or `test` |
| `--lang` | `hi` | Language code (e.g. `hi`, `bn`, `ta`) or `all` |
| `--max-samples` | 20 (dev) / all (test) | Max samples per language |
| `--tasks` | all 4 | Tasks to run: `crosssum`, `flores`, `xquad`, `xorqa` |
| `--model-backend` | `small` | `small` (dummy) or `hf` (HuggingFace) |
| `--model-name` | None | HuggingFace model ID (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) |
| `--device` | `cpu` | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--max-new-tokens` | 128 | Max tokens to generate per sample |
| `-o` / `--output` | None | Output JSON file path |
| `--seed` | 42 | Random seed |

### Examples

```bash
# Single task, single language
python -m benchmark_indicgenbench \
  --tasks xquad \
  --lang hi \
  --model-backend hf \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --max-samples 50 \
  -o results_xquad_hi.json

# All tasks, multiple runs via config
python -m benchmark_indicgenbench --config configs/dev.yaml -o results.json

# Override config values via CLI
python -m benchmark_indicgenbench \
  --config configs/test.yaml \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --lang bn \
  -o results_bn.json
```

## Configs

Three preset configurations for different stages of development:

| Config | Split | Languages | Samples | Model | Purpose |
|--------|-------|-----------|---------|-------|---------|
| `configs/verify.yaml` | dev | hi | 5 | dummy | Smoke test, CI |
| `configs/dev.yaml` | dev | hi, bn | 20 | hf | Development, debugging |
| `configs/test.yaml` | test | all 29 | all | hf | Full evaluation |

## Output Format

Results are written as JSON:

```json
{
  "config": {
    "split": "dev",
    "languages": ["hi"],
    "max_samples_per_lang": 5,
    "backend": "hf",
    "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
    "max_new_tokens": 64,
    "seed": 42
  },
  "tasks": {
    "crosssum": {
      "hi": { "rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.031, "n": 5.0 }
    },
    "flores": {
      "hi": { "bleu": 0.48, "chrf": 12.55, "meteor": 0.048, "n": 5.0 }
    },
    "xquad": {
      "hi": { "exact_match": 0.40, "token_f1": 0.44, "n": 5.0 }
    },
    "xorqa": {
      "hi": { "exact_match": 0.20, "token_f1": 0.20, "n": 5.0 }
    }
  }
}
```

## Architecture

```
benchmark_indicgenbench/
├── cli.py                  # Argparse CLI with YAML config override
├── config.py               # Dataclass-based config (DataConfig, ModelConfig, RunConfig)
├── runner.py               # Orchestrator: load data -> run tasks -> aggregate -> output
├── data/
│   └── loader.py           # HuggingFace dataset loading (datasets lib + direct JSON)
├── tasks/
│   ├── crosssum.py         # CrossSum-IN evaluator
│   ├── flores.py           # Flores-IN evaluator
│   ├── xquad.py            # XQuAD-IN evaluator
│   └── xorqa.py            # XorQA-IN evaluator
├── metrics/
│   ├── qa.py               # Exact Match, Token F1
│   ├── translation.py      # BLEU, chrF, METEOR (sacrebleu)
│   └── summarization.py    # ROUGE-1/2/L, METEOR (rouge-score)
└── models/
    ├── base.py             # Abstract GenerationModelBase
    ├── registry.py         # Factory pattern with @register_model decorator
    ├── hf_backend.py       # HuggingFace AutoModelForCausalLM backend
    └── small.py            # Dummy model for smoke testing
```

**Design choices:**
- **No official harness exists** for IndicGenBench — this is a custom implementation following the paper's evaluation protocol.
- **Data loading**: CrossSum and Flores load via `datasets` library. XQuAD and XorQA download JSON directly from HuggingFace Hub (the `datasets` library has schema casting issues with these).
- **Model abstraction**: Pluggable backends via a registry. `small` backend enables pipeline testing without downloading model weights. `hf` backend wraps any HuggingFace causal LM.
- **Config layering**: YAML file + CLI overrides via deep merge, following the same pattern as the Indic-Rag-Suite benchmark in this repo.

## Dependencies

| Package | Version | Used for |
|---------|---------|----------|
| `datasets` | >= 2.14 | Loading CrossSum, Flores from HuggingFace |
| `huggingface_hub` | >= 0.17 | Direct JSON download for XQuAD, XorQA |
| `torch` | >= 1.13 | Model inference |
| `transformers` | >= 4.30 | AutoModelForCausalLM, tokenizer |
| `sacrebleu` | >= 2.3 | BLEU, chrF (corpus-level) |
| `rouge-score` | >= 0.1.2 | ROUGE-1/2/L |
| `nltk` | >= 3.8 | METEOR |
| `PyYAML` | >= 6.0 | Config loading |

## Data Sources

| Dataset | HuggingFace ID |
|---------|----------------|
| CrossSum-IN | [`google/IndicGenBench_crosssum_in`](https://huggingface.co/datasets/google/IndicGenBench_crosssum_in) |
| Flores-IN | [`google/IndicGenBench_flores_in`](https://huggingface.co/datasets/google/IndicGenBench_flores_in) |
| XQuAD-IN | [`google/IndicGenBench_xquad_in`](https://huggingface.co/datasets/google/IndicGenBench_xquad_in) |
| XorQA-IN | [`google/IndicGenBench_xorqa_in`](https://huggingface.co/datasets/google/IndicGenBench_xorqa_in) |

## References

- **Paper**: [IndicGenBench: A Multilingual Benchmark to Evaluate Generation Capabilities of LLMs on Indic Languages](https://arxiv.org/abs/2404.16816) (ACL 2024)
- **Official data**: [google-research-datasets/indic-gen-bench](https://github.com/google-research-datasets/indic-gen-bench)
