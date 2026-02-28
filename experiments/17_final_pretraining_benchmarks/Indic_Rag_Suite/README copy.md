# Benchmark Indic-Rag-Suite

Unified evaluation harness for **Indic-Rag-Suite** and **IndicMSMARCO**: retrieval metrics (Hit@1, Hit@5, MRR, **MRR@10**, Recall@k, Precision@k, NDCG@k) and generation metrics (Exact Match, Token F1, optional BLEU/ROUGE, optional RAGAS). Supports **dev**, **test**, and **verify** flows with small models or full models (e.g. Gemma-1B).

---

## 1. What harness to use

**Use this repository (`benchmark_indic_rag_suite`) as the harness.** The self-contained evaluation pipeline for:

- **[Indic-Rag-Suite](https://huggingface.co/datasets/ai4bharat/Indic-Rag-Suite)** (18 languages): RAG benchmark with question–paragraph–answer.
- **[IndicMSMARCO](https://huggingface.co/datasets/ai4bharat/IndicMSMARCO)** (13 languages): retrieval benchmark; **official metric is MRR@10** (only ranks 1–10 count).

This harness loads the chosen dataset, runs retrieval and/or generation with your chosen backends, and outputs all metrics in a single JSON.

---

## 2. How to use the harness

### Install (optional)

From the repo root you can run **without** installing, using `python -m` (the scripts set `PYTHONPATH` for you). To get the `benchmark-indic-rag-suite` command on your PATH:

**Using pip:**

```bash
cd /path/to/benchmark_indic_rag_suite_2802
pip install -e .
```

**Using [uv](https://docs.astral.sh/uv/):**

```bash
# Install uv (if needed): curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/benchmark_indic_rag_suite_2802
uv venv                          # create .venv (optional)
source .venv/bin/activate         # Linux/macOS; on Windows: .venv\Scripts\activate
uv pip install -e .
```

After installing with uv, you can run the benchmark with `uv run` without activating the venv:

```bash
uv run benchmark_indic_rag_suite --split dev --lang hi -o results_dev.json
```

### Run the benchmark

**From the repo root** use either:

- **`python -m benchmark_indic_rag_suite ...`** — works without `pip install` (scripts use this).
- **`benchmark-indic-rag-suite ...`** — only after `pip install -e .`.

```bash
# Verify: small models, 10 samples, one language (fast sanity check)
python -m benchmark_indic_rag_suite --split dev --lang hi --max-samples 10 \
  --retrieval-backend small --generation-backend small \
  --tasks retrieval generation -o results_verify.json

# Dev: 20 samples per language, one language
python -m benchmark_indic_rag_suite --split dev --lang hi -o results_dev.json

# Test: full evaluation (Indic-Rag-Suite, all languages)
python -m benchmark_indic_rag_suite --split test --lang all -o results_test.json

# IndicMSMARCO with MRR@10 (paper standard), one language (default = small retrieval model)
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang hi -o results_msmarco_hi.json

# IndicMSMARCO with BGE-M3: retrieval + generation (omit --tasks to run both; use --tasks retrieval for retrieval only)
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang hi \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 -o results_msmarco_bge_m3_hi.json

# Generation with Gemma-1B (GPU)
python -m benchmark_indic_rag_suite --split dev --lang hi --max-samples 20 \
  --generation-backend hf --generation-model google/gemma-2-1b --device cuda \
  -o results_gemma.json

# With extra metrics: F1 (default), BLEU, ROUGE-L, and save per-sample predictions
python -m benchmark_indic_rag_suite --split dev --lang hi --use-bleu --use-rouge --save-predictions -o results_dev.json

# Optional RAGAS evaluation (install ragas separately)
python -m benchmark_indic_rag_suite --split dev --lang hi --generation-evaluator ragas -o results_ragas.json
```

### Convenience scripts

If you get **permission denied**, make scripts executable once:

```bash
chmod +x scripts/*.sh
```

| Script | Purpose |
|--------|--------|
| `scripts/run_verify.sh` | Small models, 10 samples, hi; writes `results_verify.json` |
| `scripts/run_verify_gemma.sh` | Same but generation with Gemma-1B (GPU) |
| `scripts/run_dev.sh` | Dev run; override with `DATASET`, `LANG`, `MAX_SAMPLES`, `OUT` |
| `scripts/run_test.sh` | Test run; override with `DATASET`, `SPLIT`, `LANG`, `OUT`, optionally `MAX_SAMPLES` |

#### Passing different parameters (dataset, language, etc.)

**Option A – Environment variables (with the scripts):**

```bash
# IndicMSMARCO, Hindi, 50 samples
DATASET=ai4bharat/IndicMSMARCO LANG=hi MAX_SAMPLES=50 ./scripts/run_dev.sh

# IndicMSMARCO, all 13 languages, 100 samples per language
DATASET=ai4bharat/IndicMSMARCO LANG=all MAX_SAMPLES=100 OUT=results_msmarco_dev.json ./scripts/run_dev.sh

# Indic-Rag-Suite, Tamil or Bengali
LANG=ta ./scripts/run_dev.sh
LANG=bn MAX_SAMPLES=30 ./scripts/run_dev.sh
```

**Option B – CLI directly:**

```bash
# Switch to IndicMSMARCO
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang hi -o results_msmarco_hi.json

# Change language (e.g. Tamil, Bengali, all)
python -m benchmark_indic_rag_suite --dataset ai4bharat/Indic-Rag-Suite --lang ta -o results_ta.json
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --lang all -o results_msmarco_all.json
```

**Language codes:** `hi`, `ta`, `te`, `bn`, `gu`, `mr`, `ml`, `kn`, `or`, `pa`, `ne`, `as`, `ur`, `en`, etc. Use `all` for every language in the chosen dataset.

### Config file

YAML configs in `configs/`:

- `configs/verify.yaml` – minimal verify run
- `configs/dev.yaml` – dev subset
- `configs/test.yaml` – test, all languages

Under `run` you can set: `use_f1`, `use_squad_normalize`, `use_bleu`, `use_rouge`, `save_predictions`, `generation_evaluator` (`default` | `ragas`), `recall_at_k_list`, `ndcg_at_k_list`.

```bash
python -m benchmark_indic_rag_suite --config configs/verify.yaml -o results_verify.json
```

### Programmatic

```python
from benchmark_indic_rag_suite import run_benchmark, load_config

overrides = {
    "data": {"dataset_name": "ai4bharat/Indic-Rag-Suite", "languages": ["hi"], "split": "dev"},
    "run": {"tasks": ["retrieval", "generation"]},
}
results = run_benchmark(overrides=overrides)
```

---

## 3. Project structure

| Path | Description |
|------|-------------|
| `benchmark_indic_rag_suite/` | Main package |
| `benchmark_indic_rag_suite/config.py` | `DataConfig`, `ModelConfig`, `RunConfig`, `BenchmarkConfig`, `load_config` |
| `benchmark_indic_rag_suite/data/loader.py` | Load Indic-Rag-Suite or IndicMSMARCO from HuggingFace; normalizes to query/passage/answer; supports sharding |
| `benchmark_indic_rag_suite/models/` | `base.py`, `registry.py`, `small.py`, `hf_backends.py` – retrieval and generation backends (small, hf) |
| `benchmark_indic_rag_suite/metrics/` | `retrieval_metrics.py` (Hit@k, MRR, MRR@K, Recall@k, Precision@k, NDCG@k), `generation_metrics.py` (EM, token F1, BLEU, ROUGE-L, SQuAD-style) |
| `benchmark_indic_rag_suite/evaluation/` | `retrieval.py`, `generation.py` – run eval per language; `generation_evaluators.py` – optional RAGAS |
| `benchmark_indic_rag_suite/runner.py` | Orchestrates config, data, models, evaluation, JSON output |
| `benchmark_indic_rag_suite/cli.py` | Argparse → overrides → `run_benchmark` |
| `configs/` | YAML configs for verify, dev, test |
| `scripts/` | Shell scripts for verify, verify+Gemma, dev, test |
| `merge_results.py` | Merge sharded result JSONs (e.g. after `--shard-index i --shard-total N`) |

---

## 4. Metrics

### Retrieval

| Metric | Meaning | Notes |
|--------|---------|-------|
| **Hit@1, Hit@5, Hit@k** | Fraction of queries where the relevant passage is in top k | Configurable k (default: 1, 5, 10, 20) |
| **MRR** | Mean Reciprocal Rank (no cutoff) | |
| **MRR@10** | MRR with cutoff at rank 10; if first relevant is at rank > 10, contribution = 0 | **Official for IndicMSMARCO / MS MARCO** |
| **Recall@1, @5, @10, @20** | Fraction of queries with relevant passage in top k | Set via `--recall-at-k` or config `recall_at_k_list` |
| **Precision@k** | For single relevant doc: 1/k if hit in top-k else 0 | Same k list as Recall |
| **NDCG@5, NDCG@10** | Normalized DCG at k (single relevant doc) | Set via `--ndcg-at-k` or config `ndcg_at_k_list` |

For **IndicMSMARCO** use **MRR@10** for paper-comparable numbers. The **Indic-Rag-Suite paper uses monolingual retrieval**: each language is evaluated separately with a passage pool of that language only (no cross-language distractors). This harness uses **monolingual by default**. The `--paper-retrieval` flag (cross-language pool) is not the paper protocol and is only for optional comparison.

### Generation

| Metric | Meaning |
|--------|---------|
| **Exact Match (EM)** | Fraction of answers matching gold (normalized; optional SQuAD-style with `--use-squad-normalize`) |
| **Token F1** | Token-level F1 (reported by default; disable with `--no-use-f1`) |
| **BLEU** | Sentence BLEU (optional, `--use-bleu`; requires `nltk`) |
| **ROUGE-L** | ROUGE-L F1 (optional, `--use-rouge`; requires `rouge-score`) |
| **RAGAS** | Optional extra evaluator; see [§8. RAGAS](#8-ragas-optional-generation-evaluator). Scores in `generation_ragas` in the output JSON |

---

## 5. Datasets

| Dataset | CLI | Languages | Use |
|---------|-----|-----------|-----|
| **Indic-Rag-Suite** | `--dataset ai4bharat/Indic-Rag-Suite` (default) | 18 | Full RAG benchmark |
| **IndicMSMARCO** | `--dataset ai4bharat/IndicMSMARCO` | 13 | Retrieval benchmark; report MRR@10 |

**Monolingual (paper):** The Indic-Rag-Suite paper evaluates **monolingual** retrieval only (one language at a time, passage pool = that language). This harness uses that by default; do not use `--paper-retrieval` for paper-comparable results.

Indic-Rag-Suite has only a `train` split on HuggingFace; when you pass `--split dev` or `--split test`, the loader still uses `train` and applies `--max-samples` for dev/test-sized subsets.

**IndicMSMARCO format:** The dataset may have multiple rows per query (one per candidate passage). The loader keeps only rows where the passage is the **relevant** one (`is_selected=True` or `relevance_score=1`), so each loaded row is a (query, relevant_passage) pair. Without this filter, MRR can be artificially low (~0.03).

**Paper-comparable retrieval (monolingual):** Use a strong encoder for paper-like numbers:

```bash
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split test --lang hi \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --tasks retrieval -o results_indicmsmarco_bge_m3_hi.json
```

No `--paper-retrieval`; the default is monolingual. The **small** model (MiniLM) is for pipeline verification only and gives lower MRR on Hindi.

---

## 6. Dev, test, and verify

| Flow | When to use | How |
|------|-------------|-----|
| **Verify** | Sanity check that the pipeline and metrics run | Small models, few samples (e.g. 10), one language. `./scripts/run_verify.sh` or `--max-samples 10 --retrieval-backend small --generation-backend small` |
| **Verify with Gemma-1B** | Same, but with a real small LLM for generation | `./scripts/run_verify_gemma.sh` or `--generation-backend hf --generation-model google/gemma-2-1b --device cuda` |
| **Dev** | Iterate on code/config with fast feedback | `--split dev`, one or few languages, optional `--max-samples`. `./scripts/run_dev.sh` |
| **Test** | Full (or capped) evaluation for reporting | `--split test`, `--lang all` or selected langs. For IndicMSMARCO use `--split test` and do not set `--max-samples` for paper-like numbers. `./scripts/run_test.sh` |

---

## 7. Merging sharded results

If you run with `--shard-index i --shard-total N` and get N result files:

```bash
python merge_results.py results_shard_0.json results_shard_1.json ... -o results_merged.json
```

---

## 8. RAGAS (optional generation evaluator)

**Required evaluation for Indic-RAG-Suite:** The benchmark’s **required** evaluation is **retrieval** (e.g. MRR@10, Hit@k, Recall@k) and **generation** (Exact Match and Token F1). These work with **small/local models only** (no OpenAI or other API keys). You can run the full benchmark with `--retrieval-backend small --generation-backend small` and no API keys.

**RAGAS is optional.** It adds extra metrics (e.g. faithfulness, answer relevancy) and is **not** required for the Indic-RAG-Suite protocol. It can be run with **small/local models** (no API key) or with OpenAI/Azure. Results are written under `generation_ragas` in the output JSON.

### RAGAS

- **What it is:** [RAGAS](https://github.com/explodinggradients/ragas) (RAG Assessment) uses LLMs to score RAG outputs without human labels. It typically reports **faithfulness** (is the answer grounded in the context?), **answer relevancy** (does the answer address the question?), and optionally **context precision/recall**.
- **Install:**
  ```bash
  pip install ragas
  # or with uv:
  uv pip install ragas
  ```
  RAGAS may require specific Python and dependency versions; see the [RAGAS docs](https://docs.ragas.io/) if you hit compatibility issues.
- **Local (small) models (no API key):** If **neither** `OPENAI_API_KEY` nor Azure OpenAI env vars are set, the harness uses **local HuggingFace models** for RAGAS (same idea as the custom evaluator / small backends). You need `langchain_community` in addition to `ragas` and `transformers` (e.g. `pip install ragas langchain_community`). Set optional env vars:
  - **`RAGAS_LOCAL_LLM`** – HuggingFace model for the LLM (default: `google/flan-t5-small`).
  - **`RAGAS_LOCAL_EMBEDDING`** – HuggingFace model for embeddings (default: `sentence-transformers/paraphrase-MiniLM-L3-v2`).
  Example (no API keys):
  ```bash
  python -m benchmark_indic_rag_suite --split dev --lang hi --max-samples 10 \
    --retrieval-backend small --generation-backend small --generation-evaluator ragas -o results_ragas_local.json
  ```
- **API key (OpenAI):** If you set **`OPENAI_API_KEY`**, RAGAS will use the default OpenAI backend instead of local models. If the key is missing or invalid, RAGAS will fail and the benchmark will log the error and skip `generation_ragas`. Run with `--log-level DEBUG` to see the full traceback.
- **Azure OpenAI:** To use **Azure OpenAI** instead of OpenAI, set these **environment variables** (e.g. in your shell or a `.env` file) before running the benchmark:

  | Variable | Required | Description |
  |----------|----------|-------------|
  | `AZURE_OPENAI_API_KEY` | Yes | Your Azure OpenAI API key |
  | `AZURE_OPENAI_ENDPOINT` | Yes | Endpoint URL, e.g. `https://<your-resource>.openai.azure.com/` |
  | `AZURE_OPENAI_CHAT_DEPLOYMENT` | No | Chat model deployment name (default: `gpt-4`) |
  | `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | No | Embedding model deployment name (default: `text-embedding-ada-002`) |
  | `AZURE_OPENAI_API_VERSION` | No | API version (default: `2024-02-15-preview`) |

  Install the LangChain Azure integration: `pip install langchain-openai`. When both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` are set, the harness will use Azure for RAGAS and you do **not** need `OPENAI_API_KEY`.

  Example (bash):
  ```bash
  export AZURE_OPENAI_API_KEY="your-azure-key"
  export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
  export AZURE_OPENAI_CHAT_DEPLOYMENT="gpt-4"          # optional
  export AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"  # optional
  python -m benchmark_indic_rag_suite --split dev --lang hi --generation-evaluator ragas -o results_ragas.json
  ```
- **Run with this harness:**
  ```bash
  python -m benchmark_indic_rag_suite --split dev --lang hi --generation-evaluator ragas -o results.json
  ```
  Or in YAML config: set `run.generation_evaluator: ragas`.
- **Output:** Metrics appear under `results["tasks"]["generation_ragas"]` in the JSON (e.g. `faithfulness`, `answer_relevancy`). If RAGAS is not installed or fails, the benchmark still completes and only the default generation metrics are written; a warning is logged.


### Notes

- Use `--generation-evaluator ragas` for extra RAGAS metrics, or `default` for EM/F1 only. The default EM and token F1 are always computed and stored under `tasks.generation`.
- RAGAS receives the **gold passage** as the single context (i.e. it evaluates “given the correct passage, how faithful/relevant is the model’s answer?”). It does **not** evaluate retrieval.
- The benchmark runs without RAGAS installed. If RAGAS is requested but missing or errors, it is skipped and a warning is emitted.
- **Indic-RAG-Suite without APIs:** For evaluation with small models only (no OpenAI), use the default evaluator (EM/F1) or RAGAS without OpenAI/Azure env vars (local HuggingFace).

---

## 9. Optional features and dependencies

- **Token F1** – Always computed by default; use `--no-use-f1` to disable.
- **SQuAD-style normalization** – `--use-squad-normalize` for EM/F1 (removes articles/punctuation).
- **BLEU** – `--use-bleu` (requires `nltk`: `pip install nltk`).
- **ROUGE-L** – `--use-rouge` (requires `rouge-score`: `pip install rouge-score`).
- **RAGAS** – See [§8. RAGAS](#8-ragas-optional-generation-evaluator) for install and usage.
- **Save predictions** – `--save-predictions` writes per-sample (query, passage, prediction, answer) to `predictions.json` (or next to `-o` file as `<stem>_predictions.json`).

## 10. Dependencies

See `pyproject.toml`. Key dependencies: `datasets`, `sentence-transformers`, `transformers`, `scikit-learn`, `tqdm`, `pyyaml`, `numpy`. Optional: `nltk`, `rouge-score`, `ragas` for extra metrics.
