# Benchmark Indic-Rag-Suite

Unified evaluation harness for **Indic-Rag-Suite** and **IndicMSMARCO**: retrieval metrics (Hit@1, MRR, **MRR@10**, Recall@k, NDCG@10) and generation metrics (Exact Match, optional Token F1). Supports **dev**, **test**, and **verify** flows with small models or full models (e.g. Gemma-1B).

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

```bash
cd benchmark_indic_rag_suite
pip install -e .
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
| `benchmark_indic_rag_suite/metrics/` | `retrieval_metrics.py` (Hit@1, MRR, MRR@K, Recall@k, NDCG@k), `generation_metrics.py` (EM, token F1) |
| `benchmark_indic_rag_suite/evaluation/` | `retrieval.py`, `generation.py` – run eval per language |
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
| **Hit@1** | Fraction of queries where the relevant passage is rank 1 | |
| **MRR** | Mean Reciprocal Rank (no cutoff) | |
| **MRR@10** | MRR with cutoff at rank 10; if first relevant is at rank > 10, contribution = 0 | **Official for IndicMSMARCO / MS MARCO** |
| **Recall@1, @5, @10** | Fraction of queries with relevant passage in top k | |
| **NDCG@10** | Normalized DCG at 10 (single relevant doc) | |

For **IndicMSMARCO** use **MRR@10** for paper-comparable numbers. The **Indic-Rag-Suite paper uses monolingual retrieval**: each language is evaluated separately with a passage pool of that language only (no cross-language distractors). This harness uses **monolingual by default**. The `--paper-retrieval` flag (cross-language pool) is not the paper protocol and is only for optional comparison.

### Generation

| Metric | Meaning |
|--------|---------|
| **Exact Match (EM)** | Fraction of answers matching gold (normalized) |
| **Token F1** | Token-level F1 (optional) |

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

## 8. Dependencies

See `pyproject.toml`. Key dependencies: `datasets`, `sentence-transformers`, `transformers`, `scikit-learn`, `tqdm`, `pyyaml`, `numpy`.
