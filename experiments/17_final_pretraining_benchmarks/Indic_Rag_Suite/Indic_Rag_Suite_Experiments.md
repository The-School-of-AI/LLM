# Indic-Rag-Suite: Suggested Follow-up Experiments

This document lists concrete follow-up experiments with **exact commands**. Use it after reviewing [Indic_Rag_Suite_README_v1.md](Indic_Rag_Suite_README_v1.md) and [EVALUATION_VS_PAPER.md](EVALUATION_VS_PAPER.md).

---

## 1. Full test-set evaluation (IndicMSMARCO, all languages)

Run retrieval (and optionally generation) on the full test setup, all 13 languages, no sample cap. Output can be used for paper-style reporting (MRR@10 per language).

```bash
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split test --lang all \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --tasks retrieval generation \
  -o results_indicmsmarco_test_all.json
```

Retrieval only (faster):

```bash
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split test --lang all \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --tasks retrieval \
  -o results_indicmsmarco_retrieval_test_all.json
```

---

## 2. Paper-protocol retrieval with per-query pool (MSMARCO-XI)

Build a pool with real non-relevant candidates from MSMARCO-XI, then run retrieval so that MRR is comparable to the paper (per-query candidate set).

**2a. Build pool from MSMARCO-XI (direct mapping, from HuggingFace)**

```bash
python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method direct --output-dir pool_xi_hi --lang hi --from-hf
```

**2b. Run retrieval with that pool**

```bash
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --lang hi --tasks retrieval \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --indicmsmarco-pool pool_xi_hi \
  -o results_paper_protocol_hi.json
```

**2c. Optional: BM25-based pool (requires `pip install rank_bm25`)**

```bash
python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method bm25 --output-dir pool_bm25_hi --lang hi --from-hf --top-k 1000
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --lang hi --tasks retrieval \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --indicmsmarco-pool pool_bm25_hi \
  -o results_paper_protocol_bm25_hi.json
```

---

## 3. Indic-Rag-Suite (18 languages) with BGE-M3 + Gemma for generation

Full RAG benchmark (retrieval + generation) on Indic-Rag-Suite, one language (e.g. Hindi) or all, with a stronger generation model. Requires GPU for Gemma.

```bash
# Hindi only, 20 samples (dev)
python -m benchmark_indic_rag_suite --dataset ai4bharat/Indic-Rag-Suite --split dev --lang hi --max-samples 20 \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --generation-backend hf --generation-model google/gemma-2-1b --device cuda \
  -o results_indic_rag_suite_gemma_hi.json
```

```bash
# All languages, full run (no --max-samples; adjust if dataset has no test split)
python -m benchmark_indic_rag_suite --dataset ai4bharat/Indic-Rag-Suite --split test --lang all \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --generation-backend hf --generation-model google/gemma-2-1b --device cuda \
  -o results_indic_rag_suite_gemma_all.json
```

---

## 4. Compare retrieval backends (BGE-M3 vs small) on IndicMSMARCO Hindi

Same setup as the reported run but with the small retrieval backend to compare MRR.

```bash
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang hi \
  --retrieval-backend small --generation-backend small \
  -o results_indicmsmarco_small_hi.json
```

---

## 5. Additional languages (single-language runs)

Same as the reported BGE-M3 run but for another language (e.g. Tamil, Bengali).

```bash
# Tamil
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang ta \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  -o results_msmarco_bge_m3_ta.json
```

```bash
# Bengali
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split dev --lang bn \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  -o results_msmarco_bge_m3_bn.json
```

---

## 6. Synthetic full pool from HuggingFace (1000 queries, 1000 passages/query)

Build a full-sized pool from the same IndicMSMARCO HF data (negatives = other queries’ gold passages; MRR will stay high) to exercise the paper-protocol pipeline end-to-end.

```bash
python scripts/build_indicmsmarco_pool_from_hf.py --output-dir pool_hi --lang hi
python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --lang hi --tasks retrieval \
  --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
  --indicmsmarco-pool pool_hi \
  -o results_pool_hi.json
```

---

## Notes

- For paper-comparable MRR, use experiments **2a–2c** (per-query pools from MSMARCO-XI).
- For reporting across all IndicMSMARCO languages, use experiment **1**.
- Results from these runs can be summarized in a future results README (e.g. Indic_Rag_Suite_README_v2.md) using the same structure as [Indic_Rag_Suite_README_v1.md](Indic_Rag_Suite_README_v1.md).
