# Why Our IndicMSMARCO Scores Are Higher Than the Paper

## Dataset finding: HuggingFace has only 1 row per query

Inspection of `ai4bharat/IndicMSMARCO` (e.g. Hindi config) shows:

- **Total rows:** 1000 per language  
- **Unique query_ids:** 1000  
- **Rows per query_id:** 1 (min=1, max=1)  
- Each row has `relevance_score=1.0` and `is_selected=True`.

So the public dataset does **not** include multiple candidate passages per query; it only has the single relevant (query, passage) pair per query. Paper-style evaluation (many candidates per query) cannot be reproduced from this dataset alone without an external candidate pool.

---

## How the paper evaluates (Table 2, [arXiv:2506.01615](https://arxiv.org/abs/2506.01615))

From the **IndicRAGSuite** paper:

1. **Benchmark:** IndicMSMARCO — 1,000 queries from the MS MARCO development set, translated into 13 Indian languages, with human-verified translations.
2. **Metric:** **MRR (Mean Reciprocal Rank)** — same as standard MS MARCO passage ranking.
3. **Task:** For each query, the model ranks a **set of candidate passages**; MRR is 1/(rank of the relevant passage), averaged over queries.
4. **Reported scores (Table 2):** BGE-M3 on Hindi = **0.52** (52%), Multilingual e5-large on Hindi = **0.52**. Scores across languages range from ~0.30 (Assamese) to ~0.52 (Hindi, Telugu). So the paper’s numbers are in the **~0.30–0.52** range.

Standard **MS MARCO** passage ranking uses **~1,000 candidate passages per query** (e.g. from BM25). The model must re-rank those 1,000 candidates; the relevant passage is mixed in with many non-relevant ones. That makes the task hard and keeps MRR in the ~0.3–0.5 range for strong models.

---

## How our harness evaluates (current behavior)

1. **Data after loading:** The loader keeps **only rows where the passage is relevant** (`is_selected=True` or `relevance_score > 0`). So we end up with **one (query, passage) pair per query** — 1,000 queries and 1,000 passages (each passage is the gold for one query).
2. **Passage pool:** In `retrieval.py`, the pool for a language is **all those 1,000 gold passages**. So for every query we rank **the same 1,000 passages** (the gold passages for all 1,000 queries).
3. **Gold mapping:** Query *i*’s relevant passage is passage *i* in that pool (`gold_indices = np.arange(n)`). So we ask: “For query *i*, at what rank does passage *i* appear among the 1,000 gold passages?”
4. **Result:** The “negatives” for each query are the **gold passages of other queries** — different questions and different answers. A good encoder easily separates them, so we get **very high MRR** (e.g. **~88%** for BGE-M3 on Hindi).

So:

- **Paper:** For each query, rank **~1,000 candidates (many non-relevant)** → MRR ~0.52.
- **Our script:** For each query, rank **1,000 passages that are each the gold for some other query** → much easier → MRR ~0.88.

The difference is **not** MRR vs MRR@10; it’s **pool definition**: paper = many non-relevant candidates per query (MS MARCO-style); ours = one pool of all gold passages (effectively “match query to its gold among other golds”).

---

## What we’d need for paper-comparable scores

To align with the paper (and MS MARCO-style evaluation):

1. **Per-query candidate set**  
   For each query we need **all candidate passages** (e.g. ~1,000) for that query, with exactly one (or more) marked as relevant — not just the single relevant row.

2. **Evaluation protocol**  
   - For each query: encode the query and **that query’s** candidate passages only.  
   - Rank those candidates by similarity.  
   - Compute 1 / (rank of the relevant passage).  
   - Average over queries → **MRR**.

3. **Data**  
   - Either the IndicMSMARCO dataset provides **multiple rows per query** (many passages per `query_id`, with relevance labels). In that case we must **stop filtering to only relevant rows** and instead group by `query_id`, keep all passages, and run the per-query ranking above.  
   - Or the public dataset only has (query, relevant_passage) pairs. Then we’d need the **official evaluation pool** (e.g. 1,000 candidates per query from the paper’s setup) or we’d need to reproduce it (e.g. build a corpus and use BM25 to get 1,000 candidates per query), then run the same protocol.

---

## Summary

| Aspect            | Paper (Table 2)           | Our current script                    |
|------------------|---------------------------|----------------------------------------|
| Pool per query   | ~1,000 candidates (many non-relevant) | One shared pool of 1,000 gold passages |
| Gold for query *i* | One (or more) relevant in that query’s 1,000 | Passage *i* in the shared pool         |
| MRR (e.g. BGE-M3, Hindi) | **~0.52**                 | **~0.88** (not comparable)            |
| Comparable?      | —                         | **No** — different evaluation setup    |

So the paper evaluates with **MS MARCO-style ranking over many candidates per query**; our script evaluates **matching each query to its gold in a pool of all golds**. To get **paper-comparable** scores we need to switch to per-query candidate sets and compute MRR over those (and ensure the dataset or official pool provides those candidates).

---

## What we added: paper-protocol evaluation with an external pool

When you have a **per-query candidate pool** (e.g. from the paper authors or built with BM25 over a passage corpus), you can run paper-style MRR:

1. **Pool file format:** A directory containing one JSONL file per language: `hi.jsonl`, `bn.jsonl`, etc.  
   Each line is one query:
   ```json
   {"query_id": "...", "query": "query text", "passages": [{"passage": "text", "relevant": true}, {"passage": "text", "relevant": false}, ...]}
   ```
   Each query must have at least one passage with `"relevant": true`.

2. **Run with pool:**
   ```bash
   python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --split test --lang hi \
     --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
     --tasks retrieval \
     --indicmsmarco-pool /path/to/pool_dir \
     -o results_paper_hi.json
   ```
   This uses `run_retrieval_eval_per_query` and reports MRR (and Hit@1, etc.) as the average of 1/rank over queries, matching the paper’s setup.

3. **Getting a pool to try (full 1000 queries):**  
   - **`pool_sample/hi.jsonl`** in this repo has only 2 rows for a minimal sanity check.  
   - To get a **full-sized pool** (1000 queries, 1000 passages per query) from the **same** HuggingFace IndicMSMARCO data, run:
     ```bash
     python scripts/build_indicmsmarco_pool_from_hf.py --output-dir pool_hi --lang hi
     ```
     That script downloads `ai4bharat/IndicMSMARCO` from HF and writes `pool_hi/hi.jsonl` with one line per query; each line’s `passages` = [gold] + [all other queries’ gold passages as negatives]. So you get 1000 queries and 1000 candidates per query with **no other dataset**. You can then run:
     ```bash
     python -m benchmark_indic_rag_suite --dataset ai4bharat/IndicMSMARCO --lang hi --tasks retrieval \
       --retrieval-backend hf --retrieval-model BAAI/bge-m3 \
       --indicmsmarco-pool pool_hi -o results_pool_hi.json
     ```
     **Note:** Because the negatives are still other queries’ gold passages, MRR will remain high (not paper-like). The script is for testing the pipeline and having a real-sized pool.

4. **Building a paper-style pool from MSMARCO-XI:** You can build a pool with real non-relevant candidates using **[MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)** (same [IndicRAGSuite](https://arxiv.org/abs/2506.01615) collection) and the script `scripts/build_indicmsmarco_pool_from_msmarco_xi.py`:

   - **Direct mapping** (no BM25): Uses MSMARCO-XI’s (query, passages, is_selected) as the pool. Run:
     ```bash
     python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method direct --output-dir pool_xi_hi --lang hi --from-hf
     ```
     This loads IndicMSMARCO (1000 queries) and MSMARCO-XI validation from HuggingFace, matches by `query_id`, and writes `pool_xi_hi/hi.jsonl`. Where query_ids align, each query gets multiple passages with real relevance labels (paper-like MRR).

   - **From local MSMARCO-XI files:** If you download the dataset repo (e.g. `hinval.jsonl`, `hintrain.jsonl`), use:
     ```bash
     python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method direct --output-dir pool_xi_hi --lang hi --msmarco-xi-dir /path/to/MSMARCO-XI-data
     ```

   - **BM25 over MSMARCO-XI corpus:** To get ~1000 candidates per query from the full train passage set (pip install rank_bm25):
     ```bash
     python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method bm25 --output-dir pool_bm25_hi --lang hi --from-hf --top-k 1000
     ```

   Then run retrieval with the new pool: `--indicmsmarco-pool pool_xi_hi` (or `pool_bm25_hi`).

5. **Do we need to change dataset loading or any other logic?**  
   **No.** When you pass `--indicmsmarco-pool`, retrieval uses **only** the pool directory (via `load_indicmsmarco_pool`). The normal HuggingFace dataset loading is not used for retrieval in that case. If you also run generation, it still uses the usual `load_benchmark_data` from HF. So existing dataset loading and the rest of the harness stay as-is; the pool is an alternative input only for retrieval.
