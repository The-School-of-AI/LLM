#!/usr/bin/env python3
"""
Build per-query pool JSONL files from HuggingFace IndicMSMARCO for paper-protocol evaluation.

The HF dataset has only one row per query (the relevant passage). This script builds a
synthetic pool per query: for each query, pool = [gold passage] + [all other queries' gold
passages as negatives]. So each query gets 1000 passages (1 relevant, 999 negatives).
This lets you run the full pipeline with --indicmsmarco-pool and 1000 queries.

Note: Because negatives are other queries' gold passages, MRR will still be high (not
paper-like). For paper-comparable MRR you need a real candidate set (e.g. from the
paper authors or BM25 over a large corpus). This script is for testing the pipeline
and having a full-sized pool to try.

Usage:
  python scripts/build_indicmsmarco_pool_from_hf.py --output-dir pool_hi --lang hi
  python scripts/build_indicmsmarco_pool_from_hf.py --output-dir pool_all --lang all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# IndicMSMARCO language codes (match benchmark_indic_rag_suite.config)
INDIC_MSMARCO_LANGUAGES = [
    "as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IndicMSMARCO pool JSONL from HF dataset")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory (e.g. pool_hi)")
    parser.add_argument("--lang", default="hi", help="Language code or 'all'")
    parser.add_argument("--max-queries", type=int, default=None, help="Cap queries per language (for testing)")
    parser.add_argument("--dataset", default="ai4bharat/IndicMSMARCO", help="HuggingFace dataset name")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        return 1

    languages = INDIC_MSMARCO_LANGUAGES if (args.lang or "").lower() == "all" else [args.lang]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang in languages:
        try:
            ds = load_dataset(args.dataset, lang, split="train", trust_remote_code=False)
        except Exception as e:
            print(f"Warning: could not load {args.dataset} {lang}: {e}", file=sys.stderr)
            continue

        rows = []
        for i, row in enumerate(ds):
            if args.max_queries and i >= args.max_queries:
                break
            qid = row.get("query_id", str(i))
            query = (row.get("query") or "").strip()
            passage = (row.get("passage") or "").strip()
            if not query or not passage:
                continue
            rows.append({"query_id": qid, "query": query, "passage": passage})

        if not rows:
            print(f"No rows for {lang}", file=sys.stderr)
            continue

        # Build pool per query: [gold] + [all other passages as negatives]
        passages_list = [r["passage"] for r in rows]
        out_file = out_dir / f"{lang}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                # Relevant passage at index i; others as negatives
                pool_entries = []
                for j, p in enumerate(passages_list):
                    pool_entries.append({"passage": p, "relevant": j == i})
                line = json.dumps(
                    {"query_id": r["query_id"], "query": r["query"], "passages": pool_entries},
                    ensure_ascii=False,
                )
                f.write(line + "\n")
        print(f"Wrote {len(rows)} queries to {out_file} (pool size {len(passages_list)} per query)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
