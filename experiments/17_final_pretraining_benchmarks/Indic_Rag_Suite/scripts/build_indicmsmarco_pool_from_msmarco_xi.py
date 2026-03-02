#!/usr/bin/env python3
"""
Build paper-style per-query pool JSONL from MSMARCO-XI for IndicMSMARCO evaluation.

MSMARCO-XI (ai4bharat/MSMARCO-XI) has multiple passages per query with is_selected
relevance. This script produces pool files you can use with --indicmsmarco-pool.

Two modes:
  1. direct  – Map MSMARCO-XI (query, passages, is_selected) to our pool JSONL.
               Use IndicMSMARCO query set and fill pools from MSMARCO-XI when
               query_id matches. Real non-relevant passages → paper-like MRR.
  2. bm25    – Build a passage corpus from MSMARCO-XI, run BM25 per IndicMSMARCO
               query, take top-K (e.g. 1000), add gold if missing. Needs rank_bm25.

Data source:
  - Local: --msmarco-xi-dir /path/to/dir containing hintrain.jsonl, hinval.jsonl, etc.
  - HuggingFace: --from-hf loads ai4bharat/MSMARCO-XI (validation) and filters by lang.

Usage:
  # Direct mapping (MSMARCO-XI validation rows → pool; align to IndicMSMARCO query_ids)
  python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method direct --output-dir pool_xi_hi --lang hi --from-hf

  # Or from local JSONL (download from https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
  python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method direct --output-dir pool_xi_hi --lang hi --msmarco-xi-dir ./MSMARCO-XI-data

  # BM25 over MSMARCO-XI train corpus (pip install rank_bm25)
  python scripts/build_indicmsmarco_pool_from_msmarco_xi.py --method bm25 --output-dir pool_bm25_hi --lang hi --from-hf --top-k 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Our lang code -> MSMARCO-XI file prefix (from dataset card: hintrain, hinval, etc.)
LANG_TO_PREFIX = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "te": "tel",
    "ur": "urd",
}

INDIC_MSMARCO_LANGUAGES = list(LANG_TO_PREFIX.keys())


def _load_msmarco_xi_from_dir(data_dir: Path, lang: str, split: str) -> list[dict]:
    """Load MSMARCO-XI from local JSONL or Parquet (e.g. hinval.jsonl or hinval.parquet)."""
    prefix = LANG_TO_PREFIX.get(lang)
    if not prefix:
        return []
    suffix = "val" if split == "validation" else "train"
    path = None
    for ext in (".jsonl", ".parquet"):
        p = data_dir / f"{prefix}{suffix}{ext}"
        if p.exists():
            path = p
            break
    if path is None:
        # Try validation/ or train/ subdir (repo layout)
        sub = "validation" if split == "validation" else "train"
        for ext in (".jsonl", ".parquet"):
            p = data_dir / sub / f"{prefix}{suffix}{ext}"
            if p.exists():
                path = p
                break
    if path is None:
        return []
    rows = []
    if path.suffix == ".parquet":
        try:
            from datasets import load_dataset
            ds = load_dataset("parquet", data_files=str(path), split="train", trust_remote_code=False)
            rows = [ds[i] for i in range(len(ds))]
        except Exception as e:
            print(f"Failed to read {path}: {e}", file=sys.stderr)
            return []
    else:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# target_lang in MSMARCO-XI examples: asm_Beng, hin_Deva, etc.
LANG_TO_TARGET_PREFIX = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san",
    "ta": "tam", "te": "tel", "ur": "urd",
}


# HuggingFace no longer supports trust_remote_code for dataset loading scripts.
# MSMARCO-XI has Parquet files per language (e.g. validation/hinval.parquet). Load those directly.
HF_MSMARCO_XI_PARQUET_URL = (
    "https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{split}/{prefix}{suffix}.parquet"
)


def _load_msmarco_xi_from_hf(lang: str, split: str) -> list[dict]:
    """Load MSMARCO-XI from HuggingFace by loading the language's Parquet file directly (no custom script)."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []
    prefix = LANG_TO_PREFIX.get(lang)
    if not prefix:
        print(f"Unknown language: {lang}", file=sys.stderr)
        return []
    suffix = "val" if split == "validation" else "train"
    url = HF_MSMARCO_XI_PARQUET_URL.format(split=split, prefix=prefix, suffix=suffix)
    print(f"Loading MSMARCO-XI for '{lang}' ({split}) from Parquet...", file=sys.stderr)
    print("First-time download can take 5–15 min (e.g. ~460 MB for Hindi validation).", file=sys.stderr)
    sys.stderr.flush()
    try:
        ds = load_dataset("parquet", data_files=url, split="train")
    except Exception as e:
        print(f"HF load failed: {e}", file=sys.stderr)
        return []
    print(f"  Loaded {len(ds)} rows.", file=sys.stderr)
    sys.stderr.flush()
    return [ds[i] for i in range(len(ds))]


def _normalize_msmarco_xi_row(row: dict) -> tuple[str, str, list[tuple[str, bool]]] | None:
    """Return (query_id, query, [(passage, relevant), ...]) or None."""
    qid = row.get("query_id")
    query = (row.get("query") or "").strip()
    passages = row.get("passages") or {}
    if isinstance(passages, list):
        # Some formats have list of {passage, is_selected}
        out = []
        for p in passages:
            text = (p.get("passage") or p.get("text") or p.get("Translated_passage") or "").strip()
            sel = p.get("is_selected") or p.get("relevant")
            if isinstance(sel, (list, tuple)):
                sel = (sel[0] if sel else False)
            out.append((text, bool(sel and (sel == 1 or str(sel).lower() in ("true", "1")))))
        if not query or not out:
            return None
        return (str(qid), query, out)
    # Dict with Translated_passages and is_selected lists
    trans = passages.get("Translated_passages") or passages.get("passages") or []
    sel_list = passages.get("is_selected") or []
    if not trans:
        return None
    out = []
    for j, p in enumerate(trans):
        text = (p if isinstance(p, str) else (p.get("passage") or p.get("text") or "")).strip()
        sel = sel_list[j] if j < len(sel_list) else 0
        out.append((text, bool(sel and (sel == 1 or str(sel).lower() in ("true", "1")))))
    if not query:
        return None
    return (str(qid), query, out)


def run_direct(
    indicmsmarco_queries: list[dict],
    msmarco_xi_rows: list[dict],
    out_path: Path,
    max_queries: int | None,
) -> None:
    """Align IndicMSMARCO query_ids to MSMARCO-XI and write pool JSONL (direct mapping)."""
    # Build qid -> (query, passages with relevant) from MSMARCO-XI
    xi_by_qid: dict[str, tuple[str, list[tuple[str, bool]]]] = {}
    for row in msmarco_xi_rows:
        norm = _normalize_msmarco_xi_row(row)
        if norm:
            qid, query, passages = norm
            if passages and any(r for _, r in passages):
                xi_by_qid[str(qid)] = (query, passages)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in indicmsmarco_queries:
            if max_queries and written >= max_queries:
                break
            qid = str(r.get("query_id", ""))
            query = (r.get("query") or "").strip()
            gold = (r.get("passage") or "").strip()
            if not query or not gold:
                continue
            if qid in xi_by_qid:
                _, passages = xi_by_qid[qid]
                pool_entries = [{"passage": p, "relevant": rel} for p, rel in passages]
            else:
                # Fallback: single gold (pool size 1)
                pool_entries = [{"passage": gold, "relevant": True}]
            line = json.dumps({"query_id": qid, "query": query, "passages": pool_entries}, ensure_ascii=False)
            f.write(line + "\n")
            written += 1
    print(f"Wrote {written} queries to {out_path} (direct from MSMARCO-XI)")


def run_bm25(
    indicmsmarco_queries: list[dict],
    corpus_passages: list[str],
    out_path: Path,
    top_k: int,
    max_queries: int | None,
) -> None:
    """Build pool by BM25 retrieval over corpus; add gold if not in top_k."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("BM25 mode needs: pip install rank_bm25", file=sys.stderr)
        sys.exit(1)

    n_corpus = len(corpus_passages)
    print(f"Tokenizing {n_corpus} passages (this may take 1–2 min)...", file=sys.stderr)
    sys.stderr.flush()
    tokenize = lambda t: t.lower().split()
    tokenized_corpus = [tokenize(p) for p in corpus_passages]
    print("Building BM25 index...", file=sys.stderr)
    sys.stderr.flush()
    bm25 = BM25Okapi(tokenized_corpus)
    n_queries = len(indicmsmarco_queries)
    print(f"Running BM25 for {n_queries} queries (may take several minutes)...", file=sys.stderr)
    sys.stderr.flush()

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for q_idx, r in enumerate(indicmsmarco_queries):
            if max_queries and written >= max_queries:
                break
            if q_idx > 0 and q_idx % 100 == 0:
                print(f"  BM25: {q_idx}/{n_queries} queries...", file=sys.stderr)
                sys.stderr.flush()
            query = (r.get("query") or "").strip()
            gold = (r.get("passage") or "").strip()
            if not query or not gold:
                continue
            scores = bm25.get_scores(tokenize(query))
            top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[: top_k]
            seen = set()
            pool_entries = []
            for i in top_indices:
                p = corpus_passages[i]
                if p in seen:
                    continue
                seen.add(p)
                pool_entries.append({"passage": p, "relevant": p.strip() == gold.strip()})
            if gold.strip() not in seen:
                pool_entries.append({"passage": gold, "relevant": True})
            line = json.dumps(
                {"query_id": r.get("query_id", ""), "query": query, "passages": pool_entries},
                ensure_ascii=False,
            )
            f.write(line + "\n")
            written += 1
    print(f"Wrote {written} queries to {out_path} (BM25 top-{top_k} over corpus)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build IndicMSMARCO pool from MSMARCO-XI")
    parser.add_argument("--method", choices=["direct", "bm25"], default="direct")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--lang", default="hi")
    parser.add_argument("--msmarco-xi-dir", default=None, help="Local dir with hintrain.jsonl, hinval.jsonl, ...")
    parser.add_argument("--from-hf", action="store_true", help="Load MSMARCO-XI from HuggingFace")
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--top-k", type=int, default=1000, help="For BM25: number of candidates per query")
    parser.add_argument("--max-corpus-passages", type=int, default=None, help="Cap corpus size for BM25 (faster run; default: use all)")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load IndicMSMARCO (our 1000 queries)
    try:
        ds_indic = load_dataset("ai4bharat/IndicMSMARCO", args.lang, split="train", trust_remote_code=False)
    except Exception as e:
        print(f"Failed to load IndicMSMARCO {args.lang}: {e}", file=sys.stderr)
        return 1
    indic_queries = []
    for i in range(len(ds_indic)):
        row = ds_indic[i]
        indic_queries.append({
            "query_id": row.get("query_id"),
            "query": (row.get("query") or "").strip(),
            "passage": (row.get("passage") or "").strip(),
        })

    # Load MSMARCO-XI
    if args.msmarco_xi_dir:
        xi_rows = _load_msmarco_xi_from_dir(Path(args.msmarco_xi_dir), args.lang, args.split)
    elif args.from_hf:
        xi_rows = _load_msmarco_xi_from_hf(args.lang, args.split)
    else:
        print("Provide --msmarco-xi-dir or --from-hf", file=sys.stderr)
        return 1

    if not xi_rows and args.method == "direct":
        print("No MSMARCO-XI rows loaded; check --msmarco-xi-dir or --from-hf and lang.", file=sys.stderr)
        return 1

    out_file = out_dir / f"{args.lang}.jsonl"

    if args.method == "direct":
        run_direct(indic_queries, xi_rows, out_file, args.max_queries)
    else:
        # BM25: build corpus from MSMARCO-XI passages (all unique)
        print("Building passage corpus from MSMARCO-XI rows...", file=sys.stderr)
        sys.stderr.flush()
        corpus = []
        for row in xi_rows:
            norm = _normalize_msmarco_xi_row(row)
            if norm:
                _, _, passages = norm
                for p, _ in passages:
                    if p and p.strip():
                        corpus.append(p.strip())
        corpus = list(dict.fromkeys(corpus))
        if args.max_corpus_passages and len(corpus) > args.max_corpus_passages:
            print(f"Capping corpus at {args.max_corpus_passages} (was {len(corpus)}) for faster run.", file=sys.stderr)
            corpus = corpus[: args.max_corpus_passages]
        print(f"Corpus: {len(corpus)} unique passages.", file=sys.stderr)
        sys.stderr.flush()
        if not corpus:
            print("No passages in MSMARCO-XI for BM25 corpus.", file=sys.stderr)
            return 1
        run_bm25(indic_queries, corpus, out_file, args.top_k, args.max_queries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
