"""
Data loading for Indic-Rag-Suite and IndicMSMARCO.
Normalizes rows to {query, passage, answer}. Shardable for distributed runs.
"""

from __future__ import annotations

import logging
from benchmark_indic_rag_suite.config import (
    INDIC_MSMARCO_DATASET,
    INDIC_MSMARCO_LANGUAGES,
    INDIC_RAG_SUITE_LANGUAGES,
)

logger = logging.getLogger(__name__)

INDIC_RAG_SUITE_DATASET = "ai4bharat/Indic-Rag-Suite"


def _get_language_list(languages: list[str], dataset_name: str) -> list[str]:
    if len(languages) == 1 and languages[0].lower() == "all":
        if INDIC_MSMARCO_DATASET in dataset_name:
            return INDIC_MSMARCO_LANGUAGES.copy()
        return INDIC_RAG_SUITE_LANGUAGES.copy()
    return list(languages)


def _normalize_row(row: dict, dataset_name: str) -> dict:
    if INDIC_RAG_SUITE_DATASET in dataset_name:
        return {
            "query": (row.get("question") or "").strip(),
            "passage": (row.get("paragraph") or "").strip(),
            "answer": (row.get("answer") or "").strip(),
            **{k: v for k, v in row.items() if k not in ("question", "paragraph", "answer")},
        }
    return {
        "query": (row.get("query") or "").strip(),
        "passage": (row.get("passage") or "").strip(),
        "answer": (row.get("answer") or "").strip(),
        **{k: v for k, v in row.items() if k not in ("query", "passage", "answer")},
    }


def load_benchmark_data(
    dataset_name: str = INDIC_RAG_SUITE_DATASET,
    languages: list[str] | None = None,
    split: str = "train",
    max_samples_per_lang: int | None = None,
    cache_dir: str | None = None,
    shard_index: int = 0,
    shard_total: int = 1,
) -> dict[str, list[dict]]:
    """
    Load benchmark data per language. Returns dict[lang_code, list[dict]]
    with normalized keys: query, passage, answer.
    """
    from datasets import load_dataset

    languages = languages or ["hi"]
    lang_list = _get_language_list(languages, dataset_name)
    out: dict[str, list[dict]] = {}

    load_split = split
    if dataset_name == INDIC_RAG_SUITE_DATASET and split in ("dev", "test"):
        load_split = "train"
        logger.info("Using 'train' split (Indic-Rag-Suite has no '%s'; --max-samples still applies)", split)

    for lang in lang_list:
        try:
            ds = load_dataset(
                dataset_name,
                lang,
                split=load_split,
                cache_dir=cache_dir,
                trust_remote_code=False,
            )
        except Exception as e:
            if load_split == split and ("Split" in str(e) or "split" in str(e).lower()):
                ds = load_dataset(
                    dataset_name,
                    lang,
                    split="train",
                    cache_dir=cache_dir,
                    trust_remote_code=False,
                )
                logger.info("Using 'train' split for %s (no '%s'); --max-samples still applies", lang, split)
            else:
                raise
        n = len(ds)
        if max_samples_per_lang is not None:
            n = min(n, max_samples_per_lang)
            ds = ds.select(range(n))

        if shard_total > 1:
            shard_size = (n + shard_total - 1) // shard_total
            start = shard_index * shard_size
            end = min(start + shard_size, n)
            if start >= n:
                out[lang] = []
                continue
            ds = ds.select(range(start, end))

        rows = [_normalize_row(dict(row), dataset_name) for row in ds]
        rows = [r for r in rows if (r.get("query") or "").strip() or (r.get("passage") or "").strip()]

        # IndicMSMARCO: may have multiple rows per query (many candidate passages); only some are relevant.
        # Keep only rows where this passage is the relevant one, so (query_i, passage_i) = gold pair.
        if INDIC_MSMARCO_DATASET in dataset_name:
            def _is_relevant(r: dict) -> bool:
                # Support multiple column names and types (parquet may use different names)
                sel = r.get("is_selected") or r.get("selected")
                if sel is not None and (sel is True or sel == 1 or (isinstance(sel, str) and sel.lower() in ("true", "1"))):
                    return True
                rel = r.get("relevance_score") or r.get("relevance")
                if rel is not None:
                    try:
                        return float(rel) > 0
                    except (TypeError, ValueError):
                        pass
                return False

            relevant_rows = [r for r in rows if _is_relevant(r)]
            if relevant_rows:
                before = len(rows)
                rows = relevant_rows
                logger.info(
                    "IndicMSMARCO %s: kept only relevant (query, passage) rows: %d -> %d",
                    lang, before, len(rows),
                )
            elif rows and rows[0].get("query_id") is not None:
                # One row per query_id when no relevance flag: keep first row per query_id (assume 1:1)
                from collections import OrderedDict
                by_qid = OrderedDict()
                for r in rows:
                    qid = r.get("query_id")
                    if qid not in by_qid:
                        by_qid[qid] = r
                rows = list(by_qid.values())
                logger.info("IndicMSMARCO %s: one row per query_id (no relevance column): %d rows", lang, len(rows))
            else:
                logger.info(
                    "IndicMSMARCO %s: no relevance/is_selected column found; using all %d rows. "
                    "If MRR is very low, dataset may have multiple passages per query.",
                    lang, len(rows),
                )
                if rows:
                    logger.debug("IndicMSMARCO %s first row keys: %s", lang, list(rows[0].keys()))
        out[lang] = rows
    return out
