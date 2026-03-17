"""
Near-deduplication via MinHash LSH (datasketch).
Identifies records with Jaccard similarity above the configured threshold
and keeps only one per near-duplicate cluster.

Memory estimate: 200K records × 128 perms × 8 bytes ≈ 200 MB (in-process).
For >5M records, use datasketch's Redis backend (storage_config parameter).
"""
from __future__ import annotations

import logging

from pipeline.filters.base import BaseFilter
from pipeline.config import NearDedupConfig

logger = logging.getLogger(__name__)


def _extract_text(record: dict, field: str) -> str:
    """Extract the text to hash based on the configured field."""
    turns = record.get("conversations", [])
    if field == "user_content":
        parts = [t.get("content") or "" for t in turns if t.get("role") == "user"]
    elif field == "assistant_content":
        parts = [t.get("content") or "" for t in turns if t.get("role") == "assistant"]
    else:  # full_text
        parts = [t.get("content") or "" for t in turns]
    return " ".join(parts).lower()


def _build_ngrams(text: str, n: int) -> list[bytes]:
    tokens = text.split()
    if len(tokens) < n:
        return [" ".join(tokens).encode("utf-8")] if tokens else []
    return [" ".join(tokens[i : i + n]).encode("utf-8") for i in range(len(tokens) - n + 1)]


class NearDedup(BaseFilter):
    name = "near_dedup"

    def __init__(self, cfg: NearDedupConfig) -> None:
        self._cfg = cfg
        try:
            from datasketch import MinHashLSH, MinHash as DMinHash
            self._MinHash = DMinHash
            self._lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
        except ImportError:
            raise ImportError("datasketch is required: pip install datasketch")
        self._counter = 0

    def filter(self, record: dict) -> tuple[bool, str]:
        text = _extract_text(record, self._cfg.hash_field)
        ngrams = _build_ngrams(text, self._cfg.ngram_size)

        m = self._MinHash(num_perm=self._cfg.num_perm)
        for ng in ngrams:
            m.update(ng)

        key = f"r{self._counter}"
        self._counter += 1

        try:
            results = self._lsh.query(m)
        except Exception:
            results = []

        if results:
            return False, f"near_duplicate:similar_to={results[0]}"

        try:
            self._lsh.insert(key, m)
        except Exception as exc:
            logger.debug("LSH insert error (likely duplicate key): %s", exc)

        return True, ""
