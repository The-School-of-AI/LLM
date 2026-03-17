"""
Repetition filter — drops responses with excessive n-gram repetition.
Computes the ratio of repeated n-grams to total n-grams in the text.
"""
from __future__ import annotations

from pipeline.filters.base import BaseFilter
from pipeline.config import RepetitionFilterConfig


def _repetition_ratio(text: str, n: int) -> float:
    """Fraction of n-grams that are duplicates of an earlier n-gram."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    seen: set[tuple] = set()
    repeated = 0
    for ng in ngrams:
        if ng in seen:
            repeated += 1
        seen.add(ng)
    return repeated / len(ngrams)


class RepetitionFilter(BaseFilter):
    name = "repetition_filter"

    def __init__(self, cfg: RepetitionFilterConfig) -> None:
        self._cfg = cfg

    def filter(self, record: dict) -> tuple[bool, str]:
        turns = record.get("conversations", [])
        for turn in turns:
            if turn.get("role") not in self._cfg.check_roles:
                continue
            text = turn.get("content") or ""
            if not text.strip():
                continue
            ratio = _repetition_ratio(text, self._cfg.ngram_size)
            if ratio > self._cfg.max_repetition_ratio:
                return False, f"repetition:ratio={ratio:.3f}>{self._cfg.max_repetition_ratio}"
        return True, ""
