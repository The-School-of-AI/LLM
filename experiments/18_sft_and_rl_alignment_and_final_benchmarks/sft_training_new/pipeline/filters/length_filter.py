"""
Length filter — drops conversations that are too short or too long.
Checks character counts by default; uses the tokenizer if configured.
"""
from __future__ import annotations

from pipeline.filters.base import BaseFilter
from pipeline.config import LengthFilterConfig


def _all_content(record: dict) -> str:
    """Concatenate all turn content strings."""
    turns = record.get("conversations", [])
    return " ".join(t.get("content") or "" for t in turns)


class LengthFilter(BaseFilter):
    name = "length_filter"

    def __init__(self, cfg: LengthFilterConfig) -> None:
        self._cfg = cfg
        self._tokenizer = None
        if cfg.tokenizer_name_or_path:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    cfg.tokenizer_name_or_path, trust_remote_code=True
                )
            except Exception:
                pass  # Silently fall back to char counts

    def filter(self, record: dict) -> tuple[bool, str]:
        text = _all_content(record)
        n_chars = len(text)

        if n_chars < self._cfg.min_chars:
            return False, f"too_short_chars:{n_chars}<{self._cfg.min_chars}"
        if n_chars > self._cfg.max_chars:
            return False, f"too_long_chars:{n_chars}>{self._cfg.max_chars}"

        if self._tokenizer is not None:
            n_tokens = len(self._tokenizer.encode(text, add_special_tokens=False))
            if n_tokens < self._cfg.min_tokens:
                return False, f"too_short_tokens:{n_tokens}<{self._cfg.min_tokens}"
            if n_tokens > self._cfg.max_tokens:
                return False, f"too_long_tokens:{n_tokens}>{self._cfg.max_tokens}"

        return True, ""
