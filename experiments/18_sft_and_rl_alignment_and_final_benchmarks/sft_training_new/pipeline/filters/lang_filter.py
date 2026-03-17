"""
Language detection filter.
Detects the language of user turns (or all turns, depending on config).
Drops records whose detected language is not in the allowed list.
Also attaches ``_lang`` metadata to passing records for Stage 6 reporting.
"""
from __future__ import annotations

import logging

from pipeline.filters.base import BaseFilter
from pipeline.config import LangFilterConfig

logger = logging.getLogger(__name__)


def _sample_text(record: dict, sample_from: str) -> str:
    turns = record.get("conversations", [])
    if sample_from == "user":
        parts = [t.get("content") or "" for t in turns if t.get("role") == "user"]
    elif sample_from == "assistant":
        parts = [t.get("content") or "" for t in turns if t.get("role") == "assistant"]
    else:  # all
        parts = [t.get("content") or "" for t in turns]
    return " ".join(parts)[:2000]  # cap for speed


class LangFilter(BaseFilter):
    name = "lang_filter"

    def __init__(self, cfg: LangFilterConfig) -> None:
        self._cfg = cfg
        self._backend = cfg.backend
        self._ft_model = None

        if self._backend == "fasttext":
            if not cfg.fasttext_model_path:
                raise ValueError("lang_filter.fasttext_model_path must be set when backend=fasttext")
            try:
                import fasttext
                self._ft_model = fasttext.load_model(cfg.fasttext_model_path)
            except ImportError:
                raise ImportError("fasttext is required: pip install fasttext-wheel")

    def filter(self, record: dict) -> tuple[bool, str]:
        text = _sample_text(record, self._cfg.sample_from)
        if not text.strip():
            if self._cfg.on_error == "drop":
                return False, "lang_detection:empty_text"
            return True, ""

        try:
            lang, confidence = self._detect(text)
        except Exception as exc:
            logger.debug("Language detection failed: %s", exc)
            if self._cfg.on_error == "drop":
                return False, f"lang_detection:error:{exc}"
            # Attach unknown lang and keep
            record["_lang"] = "unknown"
            return True, ""

        if confidence < self._cfg.min_confidence:
            # Low-confidence detection — treat as unknown, keep
            record["_lang"] = f"{lang}(low_conf)"
            return True, ""

        record["_lang"] = lang

        if lang not in self._cfg.allowed_languages:
            return False, f"lang_not_allowed:{lang}"
        return True, ""

    def _detect(self, text: str) -> tuple[str, float]:
        if self._backend == "langdetect":
            from langdetect import detect_langs
            results = detect_langs(text)
            if not results:
                return "unknown", 0.0
            top = results[0]
            return top.lang, top.prob
        elif self._backend == "fasttext":
            labels, probs = self._ft_model.predict(text.replace("\n", " "), k=1)
            lang = labels[0].replace("__label__", "")
            return lang, float(probs[0])
        else:
            raise ValueError(f"Unknown lang_filter backend: {self._backend}")
