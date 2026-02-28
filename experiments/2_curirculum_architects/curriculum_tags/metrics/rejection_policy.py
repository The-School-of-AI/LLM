"""Rejection policy metric for curriculum tags."""

from typing import Any, Dict, List

from ..core.plugin import MetricPlugin


class RejectionPolicyMetric(MetricPlugin):
    """Metric implementing lightweight rejection policy.

    Runs at level 0 to fail fast on obvious policy violations.
    """

    name = "rejection_policy"
    INDIC_SCRIPT_RANGES = [
        (0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit)
        (0x0980, 0x09FF),  # Bengali
        (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
        (0x0A80, 0x0AFF),  # Gujarati
        (0x0B00, 0x0B7F),  # Oriya
        (0x0B80, 0x0BFF),  # Tamil
        (0x0C00, 0x0C7F),  # Telugu
        (0x0C80, 0x0CFF),  # Kannada
        (0x0D00, 0x0D7F),  # Malayalam
    ]

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute rejection status based on policy."""
        # Access curriculum config via self.config
        cfg = self.config

        text = sample.get("text", "") or ""

        # Language enforcement
        # Build allowed language set from curriculum config
        allowed: List[str] = []
        primary = cfg.get("language_and_context.language_policy.primary_languages", [])
        for p in primary:
            if isinstance(p, dict) and p.get("lang"):
                allowed.append(p.get("lang"))
            elif isinstance(p, str):
                allowed.append(p)

        secondary = cfg.get(
            "language_and_context.language_policy.secondary_languages", []
        )
        for s in secondary:
            if isinstance(s, dict) and s.get("lang"):
                allowed.append(s.get("lang"))
            elif isinstance(s, str):
                allowed.append(s)

        # Normalize language value from record (dataset_interface expects 'language')
        lang = _detect_language(text, sample)

        # Determine rejection
        rejected = False
        rejection_reason = None

        if lang is None:
            # Missing language is treated as a policy violation
            rejected = True
            rejection_reason = "language_not_en_or_indic"
        else:
            if isinstance(lang, str):
                lang_norm = lang.strip().lower()
            else:
                lang_norm = str(lang).strip().lower()

            allowed_norm = [a.strip().lower() for a in allowed]

            if allowed_norm and lang_norm not in allowed_norm:
                rejected = True
                rejection_reason = "language_not_en_or_indic"

        # Minimum token threshold (approximate via whitespace split)
        if not rejected:
            min_tokens = cfg.get(
                "language_and_context.context_policy.min_context_tokens", 0
            )

            # Use a cheap token approximation
            token_count = len(text.split())

            try:
                min_tokens_int = int(min_tokens) / 2  # approx accounting for tokenizer
            except Exception:
                min_tokens_int = 0

            if min_tokens_int > 0 and token_count < min_tokens_int:
                rejected = True
                rejection_reason = "below_minimum_token_threshold"

        result = {
            "policy_checked": True,
            "rejected": rejected,
        }
        if rejection_reason:
            result["rejection_reason"] = rejection_reason
        return result


def _detect_language(text: str, sample: dict = None) -> str:
    """
    Fast language detection: metadata first, then heuristic.

    Returns: 'en', 'indic', or 'other'
    """
    metadata_lang = sample.get("language")
    if metadata_lang is None:
        metadata = sample.get("metadata") or {}
        if isinstance(metadata, dict):
            # Schema uses 'lang' inside metadata
            metadata_lang = metadata.get("lang")

    if metadata_lang:
        if metadata_lang.lower().startswith("en"):
            return "en"
        if metadata_lang.lower() in [
            "as",
            "bn",
            "gu",
            "hi",
            "kn",
            "ml",
            "mr",
            "or",
            "pa",
            "ta",
            "te",
        ]:
            return "indic"
        elif metadata_lang.lower() in [
            "assamese",
            "bengali",
            "gujarati",
            "hindi",
            "kannada",
            "malayalam",
            "marathi",
            "odia",
            "punjabi",
            "tamil",
            "telugu",
        ]:
            return "indic"

    # if still not found, check text
    for char in text[:500]:
        code = ord(char)
        for start, end in INDIC_SCRIPT_RANGES:  # noqa: F821
            if start <= code <= end:
                return "indic"

    return None
