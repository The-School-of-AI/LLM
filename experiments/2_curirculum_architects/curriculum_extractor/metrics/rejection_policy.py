"""Rejection policy metric that enforces canonical curriculum rejection reasons.

This metric enforces a small, safe subset of curriculum-level rejections at
extraction time:

- language_not_en_or_indic: rejects samples whose declared language is not
  in the approved primary/secondary language lists
- below_minimum_token_threshold: rejects samples with token counts below the
  curriculum-configured minimum (approximate token count via whitespace)

NOTE: Stage- and band-aware rules (e.g. indic_not_allowed_at_stage,
agentic_not_allowed_in_band_at_stage) are intentionally NOT enforced here
because band/stage assignment happens downstream in post-processing.
"""

from typing import Any, Dict, List

from ..core.plugin import ExtractionResult, MetricPlugin, ReadOnlyRecord


class RejectionPolicyMetric(MetricPlugin):
    """Metric implementing lightweight rejection policy.

    Runs at level 0 to fail fast on obvious policy violations.
    """

    name = "rejection_policy"
    level = 0

    def compute(self, record: ReadOnlyRecord) -> Dict[str, Any]:
        # Provide a simple marker that the check ran
        return {"policy_checked": True}

    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        # Access curriculum config via self.config
        cfg = self.config

        text = record.get("text", "") or ""

        # Language enforcement
        # Build allowed language set from curriculum config
        allowed: List[str] = []
        primary = cfg.get("language_and_context.language_policy.primary_languages", [])
        for p in primary:
            if isinstance(p, dict) and p.get("lang"):
                allowed.append(p.get("lang"))
            elif isinstance(p, str):
                allowed.append(p)

        secondary = cfg.get("language_and_context.language_policy.secondary_languages", [])
        for s in secondary:
            if isinstance(s, dict) and s.get("lang"):
                allowed.append(s.get("lang"))
            elif isinstance(s, str):
                allowed.append(s)

        # Normalize language value from record (dataset_interface expects 'language')
        lang = record.get("language")
        if lang is None:
            # Missing language is treated as a policy violation
            reason = "language_not_en_or_indic"
            metrics = self.compute(record)
            # Include rejection reason in metrics so metadata contains it
            metrics["rejection_reason"] = reason
            return ExtractionResult(metrics=metrics, rejected=True, rejection_reason=reason)

        if isinstance(lang, str):
            lang_norm = lang.strip().lower()
        else:
            lang_norm = str(lang).strip().lower()

        allowed_norm = [a.strip().lower() for a in allowed]

        if allowed_norm and lang_norm not in allowed_norm:
            reason = "language_not_en_or_indic"
            metrics = self.compute(record)
            metrics["rejection_reason"] = reason
            return ExtractionResult(metrics=metrics, rejected=True, rejection_reason=reason)

        # Minimum token threshold (approximate via whitespace split)
        min_tokens = cfg.get("language_and_context.context_policy.min_context_tokens", 0)

        # Use a cheap token approximation
        token_count = len(text.split())

        try:
            min_tokens_int = int(min_tokens)
        except Exception:
            min_tokens_int = 0

        if min_tokens_int > 0 and token_count < min_tokens_int:
            reason = "below_minimum_token_threshold"
            metrics = self.compute(record)
            metrics["rejection_reason"] = reason
            return ExtractionResult(metrics=metrics, rejected=True, rejection_reason=reason)

        # Passed quick checks
        return ExtractionResult(metrics=self.compute(record), rejected=False)
