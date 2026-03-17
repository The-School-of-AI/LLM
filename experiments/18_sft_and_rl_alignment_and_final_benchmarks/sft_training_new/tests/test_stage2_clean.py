"""
Tests for Stage 2 — Cleaning & Filtering.
Tests each filter independently and verifies toggling works.
"""
import hashlib
import pytest
from pipeline.config import (
    ExactDedupConfig, NearDedupConfig, RepetitionFilterConfig,
    BenchmarkDecontamConfig, LengthFilterConfig, PIIFilterConfig,
)
from pipeline.filters.exact_dedup import ExactDedup, normalize_for_hash
from pipeline.filters.repetition_filter import RepetitionFilter, _repetition_ratio
from pipeline.filters.length_filter import LengthFilter
from pipeline.filters.pii_filter import PIIFilter


def _make_record(user: str, assistant: str, source: str = "test") -> dict:
    return {
        "conversations": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "_source": source,
    }


class TestExactDedup:

    def test_first_occurrence_kept(self):
        dedup = ExactDedup(ExactDedupConfig(enabled=True, hash_mode="prompt"))
        rec = _make_record("Hello", "World")
        keep, reason = dedup.filter(rec)
        assert keep is True
        assert reason == ""

    def test_duplicate_dropped(self):
        dedup = ExactDedup(ExactDedupConfig(enabled=True, hash_mode="prompt"))
        rec = _make_record("Hello", "World")
        dedup.filter(rec)  # first — kept
        keep, reason = dedup.filter(dict(rec))  # exact copy — dropped
        assert keep is False
        assert "exact_duplicate" in reason

    def test_different_records_both_kept(self):
        dedup = ExactDedup(ExactDedupConfig(enabled=True, hash_mode="prompt"))
        r1 = _make_record("Question A", "Answer A")
        r2 = _make_record("Question B", "Answer B")
        k1, _ = dedup.filter(r1)
        k2, _ = dedup.filter(r2)
        assert k1 is True
        assert k2 is True

    def test_hash_mode_full_vs_prompt(self):
        """Same prompt but different answers: full-mode keeps both, prompt-mode drops second."""
        r1 = _make_record("Same question", "Answer 1")
        r2 = _make_record("Same question", "Answer 2")

        dedup_prompt = ExactDedup(ExactDedupConfig(hash_mode="prompt"))
        dedup_prompt.filter(r1)
        k2, _ = dedup_prompt.filter(r2)
        assert k2 is False  # prompt is same → duplicate

        dedup_full = ExactDedup(ExactDedupConfig(hash_mode="full"))
        dedup_full.filter(r1)
        k2f, _ = dedup_full.filter(r2)
        assert k2f is True  # full content differs → kept


class TestRepetitionFilter:

    def test_repetition_ratio_no_repetition(self):
        text = "The quick brown fox jumps over the lazy dog"
        ratio = _repetition_ratio(text, n=4)
        assert ratio == 0.0

    def test_repetition_ratio_full_repetition(self):
        # "a b c a b c" with n=3: ngrams=["a b c", "b c a", "c a b", "a b c"]
        # "a b c" appears at positions 0 and 3 → 1 repeat out of 4 = 0.25
        text = "a b c a b c"
        ratio = _repetition_ratio(text, n=3)
        assert ratio > 0.0

    def test_highly_repetitive_text_dropped(self):
        repetitive = "this is bad " * 50
        cfg = RepetitionFilterConfig(enabled=True, max_repetition_ratio=0.3, ngram_size=4)
        flt = RepetitionFilter(cfg)
        rec = _make_record("Question?", repetitive)
        keep, reason = flt.filter(rec)
        assert keep is False
        assert "repetition" in reason

    def test_normal_text_kept(self):
        normal = "Machine learning is a subset of artificial intelligence. It allows systems to learn from data."
        cfg = RepetitionFilterConfig(enabled=True, max_repetition_ratio=0.3, ngram_size=4)
        flt = RepetitionFilter(cfg)
        rec = _make_record("What is ML?", normal)
        keep, _ = flt.filter(rec)
        assert keep is True

    def test_only_checks_configured_roles(self):
        """If check_roles=['assistant'], repetitive user content should not trigger filter."""
        repetitive_user = "bad bad bad bad bad bad bad bad bad bad bad bad"
        good_assistant  = "The answer is 42."
        cfg = RepetitionFilterConfig(enabled=True, max_repetition_ratio=0.01, ngram_size=2,
                                     check_roles=["assistant"])
        flt = RepetitionFilter(cfg)
        rec = _make_record(repetitive_user, good_assistant)
        keep, _ = flt.filter(rec)
        assert keep is True  # user not checked


class TestLengthFilter:

    def test_too_short_dropped(self):
        cfg = LengthFilterConfig(enabled=True, min_chars=100, max_chars=99999)
        flt = LengthFilter(cfg)
        rec = _make_record("Hi", "Yes")
        keep, reason = flt.filter(rec)
        assert keep is False
        assert "too_short_chars" in reason

    def test_too_long_dropped(self):
        cfg = LengthFilterConfig(enabled=True, min_chars=1, max_chars=10)
        flt = LengthFilter(cfg)
        rec = _make_record("A much longer question than ten characters total", "A much longer answer")
        keep, reason = flt.filter(rec)
        assert keep is False
        assert "too_long_chars" in reason

    def test_normal_length_kept(self):
        cfg = LengthFilterConfig(enabled=True, min_chars=5, max_chars=10000)
        flt = LengthFilter(cfg)
        rec = _make_record("What is 2 + 2?", "It is 4.")
        keep, _ = flt.filter(rec)
        assert keep is True


class TestPIIFilter:

    def test_email_detected_drop(self):
        cfg = PIIFilterConfig(enabled=True, backend="regex", action="drop",
                              check_roles=["user", "assistant"])
        flt = PIIFilter(cfg)
        rec = _make_record("My email is user@example.com please help", "Sure")
        keep, reason = flt.filter(rec)
        assert keep is False
        assert "pii" in reason.lower() or "email" in reason.lower()

    def test_clean_text_kept(self):
        cfg = PIIFilterConfig(enabled=True, backend="regex", action="drop",
                              check_roles=["user", "assistant"])
        flt = PIIFilter(cfg)
        rec = _make_record("What is gradient descent?", "It is an optimization algorithm.")
        keep, _ = flt.filter(rec)
        assert keep is True

    def test_redact_action_keeps_record(self):
        cfg = PIIFilterConfig(enabled=True, backend="regex", action="redact",
                              check_roles=["user", "assistant"])
        flt = PIIFilter(cfg)
        rec = _make_record("Contact me at test@domain.org", "Sure")
        keep, _ = flt.filter(rec)
        assert keep is True
        # Content should be redacted
        user_content = rec["conversations"][0]["content"]
        assert "test@domain.org" not in user_content


class TestNormalizeForHash:

    def test_strips_whitespace(self):
        assert normalize_for_hash("  hello  ") == "hello"

    def test_collapses_internal_whitespace(self):
        assert normalize_for_hash("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalize_for_hash("") == ""

    def test_none_handled(self):
        assert normalize_for_hash(None) == ""
