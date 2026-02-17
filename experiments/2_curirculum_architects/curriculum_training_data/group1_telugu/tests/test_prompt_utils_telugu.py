#!/usr/bin/env python3
"""Tests for Telugu prompt utilities (prompt_utils_telugu.py)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from group1_telugu.prompt_utils_telugu import (
    combine_qa_pairs_to_reach_min_tokens_telugu,
    count_tokens_telugu,
    ensure_answer_period,
    ensure_query_punctuation,
    format_qa_pair_telugu,
)

# ── Token counting ──


class TestCountTokensTelugu:
    def test_telugu_chars_each_one_token(self):
        # Each Telugu Unicode char = 1 token
        assert count_tokens_telugu("క") >= 1

    def test_telugu_word(self):
        # "పుస్తకం" has 5 Unicode chars in Telugu block (excluding virama/dependent forms)
        result = count_tokens_telugu("పుస్తకం")
        assert result > 0

    def test_empty_string(self):
        assert count_tokens_telugu("") == 0

    def test_whitespace_only(self):
        assert count_tokens_telugu("   ") == 0

    def test_english_word(self):
        # "hello" = 1 token (word unit)
        assert count_tokens_telugu("hello") == 1

    def test_mixed_telugu_english(self):
        result = count_tokens_telugu("క hello")
        assert result >= 2  # at least 1 Telugu + 1 English

    def test_punctuation_counted(self):
        # "?" and "." are each 1 token
        result = count_tokens_telugu("?.")
        assert result == 2

    def test_telugu_range_detection(self):
        """Characters in U+0C00-U+0C7F should be counted as individual tokens."""
        # Telugu vowels
        assert count_tokens_telugu("అ") == 1
        assert count_tokens_telugu("ఆ") == 1
        # Telugu consonant
        assert count_tokens_telugu("క") == 1

    def test_kannada_range_detection(self):
        """Kannada chars (U+0C80-U+0CFF) should also be individual tokens."""
        assert count_tokens_telugu("ಕ") == 1

    def test_devanagari_range_detection(self):
        """Devanagari chars (U+0900-U+097F) should also be individual tokens."""
        assert count_tokens_telugu("क") == 1


# ── Formatting helpers ──


class TestEnsureAnswerPeriod:
    def test_adds_period(self):
        assert ensure_answer_period("answer") == "answer."

    def test_already_has_period(self):
        assert ensure_answer_period("answer.") == "answer."

    def test_strips_whitespace(self):
        assert ensure_answer_period("  answer  ") == "answer."


class TestEnsureQueryPunctuation:
    def test_adds_question_mark(self):
        assert ensure_query_punctuation("query") == "query?"

    def test_already_has_question_mark(self):
        assert ensure_query_punctuation("query?") == "query?"

    def test_strips_trailing_punctuation(self):
        assert ensure_query_punctuation("query.") == "query?"

    def test_empty_string(self):
        assert ensure_query_punctuation("") == ""


class TestFormatQaPairTelugu:
    def test_basic_format(self):
        result = format_qa_pair_telugu("question", "answer")
        assert result == "question? answer."

    def test_preserves_existing_punctuation(self):
        result = format_qa_pair_telugu("question?", "answer.")
        assert result == "question? answer."

    def test_telugu_qa(self):
        result = format_qa_pair_telugu(
            '"పుస్తకం" పదంలో ఎన్ని అక్షరాలు ఉన్నాయి',
            "3 అక్షరాలు",
        )
        assert result.endswith(".")
        assert "?" in result

    def test_uses_period_not_danda(self):
        """Telugu uses period (.), NOT danda (।)."""
        result = format_qa_pair_telugu("question", "answer")
        assert "।" not in result
        assert result.endswith(".")


# ── Combining QA pairs ──


class TestCombineQaPairs:
    def test_empty_input(self):
        assert combine_qa_pairs_to_reach_min_tokens_telugu([]) == []

    def test_combines_to_min_tokens(self):
        # Create pairs that are individually short
        pairs = [("question" * 5 + "?", "answer" * 5) for _ in range(200)]
        result = combine_qa_pairs_to_reach_min_tokens_telugu(pairs, min_tokens=50)
        assert len(result) > 0
        # All lines should meet minimum (except possibly last merged into previous)
        for line in result:
            tokens = count_tokens_telugu(line)
            assert tokens >= 50, f"Line has only {tokens} tokens"

    def test_single_pair(self):
        pairs = [("q?", "a")]
        result = combine_qa_pairs_to_reach_min_tokens_telugu(pairs, min_tokens=1)
        assert len(result) == 1

    def test_output_is_strings(self):
        pairs = [("q?", "a") for _ in range(10)]
        result = combine_qa_pairs_to_reach_min_tokens_telugu(pairs, min_tokens=5)
        assert all(isinstance(s, str) for s in result)
