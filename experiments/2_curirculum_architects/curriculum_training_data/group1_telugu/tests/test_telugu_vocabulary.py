#!/usr/bin/env python3
"""Tests for Telugu vocabulary (telugu_vocabulary.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from group1_telugu.telugu_grammar import get_telugu_aksharas
from group1_telugu.telugu_vocabulary import (
    ALL_WORDS_UNIQUE,
    CLASSIFICATION_CATEGORIES,
    DAYS_OF_WEEK,
    EASY_ANIMALS,
    EASY_BODY_PARTS,
    EASY_COLORS,
    EASY_FOOD,
    EASY_NATURE,
    EASY_OBJECTS,
    EASY_PEOPLE,
    EASY_WORDS_UNIQUE,
    HARD_ABSTRACT,
    HARD_COMPLEX_NOUNS,
    HARD_WORDS_UNIQUE,
    MEDIUM_ANIMALS,
    MEDIUM_FOOD,
    MEDIUM_HOUSEHOLD,
    MEDIUM_NATURE,
    MEDIUM_OBJECTS,
    MEDIUM_PROFESSIONS,
    MEDIUM_VEHICLES,
    MEDIUM_WORDS_UNIQUE,
    MONTHS,
    NUMBERS,
    RHYMING_PAIRS,
    VARGAS,
)

# ── Word count thresholds ──


class TestWordCounts:
    def test_total_unique_words_at_least_950(self):
        assert (
            len(ALL_WORDS_UNIQUE) >= 950
        ), f"Need >= 950 unique words, got {len(ALL_WORDS_UNIQUE)}"

    def test_easy_words_nonempty(self):
        assert len(EASY_WORDS_UNIQUE) > 0

    def test_medium_words_nonempty(self):
        assert len(MEDIUM_WORDS_UNIQUE) > 0

    def test_hard_words_nonempty(self):
        assert len(HARD_WORDS_UNIQUE) > 0

    def test_no_duplicates_in_unique_lists(self):
        assert len(EASY_WORDS_UNIQUE) == len(set(EASY_WORDS_UNIQUE))
        assert len(MEDIUM_WORDS_UNIQUE) == len(set(MEDIUM_WORDS_UNIQUE))
        assert len(HARD_WORDS_UNIQUE) == len(set(HARD_WORDS_UNIQUE))
        assert len(ALL_WORDS_UNIQUE) == len(set(ALL_WORDS_UNIQUE))


# ── Category lists non-empty ──


class TestCategoryLists:
    @pytest.mark.parametrize(
        "category, min_count",
        [
            (EASY_ANIMALS, 30),
            (EASY_OBJECTS, 50),
            (EASY_BODY_PARTS, 20),
            (EASY_COLORS, 10),
            (EASY_NATURE, 30),
            (EASY_PEOPLE, 20),
            (EASY_FOOD, 30),
            (MEDIUM_ANIMALS, 20),
            (MEDIUM_OBJECTS, 40),
            (MEDIUM_PROFESSIONS, 30),
            (MEDIUM_NATURE, 40),
            (MEDIUM_VEHICLES, 20),
            (MEDIUM_FOOD, 50),
            (MEDIUM_HOUSEHOLD, 30),
            (HARD_COMPLEX_NOUNS, 20),
            (HARD_ABSTRACT, 20),
        ],
    )
    def test_category_min_count(self, category, min_count):
        assert (
            len(category) >= min_count
        ), f"Category has {len(category)} words, need >= {min_count}"


# ── Telugu script validation ──


class TestTeluguScript:
    def _is_telugu_char(self, ch):
        return "\u0c00" <= ch <= "\u0c7f"

    def test_all_words_contain_telugu(self):
        """Every word must contain at least one Telugu character."""
        for word in ALL_WORDS_UNIQUE:
            has_telugu = any(self._is_telugu_char(ch) for ch in word)
            assert has_telugu, f"Word '{word}' has no Telugu characters"

    def test_no_kannada_chars(self):
        """No word should contain Kannada characters."""
        for word in ALL_WORDS_UNIQUE:
            for ch in word:
                assert not (
                    "\u0c80" <= ch <= "\u0cff"
                ), f"Word '{word}' contains Kannada character: {ch}"

    def test_no_devanagari_chars(self):
        """No word should contain Devanagari characters."""
        for word in ALL_WORDS_UNIQUE:
            for ch in word:
                assert not (
                    "\u0900" <= ch <= "\u097f"
                ), f"Word '{word}' contains Devanagari character: {ch}"

    def test_no_empty_words(self):
        """No word should be empty or whitespace-only."""
        for word in ALL_WORDS_UNIQUE:
            assert word.strip(), "Empty or whitespace-only word found"


# ── Numbers ──


class TestNumbers:
    def test_exactly_100_numbers(self):
        assert len(NUMBERS) == 100

    def test_first_number_is_okati(self):
        assert NUMBERS[0] == "ఒకటి"

    def test_last_number_is_vanda(self):
        assert NUMBERS[99] == "వంద"

    def test_tens(self):
        # index 9 = 10, 19 = 20, 29 = 30, etc.
        assert NUMBERS[9] == "పది"
        assert NUMBERS[19] == "ఇరవై"
        assert NUMBERS[29] == "ముప్పై"
        assert NUMBERS[39] == "నలభై"
        assert NUMBERS[49] == "యాభై"

    def test_all_numbers_have_telugu(self):
        for i, word in enumerate(NUMBERS):
            has_telugu = any("\u0c00" <= ch <= "\u0c7f" for ch in word)
            assert has_telugu, f"Number {i+1} ('{word}') has no Telugu chars"

    def test_all_numbers_nonempty(self):
        for i, word in enumerate(NUMBERS):
            assert word.strip(), f"Number {i+1} is empty"


# ── Days and Months ──


class TestDaysMonths:
    def test_seven_days(self):
        assert len(DAYS_OF_WEEK) == 7

    def test_twelve_months(self):
        assert len(MONTHS) == 12

    def test_sunday_first(self):
        assert DAYS_OF_WEEK[0] == "ఆదివారం"

    def test_saturday_last(self):
        assert DAYS_OF_WEEK[6] == "శనివారం"


# ── Vargas ──


class TestVargas:
    def test_seven_varga_groups(self):
        assert len(VARGAS) == 7

    def test_ka_varga(self):
        assert "క" in VARGAS
        assert VARGAS["క"] == ["క", "ఖ", "గ", "ఘ", "ఙ"]

    def test_ta_varga(self):
        assert "త" in VARGAS
        assert VARGAS["త"] == ["త", "థ", "ద", "ధ", "న"]

    def test_pa_varga(self):
        assert "ప" in VARGAS
        assert VARGAS["ప"] == ["ప", "ఫ", "బ", "భ", "మ"]

    def test_all_vargas_have_members(self):
        for key, members in VARGAS.items():
            assert len(members) >= 4, f"Varga '{key}' has only {len(members)} members"


# ── Classification ──


class TestClassification:
    def test_three_categories(self):
        assert len(CLASSIFICATION_CATEGORIES) == 3

    def test_expected_keys(self):
        assert "జంతువు" in CLASSIFICATION_CATEGORIES
        assert "వ్యక్తి" in CLASSIFICATION_CATEGORIES
        assert "వస్తువు" in CLASSIFICATION_CATEGORIES

    def test_categories_nonempty(self):
        for key, words in CLASSIFICATION_CATEGORIES.items():
            assert len(words) > 0, f"Category '{key}' is empty"

    def test_total_classification_words(self):
        total = sum(len(v) for v in CLASSIFICATION_CATEGORIES.values())
        assert total >= 200, f"Only {total} classification words, need >= 200"


# ── Rhyming pairs ──


class TestRhymingPairs:
    def test_at_least_100_pairs(self):
        assert (
            len(RHYMING_PAIRS) >= 100
        ), f"Only {len(RHYMING_PAIRS)} rhyming pairs, need >= 100"

    def test_pairs_are_different_words(self):
        for word, rhyme in RHYMING_PAIRS.items():
            assert word != rhyme, f"Word '{word}' rhymes with itself"

    def test_pairs_share_last_akshara(self):
        """Rhyming pairs should share the same last akshara."""
        mismatches = 0
        for word, rhyme in list(RHYMING_PAIRS.items())[:50]:
            w_aksharas = get_telugu_aksharas(word)
            r_aksharas = get_telugu_aksharas(rhyme)
            if w_aksharas and r_aksharas:
                if w_aksharas[-1] != r_aksharas[-1]:
                    mismatches += 1
        # Allow small tolerance (some edge cases in vocabulary)
        assert mismatches <= 5, f"{mismatches} pairs don't share last akshara"

    def test_all_pair_words_exist_in_vocabulary(self):
        """All words in rhyming pairs should be from vocabulary."""
        vocab_set = set(ALL_WORDS_UNIQUE)
        for word in RHYMING_PAIRS:
            assert word in vocab_set, f"Rhyming word '{word}' not in vocabulary"
