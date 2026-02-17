#!/usr/bin/env python3
"""Tests for Telugu akshara segmentation (telugu_grammar.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from group1_telugu.telugu_grammar import _ends_with_virama, get_telugu_aksharas

# ── Virama detection ──


class TestEndsWithVirama:
    def test_virama_present(self):
        assert _ends_with_virama("క్") is True

    def test_virama_absent(self):
        assert _ends_with_virama("క") is False

    def test_empty_string(self):
        assert _ends_with_virama("") is False

    def test_virama_only(self):
        assert _ends_with_virama("\u0c4d") is True


# ── Core akshara segmentation ──


class TestAksharaSegmentation:
    """Test akshara segmentation against known Telugu words."""

    @pytest.mark.parametrize(
        "word, expected",
        [
            # Simple words (no conjuncts)
            ("బడి", ["బ", "డి"]),
            ("నీరు", ["నీ", "రు"]),
            ("ఆవు", ["ఆ", "వు"]),
            ("ఇల్లు", ["ఇ", "ల్లు"]),
            # Doubled consonants (gemination)
            ("అమ్మ", ["అ", "మ్మ"]),
            ("కుక్క", ["కు", "క్క"]),
            ("పిల్లి", ["పి", "ల్లి"]),
            ("గొర్రె", ["గొ", "ర్రె"]),
            # Complex conjuncts
            ("పుస్తకం", ["పు", "స్త", "కం"]),
            ("విద్యార్థి", ["వి", "ద్యా", "ర్థి"]),
            ("జ్ఞానం", ["జ్ఞా", "నం"]),
            ("విద్యాలయం", ["వి", "ద్యా", "ల", "యం"]),
            # Anusvara (ం) stays with preceding
            ("నగరం", ["న", "గ", "రం"]),
            ("పుస్తకం", ["పు", "స్త", "కం"]),
            # Single akshara words
            ("ఆ", ["ఆ"]),
        ],
    )
    def test_known_words(self, word, expected):
        result = get_telugu_aksharas(word)
        assert result == expected, f"{word}: got {result}, expected {expected}"

    def test_empty_string(self):
        assert get_telugu_aksharas("") == []

    def test_returns_list(self):
        result = get_telugu_aksharas("అమ్మ")
        assert isinstance(result, list)
        assert all(isinstance(a, str) for a in result)

    def test_all_aksharas_nonempty(self):
        """Every akshara in the result should be non-empty."""
        words = ["పుస్తకం", "విద్యార్థి", "జ్ఞానం", "కుక్క", "బడి"]
        for word in words:
            aksharas = get_telugu_aksharas(word)
            assert all(len(a) > 0 for a in aksharas), f"{word}: empty akshara found"

    def test_reconstruct_word(self):
        """Joining aksharas back should reconstruct the original word."""
        words = ["పుస్తకం", "అమ్మ", "విద్యాలయం", "నీరు", "జ్ఞానం", "కుక్క"]
        for word in words:
            aksharas = get_telugu_aksharas(word)
            reconstructed = "".join(aksharas)
            assert reconstructed == word, f"{word}: reconstructed as {reconstructed}"

    def test_count_aksharas(self):
        """Verify akshara counts for key words."""
        cases = [
            ("పుస్తకం", 3),
            ("అమ్మ", 2),
            ("నీరు", 2),
            ("విద్యార్థి", 3),
            ("జ్ఞానం", 2),
            ("బడి", 2),
            ("విద్యాలయం", 4),
        ]
        for word, expected_count in cases:
            result = get_telugu_aksharas(word)
            assert (
                len(result) == expected_count
            ), f"{word}: got {len(result)} aksharas, expected {expected_count}"
