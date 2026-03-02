"""Readability metrics for text analysis."""

import re
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class ReadabilityMetric(MetricPlugin):
    """Compute readability metrics for text samples."""

    name = "readability"

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute readability scores.

        Returns:
            flesch_kincaid_grade: FK grade level
            avg_sentence_length: Average words per sentence
            avg_word_length: Average characters per word
        """
        text = sample.get("text", "")

        # Tokenize
        sentences = self._split_sentences(text)
        tokens = self._tokenize(text)

        if not tokens or not sentences:
            return {
                "flesch_kincaid_grade": 0.0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
            }

        # Compute metrics
        n_tokens = len(tokens)
        n_sentences = len(sentences)
        n_syllables = sum(self._count_syllables(t) for t in tokens)

        avg_sent_len = n_tokens / n_sentences
        avg_word_len = sum(len(t) for t in tokens) / n_tokens

        # Flesch-Kincaid grade level
        fk_grade = 0.39 * avg_sent_len + 11.8 * (n_syllables / n_tokens) - 15.59

        return {
            "flesch_kincaid_grade": round(max(0, fk_grade), 2),
            "avg_sentence_length": round(avg_sent_len, 2),
            "avg_word_length": round(avg_word_len, 2),
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Extract words from text."""
        return re.findall(r"\b[\w\']+\b", text.lower())

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r"[.!?]+\s+", text.strip())
        return [s for s in sentences if s.strip()]

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Estimate syllable count for word."""
        vowels = re.findall(r"[aeiouy]+", word.lower())
        return max(1, len(vowels))
