"""Difficulty band classification metric."""

import math
import re
from collections import Counter
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class DifficultyMetric(MetricPlugin):
    """Classify text into difficulty levels based on linguistic features."""

    name = "difficulty"

    def __init__(self, config):
        super().__init__(config)
        # Get band thresholds from curriculum if available
        self.levels = {
            "L0": 0.1,
            "L1": 0.3,
            "L2": 0.5,
            "L3": 0.7,
            "L4": 0.9,
            "L5": 1.0,
        }

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute difficulty score and band assignment.

        Returns:
            level: Assigned difficulty level (L0-L5)
            score: Continuous difficulty score (0-1)
            features: Component features
        """
        text = sample.get("text", "")

        # Extract features
        features = self._extract_features(text)

        # Compute composite score
        score = self._compute_score(features)

        # Assign level
        level = self._assign_level(score)

        return {
            "level": level,
            "score": round(score, 3),
            "features": features,
        }

    def _extract_features(self, text: str) -> Dict[str, float]:
        """Extract difficulty-related features."""
        tokens = self._tokenize(text)
        n_tokens = len(tokens)

        if n_tokens < 10:
            return {
                "token_count": n_tokens,
                "avg_word_length": 0.0,
                "rare_ratio": 0.0,
                "entropy": 0.0,
            }

        # Word length
        avg_word_len = sum(len(t) for t in tokens) / n_tokens

        # Rare words (frequency = 1)
        freq = Counter(tokens)
        rare_count = sum(1 for t in tokens if freq[t] == 1)
        rare_ratio = rare_count / n_tokens

        # Character entropy
        char_freq = Counter(text[:2000])  # Sample for efficiency
        total_chars = sum(char_freq.values())
        entropy = (
            -sum(
                (count / total_chars) * math.log2(count / total_chars)
                for count in char_freq.values()
            )
            if total_chars > 0
            else 0.0
        )

        return {
            "token_count": n_tokens,
            "avg_word_length": round(avg_word_len, 2),
            "rare_ratio": round(rare_ratio, 3),
            "entropy": round(entropy, 3),
        }

    def _compute_score(self, features: Dict[str, float]) -> float:
        """Compute normalized difficulty score (0-1)."""
        # Weighted combination
        score = 0.0
        score += 0.3 * min(features["avg_word_length"] / 10.0, 1.0)
        score += 0.4 * features["rare_ratio"]
        score += 0.3 * min(features["entropy"] / 5.0, 1.0)

        return min(max(score, 0.0), 1.0)

    def _assign_level(self, score: float) -> str:
        """Map score to difficulty level using curriculum thresholds."""
        for level_name, threshold in self.levels.items():
            if score < threshold:
                return level_name
        return list(self.levels.keys())[-1]  # Return highest level

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"\b[\w\']+\b", text.lower())
