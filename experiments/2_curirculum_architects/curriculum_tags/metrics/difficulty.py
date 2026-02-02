"""Difficulty band classification metric."""

import math
import re
from collections import Counter
from typing import Any, Dict

from ..core.plugin import MetricPlugin


class DifficultyMetric(MetricPlugin):
    """Classify text into difficulty bands using curriculum thresholds."""

    name = "difficulty"

    def __init__(self, config):
        super().__init__(config)
        # Get band thresholds from curriculum if available
        self.bands = config.get("difficulty.bands", self._default_bands())

    def _default_bands(self) -> Dict[str, float]:
        """Default difficulty band thresholds."""
        return {
            "B0": 0.15,
            "B1": 0.30,
            "B2": 0.50,
            "B3": 0.70,
            "B4": 0.85,
            "B5": 1.00,
        }

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute difficulty score and band assignment.

        Returns:
            band: Assigned difficulty band (B0-B5)
            score: Continuous difficulty score (0-1)
            features: Component features
        """
        text = sample.get("text", "")

        # Extract features
        features = self._extract_features(text)

        # Compute composite score
        score = self._compute_score(features)

        # Assign band
        band = self._assign_band(score)

        return {
            "band": band,
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
            -sum((count / total_chars) * math.log2(count / total_chars) for count in char_freq.values())
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

    def _assign_band(self, score: float) -> str:
        """Map score to difficulty band using curriculum thresholds."""
        for band_name, threshold in self.bands.items():
            if score < threshold:
                return band_name
        return list(self.bands.keys())[-1]  # Return highest band

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"\b[\w\']+\b", text.lower())
