"""Tokenizer-based difficulty metric using curriculum-style bands."""

from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer

from ..core.plugin import MetricPlugin


class TokenizerDifficultyMetric(MetricPlugin):
    """Classify text into tokenizer-based difficulty levels T0-T5."""

    name = "tokenizer_difficulty"

    TOKEN_ID_THRESHOLDS: Dict[str, Dict[str, Any]] = {
        "T0": {
            "avg_max": 6227,
            "p95_max": 29826,
            "max_max": 76039,
            "description": "T0: Very high frequency tokens, very simple language",
        },
        "T1": {
            "avg_max": 7295,
            "p95_max": 36256,
            "max_max": 87078,
            "description": "T1: High frequency tokens, everyday language",
        },
        "T2": {
            "avg_max": 8493,
            "p95_max": 43655,
            "max_max": 94111,
            "description": "T2: Medium frequency, structured knowledge",
        },
        "T3": {
            "avg_max": 9806,
            "p95_max": 51143,
            "max_max": 97799,
            "description": "T3: Lower frequency, technical content",
        },
        "T4": {
            "avg_max": 11270,
            "p95_max": 59242,
            "max_max": 99616,
            "description": "T4: Low frequency, complex reasoning",
        },
        "T5": {
            "avg_max": float("inf"),
            "p95_max": float("inf"),
            "max_max": float("inf"),
            "description": "T5: Very low frequency, advanced/rare terms",
        },
    }

    def __init__(self, config):
        super().__init__(config)

        self.model_id = config.get(
            "tokenizer_proxy.model_id", "meta-llama/Llama-3.3-70B-Instruct"
        )

        self.tokenizer = self._load_tokenizer()

    def _load_tokenizer(self):
        """Load tokenizer from local folder."""
        try:
            return AutoTokenizer.from_pretrained(
                "tokenizer", use_fast=True, local_files_only=True
            )
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            return None

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute tokenizer-based difficulty level."""
        text = sample.get("text", "")
        if not text:
            return self._empty_result()

        if not self.tokenizer:
            return {
                "error": "Tokenizer not found",
            }

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)

        if not token_ids:
            return self._empty_result()

        features = self._calculate_stats(token_ids)
        level, meta = self._assign_level(features)

        levels_order = ["T0", "T1", "T2", "T3", "T4", "T5"]
        idx = levels_order.index(level) if level in levels_order else 0
        score = idx / (len(levels_order) - 1)

        return {
            "level": level,
            "score": round(float(score), 3),
            "level_description": meta.get("description"),
            "level_reason": meta.get("reason"),
            "features": {**features},
        }

    def _calculate_stats(self, tokens: List[int]) -> Dict[str, float]:
        """Compute token ID statistics."""
        token_array = np.array(tokens)
        return {
            "avg_token_id": float(np.mean(token_array)),
            "max_token_id": int(np.max(token_array)),
            "min_token_id": int(np.min(token_array)),
            "p50_token_id": float(np.percentile(token_array, 50)),
            "p95_token_id": float(np.percentile(token_array, 95)),
            "p99_token_id": float(np.percentile(token_array, 99)),
            "token_count": int(len(tokens)),
        }

    def _assign_level(
        self, token_stats: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        """Assign a T0-T5 level based on token ID thresholds."""
        avg_id = token_stats["avg_token_id"]
        max_id = token_stats["max_token_id"]
        p95_id = token_stats["p95_token_id"]

        for level in ["T0", "T1", "T2", "T3", "T4", "T5"]:
            th = self.TOKEN_ID_THRESHOLDS[level]
            if (
                avg_id <= th["avg_max"]
                and p95_id <= th["p95_max"]
                and max_id <= th["max_max"]
            ):
                return level, {
                    "description": th["description"],
                    "reason": f"avg={avg_id:.0f}, p95={p95_id:.0f}, max={max_id}",
                }

        return "T5", {
            "description": self.TOKEN_ID_THRESHOLDS["T5"]["description"],
            "reason": "Exceeds all thresholds",
        }

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result for missing/empty text."""
        return {
            "level": "T0",
            "score": 0.0,
            "level_description": "No text",
            "level_reason": "Empty input",
            "features": {},
        }
