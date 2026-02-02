"""Tokenizer-based difficulty metric."""

import numpy as np
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer

from ..core.plugin import MetricPlugin


class TokenizerDifficultyMetric(MetricPlugin):
    """Classify text into difficulty bands using tokenizer frequency statistics."""

    name = "tokenizer_difficulty"

    def __init__(self, config):
        super().__init__(config)
        
        # Load tokenizer configuration
        # Ideally this comes from config, but we'll default to a reasonable value or raise if needed.
        # For now, we use a default if not specified, but this should likely be configured.
        self.model_id = config.get("tokenizer_proxy.model_id", "meta-llama/Llama-3.3-70B-Instruct")
        self.tokenizer_path = config.get("tokenizer_proxy.local_path", None)
        
        self.tokenizer = self._load_tokenizer()
        
    def _load_tokenizer(self):
        """Load the tokenizer."""
        try:
            if self.tokenizer_path:
                return AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True, local_files_only=True)
            else:
                return AutoTokenizer.from_pretrained(self.model_id)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer {self.model_id}: {e}")
            # Fallback for testing/development if primary fails
            try:
                fallback = "gpt2"
                print(f"Attempting fallback to {fallback}...")
                return AutoTokenizer.from_pretrained(fallback)
            except Exception as e2:
                print(f"Fallback failed: {e2}")
                return None

    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute tokenizer-based difficulty metrics."""
        if not self.tokenizer:
             return {"error": "Tokenizer not loaded"}

        text = sample.get("text", "")
        if not text:
             return self._empty_result()

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if not tokens:
            return self._empty_result()
            
        # Calculate stats
        stats = self._calculate_stats(tokens)
        
        return stats

    def _calculate_stats(self, tokens: List[int]) -> Dict[str, float]:
        token_array = np.array(tokens)
        return {
            "avg_token_id": float(np.mean(token_array)),
            "max_token_id": int(np.max(token_array)),
            "p95_token_id": float(np.percentile(token_array, 95)),
            "token_count": len(tokens)
        }

    def _empty_result(self):
        return {
            "avg_token_id": 0.0,
            "max_token_id": 0,
            "p95_token_id": 0.0,
            "token_count": 0
        }
