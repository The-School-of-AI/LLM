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
        """Compute tokenizer-based difficulty metrics and band."""
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
        
        # Assign band
        band, reason = self._assign_band(stats)
        
        return {
            "band": band,
            "reason": reason,
            "stats": stats
        }

    def _calculate_stats(self, tokens: List[int]) -> Dict[str, float]:
        token_array = np.array(tokens)
        return {
            "avg_token_id": float(np.mean(token_array)),
            "max_token_id": int(np.max(token_array)),
            "p95_token_id": float(np.percentile(token_array, 95)),
            "token_count": len(tokens)
        }

    def _assign_band(self, stats: Dict[str, float]) -> tuple[str, str]:
        """Assign band based on curriculum constraints."""
        # Check bands B0 to B5
        # We assume bands are ordered B0, B1, ... B5 in config if we iterate, 
        # but let's be explicit based on curriculum.yaml structure.
        
        # We want the *highest* band that fits? 
        # Actually the script logic was: "A record is assigned to the highest band (most difficult) that it qualifies for?"
        # Wait, usually lower bands have stricter (lower) max thresholds.
        # B0: max 10000. If I have max 5000, I fit B0. I also fit B1 (max 20000).
        # The script said: "Check bands from easiest (B0) to hardest (B5). We want the highest band that the sample qualifies for"
        # Wait, if avg=100. 
        # B0 limit 5000. 100 <= 5000. Fits B0.
        # B1 limit 10000. 100 <= 10000. Fits B1.
        # If it fits B0, it definitely fits B1, B2...
        # So "highest band it qualifies for" would mean B5? That doesn't make sense. B0 is "easiest".
        # If my score is low (easy), I should be B0.
        # If my score is high (hard), I should be B5.
        
        # Let's re-read the script logic in classify_curriculum_bands.py:
        # if avg <= B0_thresh AND max <= B0_thresh ... return B0.
        # else check B1...
        # This means we find the *first* (lowest) band that accommodates the stats.
        # Example: avg=8000.
        # B0 limit 5000. 8000 <= 5000 False. Not B0.
        # B1 limit 10000. 8000 <= 10000 True. It is B1.
        # So it classifies as the *lowest* band that contains the value.
        
        bands_order = ["B0", "B1", "B2", "B3", "B4", "B5"]
        
        for band in bands_order:
            # Get constraints from config
            # config path: difficulty_system.bands.{band}.constraints.tokenizer
            prefix = f"difficulty_system.bands.{band}.constraints.tokenizer"
            
            avg_max = self.config.get(f"{prefix}.avg_max")
            max_max = self.config.get(f"{prefix}.max_max")
            p95_max = self.config.get(f"{prefix}.p95_max")
            
            # Handle infinity strings from yaml if necessary, or missing values
            if avg_max == ".inf" or avg_max is None: avg_max = float("inf")
            if max_max == ".inf" or max_max is None: max_max = float("inf")
            if p95_max == ".inf" or p95_max is None: p95_max = float("inf")
            
            if (stats["avg_token_id"] <= avg_max and 
                stats["max_token_id"] <= max_max and 
                stats["p95_token_id"] <= p95_max):
                
                reason = f"Fits {band}: avg={stats['avg_token_id']:.1f}<={avg_max}, max={stats['max_token_id']}<={max_max}"
                return band, reason
                
        return "B5", "Exceeded all thresholds"

    def _empty_result(self):
        return {
            "band": "B0", # Default to simplest? Or None?
            "reason": "Empty text or tokens",
            "stats": {
                "avg_token_id": 0.0,
                "max_token_id": 0,
                "p95_token_id": 0.0,
                "token_count": 0
            }
        }
