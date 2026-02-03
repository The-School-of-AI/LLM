import zlib
from typing import Any, Dict


class DifficultyScorer:
    """
    Assigns a difficulty/complexity score to a text chunk.
    Higher score = More complex/difficult.
    """

    def __init__(self, method: str = "hybrid"):
        self.method = method

    def score(self, record: Dict[str, Any]) -> float:
        """
        Returns a float score.
        If record has 'perplexity', use it.
        Else, calculate 'compression_ratio' (Higher compression ratio usually means simpler text,
        but strictly speaking, higher entropy = harder.
        Here we define Difficulty ~ Entropy ~ 1/CompressionRatio?
        Actually, let's use:
        - raw_ppl if available.
        - zlib_entropy: len(compressed) / len(raw).
          Random string ~ 1.0 (Hard). Repeated 'a' ~ 0.001 (Easy).
        """
        # 1. Prefer pre-computed perplexity
        if "perplexity" in record and record["perplexity"] is not None:
            return float(record["perplexity"])

        # 2. Key "difficulty" might already exist
        if "difficulty" in record and record["difficulty"] is not None:
            return float(record["difficulty"])

        # 3. Fallback: Zlib Entropy Proxy
        text = record.get("text", "")
        if not text:
            return 0.0

        text_bytes = text.encode("utf-8")
        compressed = zlib.compress(text_bytes)

        # Ratio: CompressedSize / RawSize.
        # High randomness (high info density) -> Ratio close to 1.0 -> "Hard"
        # High redundancy (low info density) -> Ratio close to 0.0 -> "Easy"
        # We assume "Hard" = useful for later stages.
        return len(compressed) / len(text_bytes)


class QualityScorer:
    """
    Simple heuristics to reject very low quality data.
    """

    def score(self, record: Dict[str, Any]) -> float:
        # Placeholder for complex quality model.
        # For now, return length as a naive proxy (too short = bad)
        text = record.get("text", "")
        if len(text) < 50:
            return 0.0  # Reject
        return 1.0
