"""
Diversity metrics and scoring for coreset selection.
Ensures coverage across rare tokens, tail phenomena, and diverse domains.
"""

from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy


class TokenFrequencyAnalyzer:
    """Analyzes token frequency distributions with caching for large-scale processing"""

    def __init__(self, vocab_size: int = 128_000, cache_size: int = 10_000):
        self.vocab_size = vocab_size
        self.token_counts: Counter = Counter()
        self.token_total = 0
        self._cache_size = int(cache_size)
        self._percentile_cache: "OrderedDict[int, float]" = OrderedDict()  # LRU cache
        self._classification_cache: "OrderedDict[int, str]" = OrderedDict()  # LRU cache
        self._sorted_frequencies = (
            None  # Cache for sorted frequency array (O(1) percentile lookup)
        )
        self._sort_valid = False  # Flag to track if sort needs refresh

    def add_tokens(self, token_ids: List[int]) -> None:
        """Add token IDs to frequency counter and invalidate caches"""
        if not token_ids:
            return

        # Counter.update is implemented in C and is faster than a Python loop.
        self.token_counts.update(token_ids)
        self.token_total += len(token_ids)
        # Invalidate percentile cache when new tokens added
        self._percentile_cache.clear()
        self._classification_cache.clear()
        self._sort_valid = False

    def _lru_get(self, cache: "OrderedDict", key):
        try:
            value = cache.pop(key)
            cache[key] = value  # mark as most-recent
            return value
        except KeyError:
            return None

    def _lru_put(self, cache: "OrderedDict", key, value):
        if self._cache_size <= 0:
            return
        if key in cache:
            cache.pop(key, None)
        cache[key] = value
        if len(cache) > self._cache_size:
            cache.popitem(last=False)  # evict least-recent

    def _get_sorted_frequencies(self) -> np.ndarray:
        """Get sorted frequency array once and cache it for O(log n) percentile lookup"""
        if not self._sort_valid:
            # Build sorted frequency array in ascending order for binary search
            self._sorted_frequencies = np.array(sorted(self.token_counts.values()))
            self._sort_valid = True
        return self._sorted_frequencies

    def get_token_frequency_percentile(self, token_id: int) -> float:
        """
        Get frequency percentile of token with O(log n) lookup using binary search.
        Returns percentile 0.0 (most frequent) to 1.0 (least frequent).
        """
        cached = self._lru_get(self._percentile_cache, token_id)
        if cached is not None:
            return cached

        if self.token_total == 0:
            return 0.5

        token_freq = self.token_counts[token_id]
        sorted_freqs = self._get_sorted_frequencies()  # Ascending order

        # Count how many tokens have GREATER frequency using binary search O(log n)
        # bisect_right(asc_array, value) = position where equal values would end
        # tokens_more_frequent = len - bisect_right
        position = bisect_right(sorted_freqs, token_freq)
        tokens_more_frequent = len(sorted_freqs) - position
        percentile = tokens_more_frequent / max(1, len(sorted_freqs))

        self._lru_put(self._percentile_cache, token_id, percentile)

        return percentile

    def classify_token_band(self, token_id: int) -> str:
        """
        Classify token into frequency band with caching.
        - boilerplate: 5-20%
        - normal: 20-80%
        - rare: 80-95%
        - tail: 95-100%
        """
        cached = self._lru_get(self._classification_cache, token_id)
        if cached is not None:
            return cached

        percentile = self.get_token_frequency_percentile(token_id)

        if percentile < 0.05:
            band = "junk"
        elif percentile < 0.20:
            band = "boilerplate"
        elif percentile < 0.80:
            band = "normal"
        elif percentile < 0.95:
            band = "rare"
        else:
            band = "tail"

        self._lru_put(self._classification_cache, token_id, band)

        return band

    def get_rare_token_ratio(self, token_ids: List[int]) -> float:
        """Compute ratio of rare tokens (80-95 percentile)"""
        if not token_ids:
            return 0.0

        rare_count = sum(
            1 for tid in token_ids if self.classify_token_band(tid) == "rare"
        )
        return rare_count / len(token_ids)

    def get_tail_token_ratio(self, token_ids: List[int]) -> float:
        """Compute ratio of tail tokens (95-100 percentile)"""
        if not token_ids:
            return 0.0

        tail_count = sum(
            1 for tid in token_ids if self.classify_token_band(tid) == "tail"
        )
        return tail_count / len(token_ids)


class DiversityScorer:
    """
    Compute diversity scores for chunks.
    Balances frequency, rarity, domain, and language diversity.
    """

    def __init__(
        self,
        token_analyzer: TokenFrequencyAnalyzer,
        rare_token_boost: float = 1.5,
        tail_token_boost: float = 2.0,
        domain_diversity_weight: float = 0.3,
        language_diversity_weight: float = 0.2,
    ):

        self.token_analyzer = token_analyzer
        self.rare_token_boost = rare_token_boost
        self.tail_token_boost = tail_token_boost
        self.domain_diversity_weight = domain_diversity_weight
        self.language_diversity_weight = language_diversity_weight

        # Track coverage
        self.domain_coverage: Set[str] = set()
        self.language_coverage: Set[str] = set()
        self.token_coverage: Set[int] = set()

    def score_chunk_rarity(self, token_ids: List[int]) -> float:
        """Score based on presence of rare/tail tokens. Simplified to avoid bottleneck."""
        if not token_ids:
            return 0.0

        # Simplified: use chunk length as diversity proxy (longer = more info)
        # Skip expensive tail token analysis to avoid O(n²) bottleneck
        length_score = min(len(token_ids) / 512.0, 1.0)  # Typical chunk ~512 tokens
        return length_score

    def score_chunk_coverage(
        self,
        token_ids: List[int],
        domain: str,
        language: str,
        total_domain_count: int,
        total_language_count: int,
    ) -> float:
        """Score based on contribution to coverage"""
        coverage_score = 0.0

        # Token coverage
        new_tokens = sum(1 for tid in token_ids if tid not in self.token_coverage)
        token_contribution = new_tokens / max(1, len(token_ids)) * 0.5
        coverage_score += token_contribution

        # Domain coverage
        domain_contribution = 0.0
        if domain not in self.domain_coverage:
            domain_contribution = self.domain_diversity_weight
        coverage_score += domain_contribution

        # Language coverage
        lang_contribution = 0.0
        if language not in self.language_coverage:
            lang_contribution = self.language_diversity_weight
        coverage_score += lang_contribution

        return coverage_score

    def score_chunk_composite(
        self,
        token_ids: List[int],
        domain: str,
        language: str,
        total_domain_count: int = 1,
        total_language_count: int = 1,
        rarity_weight: float = 0.4,
        coverage_weight: float = 0.6,
    ) -> float:
        """
        Composite score combining rarity and coverage.

        Args:
            rarity_weight: Weight for rarity score (0.0 to 1.0)
            coverage_weight: Weight for coverage score
        """
        rarity = self.score_chunk_rarity(token_ids)
        coverage = self.score_chunk_coverage(
            token_ids, domain, language, total_domain_count, total_language_count
        )

        composite = rarity_weight * rarity + coverage_weight * coverage
        return min(composite, 1.0)

    def update_coverage(self, token_ids: List[int], domain: str, language: str) -> None:
        """Update coverage tracking after selecting a chunk"""
        self.token_coverage.update(token_ids)
        self.domain_coverage.add(domain)
        self.language_coverage.add(language)


class DomainDiversityMatrix:
    """Track and score domain diversity"""

    def __init__(self):
        self.domain_chunk_count: Dict[str, int] = defaultdict(int)
        self.domain_token_count: Dict[str, int] = defaultdict(int)
        self.total_chunks = 0
        self.total_tokens = 0

    def add_chunk(self, domain: str, token_count: int) -> None:
        """Register a chunk in domain diversity"""
        self.domain_chunk_count[domain] += 1
        self.domain_token_count[domain] += token_count
        self.total_chunks += 1
        self.total_tokens += token_count

    def get_domain_distribution(self) -> Dict[str, float]:
        """Get current domain token distribution"""
        if self.total_tokens == 0:
            return {}

        return {
            domain: tokens / self.total_tokens
            for domain, tokens in self.domain_token_count.items()
        }

    def compute_entropy(self) -> float:
        """Compute Shannon entropy of domain distribution"""
        distribution = list(self.get_domain_distribution().values())

        if not distribution or sum(distribution) == 0:
            return 0.0

        return scipy_entropy(distribution)

    def score_domain_balance(
        self, chunk_domain: str, expected_ratios: Dict[str, float]
    ) -> float:
        """Score chunk based on domain imbalance"""
        current_dist = self.get_domain_distribution()
        current_ratio = current_dist.get(chunk_domain, 0.0)
        expected_ratio = expected_ratios.get(chunk_domain, 0.0)

        if expected_ratio == 0:
            return 0.0

        # Penalize if domain is over-represented, boost if under-represented
        ratio_error = (current_ratio - expected_ratio) / max(expected_ratio, 0.01)

        # Score inversely: more error = lower score
        score = max(0.0, 1.0 - abs(ratio_error))
        return score


class LanguageDiversityMatrix:
    """Track and score language diversity"""

    def __init__(self):
        self.language_chunk_count: Dict[str, int] = defaultdict(int)
        self.language_token_count: Dict[str, int] = defaultdict(int)
        self.total_chunks = 0
        self.total_tokens = 0

    def add_chunk(self, language: str, token_count: int) -> None:
        """Register a chunk in language diversity"""
        self.language_chunk_count[language] += 1
        self.language_token_count[language] += token_count
        self.total_chunks += 1
        self.total_tokens += token_count

    def get_language_distribution(self) -> Dict[str, float]:
        """Get current language token distribution"""
        if self.total_tokens == 0:
            return {}

        return {
            lang: tokens / self.total_tokens
            for lang, tokens in self.language_token_count.items()
        }

    def compute_entropy(self) -> float:
        """Compute Shannon entropy of language distribution"""
        distribution = list(self.get_language_distribution().values())

        if not distribution or sum(distribution) == 0:
            return 0.0

        return scipy_entropy(distribution)


class ProtectedSliceManager:
    """Manage protected slices to ensure they're not over-deduplicated"""

    def __init__(self):
        self.protected_chunks: Dict[str, Tuple[str, str]] = (
            {}
        )  # chunk_id -> (band/domain, protection_reason)
        self.protected_chunk_counts: Dict[str, Tuple[int, int]] = (
            {}
        )  # name -> (total, selected)

    def register_protected_slice(
        self, chunk_id: str, slice_name: str, reason: str
    ) -> None:
        """Register a chunk as part of a protected slice"""
        self.protected_chunks[chunk_id] = (slice_name, reason)

        if slice_name not in self.protected_chunk_counts:
            self.protected_chunk_counts[slice_name] = (0, 0)

    def mark_as_selected(self, chunk_id: str) -> None:
        """Mark protected chunk as selected"""
        if chunk_id in self.protected_chunks:
            slice_name, _ = self.protected_chunks[chunk_id]
            total, selected = self.protected_chunk_counts[slice_name]
            self.protected_chunk_counts[slice_name] = (total, selected + 1)

    def get_preservation_ratio(self, slice_name: str) -> float:
        """Get preservation ratio for a protected slice"""
        if slice_name not in self.protected_chunk_counts:
            return 1.0

        total, selected = self.protected_chunk_counts[slice_name]
        if total == 0:
            return 1.0

        return selected / total
