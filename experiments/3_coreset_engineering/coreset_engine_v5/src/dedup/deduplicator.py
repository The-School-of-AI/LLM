"""
Deduplication module for coreset selection.
Supports exact and near-duplicate detection via hashing and similarity.
"""

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import xxhash


@dataclass
class DuplicateMatch:
    """Result of duplicate detection"""

    chunk_id_1: str
    chunk_id_2: str
    similarity: float  # 0.0 to 1.0
    is_exact: bool


class ExactDeduplicator:
    """Exact duplicate detection using hash signatures"""

    def __init__(self, hash_algorithm: str = "xxhash64"):
        self.hash_algorithm = hash_algorithm
        self.chunk_hashes: Dict[str, str] = {}
        self.hash_to_chunks: Dict[str, List[str]] = defaultdict(list)
        self.exact_duplicates: Set[Tuple[str, str]] = set()

    def compute_hash(self, chunk_id: str, chunk_text: str) -> str:
        """Compute hash of chunk"""
        if self.hash_algorithm == "xxhash64":
            h = xxhash.xxh64(chunk_text.encode()).hexdigest()
        elif self.hash_algorithm == "sha256":
            h = hashlib.sha256(chunk_text.encode()).hexdigest()
        else:
            raise ValueError(f"Unknown hash algorithm: {self.hash_algorithm}")

        self.chunk_hashes[chunk_id] = h
        self.hash_to_chunks[h].append(chunk_id)
        return h

    def find_exact_duplicates(self) -> List[Tuple[str, str]]:
        """Find all exact duplicate pairs"""
        duplicates = []

        for hash_val, chunk_ids in self.hash_to_chunks.items():
            if len(chunk_ids) > 1:
                # All chunks with same hash are duplicates
                for i in range(len(chunk_ids)):
                    for j in range(i + 1, len(chunk_ids)):
                        pair = tuple(sorted([chunk_ids[i], chunk_ids[j]]))
                        if pair not in self.exact_duplicates:
                            self.exact_duplicates.add(pair)
                            duplicates.append(pair)

        return duplicates

    def get_canonical_chunk(self, duplicate_pair: Tuple[str, str]) -> str:
        """Get the canonical (kept) chunk from a duplicate pair"""
        # Keep the one that appears first (by ID)
        return min(duplicate_pair)

    def mark_removed(self, chunk_id: str) -> None:
        """Mark a chunk as removed from deduplication"""
        if chunk_id in self.chunk_hashes:
            hash_val = self.chunk_hashes[chunk_id]
            if chunk_id in self.hash_to_chunks[hash_val]:
                self.hash_to_chunks[hash_val].remove(chunk_id)


class SimHasher:
    """SimHash-based approximate deduplication"""

    @staticmethod
    def compute_simhash(text: str, hash_size: int = 64) -> int:
        """
        Compute SimHash of text.
        SimHash creates a compact fingerprint that preserves similarity.
        """
        # Split into tokens (simple split on whitespace)
        tokens = text.lower().split()

        # Initialize hash vector
        hash_vector = [0] * hash_size

        # For each token, compute hash and update vector
        for token in tokens:
            # Compute hash of token
            token_hash = int(hashlib.sha256(token.encode()).hexdigest(), 16)

            # Update hash vector based on bit positions
            for i in range(hash_size):
                bit = (token_hash >> i) & 1
                if bit:
                    hash_vector[i] += 1
                else:
                    hash_vector[i] -= 1

        # Convert vector to final hash by looking at signs
        final_hash = 0
        for i in range(hash_size):
            if hash_vector[i] >= 0:
                final_hash |= 1 << i

        return final_hash

    @staticmethod
    def hamming_distance(hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes"""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += xor & 1
            xor >>= 1
        return distance

    @staticmethod
    def hamming_similarity(hash1: int, hash2: int, hash_size: int = 64) -> float:
        """Compute similarity based on Hamming distance (0.0 to 1.0)"""
        distance = SimHasher.hamming_distance(hash1, hash2)
        return 1.0 - (distance / hash_size)


class MinHasher:
    """MinHash-based approximate deduplication (Jaccard similarity)"""

    def __init__(self, num_hashes: int = 128, ngram_size: int = 3):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.hash_functions = self._generate_hash_functions(num_hashes)

    @staticmethod
    def _generate_hash_functions(num_hashes: int):
        """Generate list of independent hash functions"""
        # Use prime numbers as seeds
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        return [
            (i * 2654435761 + p) % (2**32)
            for i, p in enumerate(primes * (num_hashes // len(primes) + 1))
        ][:num_hashes]

    def get_ngrams(self, text: str) -> Set[str]:
        """Extract n-grams from text"""
        tokens = text.lower().split()
        ngrams = set()

        for i in range(len(tokens) - self.ngram_size + 1):
            ngram = " ".join(tokens[i : i + self.ngram_size])
            ngrams.add(ngram)

        return ngrams

    def compute_minhash(self, text: str) -> List[int]:
        """Compute MinHash signature"""
        ngrams = self.get_ngrams(text)

        if not ngrams:
            return [2**32 - 1] * self.num_hashes

        minhash = [2**32 - 1] * self.num_hashes

        for i, hash_seed in enumerate(self.hash_functions):
            for ngram in ngrams:
                ngram_hash = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                hash_val = (ngram_hash + hash_seed) % (2**32)
                minhash[i] = min(minhash[i], hash_val)

        return minhash

    @staticmethod
    def jaccard_similarity(minhash1: List[int], minhash2: List[int]) -> float:
        """Estimate Jaccard similarity from MinHash signatures"""
        if len(minhash1) != len(minhash2):
            raise ValueError("MinHash signatures must have same length")

        matches = sum(1 for h1, h2 in zip(minhash1, minhash2) if h1 == h2)
        return matches / len(minhash1)


class NearDeduplicator:
    """
    Near-duplicate detection using multiple strategies.
    """

    def __init__(self, strategy: str = "simhash", threshold: float = 0.85):
        """
        Initialize near deduplicator.

        Args:
            strategy: "simhash" or "minhash"
            threshold: Similarity threshold [0.0, 1.0]
        """
        self.strategy = strategy
        self.threshold = threshold
        self.chunk_signatures: Dict[str, any] = {}
        self.near_duplicates: Set[Tuple[str, str]] = set()

        if strategy == "minhash":
            self.minhash_engine = MinHasher()

    def compute_signature(self, chunk_id: str, chunk_text: str) -> any:
        """Compute signature for near-dedup"""
        if self.strategy == "simhash":
            sig = SimHasher.compute_simhash(chunk_text)
        elif self.strategy == "minhash":
            sig = self.minhash_engine.compute_minhash(chunk_text)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.chunk_signatures[chunk_id] = sig
        return sig

    def find_near_duplicates(self) -> List[Tuple[str, str, float]]:
        """Find near-duplicate pairs exceeding threshold"""
        duplicates = []
        chunk_ids = list(self.chunk_signatures.keys())

        for i in range(len(chunk_ids)):
            for j in range(i + 1, len(chunk_ids)):
                chunk_id_1 = chunk_ids[i]
                chunk_id_2 = chunk_ids[j]

                if self.strategy == "simhash":
                    sig1 = self.chunk_signatures[chunk_id_1]
                    sig2 = self.chunk_signatures[chunk_id_2]
                    similarity = SimHasher.hamming_similarity(sig1, sig2)
                elif self.strategy == "minhash":
                    sig1 = self.chunk_signatures[chunk_id_1]
                    sig2 = self.chunk_signatures[chunk_id_2]
                    similarity = MinHasher.jaccard_similarity(sig1, sig2)

                if similarity >= self.threshold:
                    pair = tuple(sorted([chunk_id_1, chunk_id_2]))
                    if pair not in self.near_duplicates:
                        self.near_duplicates.add(pair)
                        duplicates.append((pair[0], pair[1], similarity))

        return duplicates


class DuplicateRemovalStrategy:
    """Strategy for removing duplicates while preserving important content"""

    def __init__(
        self,
        exact_dedup: ExactDeduplicator,
        near_dedup: Optional[NearDeduplicator] = None,
    ):
        self.exact_dedup = exact_dedup
        self.near_dedup = near_dedup
        self.removed_chunks: Set[str] = set()

    def merge_duplicates(self, chunk_scores: Dict[str, float]) -> Dict[str, bool]:
        """
        Decide which chunks to keep based on scores.
        Higher scores are kept.

        Returns:
            Dict mapping chunk_id -> should_keep
        """
        keep_decisions = {chunk_id: True for chunk_id in chunk_scores}

        # Handle exact duplicates
        exact_dups = self.exact_dedup.find_exact_duplicates()
        for chunk_id_1, chunk_id_2 in exact_dups:
            score1 = chunk_scores.get(chunk_id_1, 0.0)
            score2 = chunk_scores.get(chunk_id_2, 0.0)

            # Remove lower-scored duplicate
            if score1 >= score2:
                keep_decisions[chunk_id_2] = False
                self.removed_chunks.add(chunk_id_2)
            else:
                keep_decisions[chunk_id_1] = False
                self.removed_chunks.add(chunk_id_1)

        # Handle near duplicates
        if self.near_dedup:
            near_dups = self.near_dedup.find_near_duplicates()
            for chunk_id_1, chunk_id_2, similarity in near_dups:
                # Skip if already removed in exact dedup
                if (
                    chunk_id_1 in self.removed_chunks
                    or chunk_id_2 in self.removed_chunks
                ):
                    continue

                score1 = chunk_scores.get(chunk_id_1, 0.0)
                score2 = chunk_scores.get(chunk_id_2, 0.0)

                # Remove lower-scored near duplicate
                if score1 >= score2:
                    keep_decisions[chunk_id_2] = False
                    self.removed_chunks.add(chunk_id_2)
                else:
                    keep_decisions[chunk_id_1] = False
                    self.removed_chunks.add(chunk_id_1)

        return keep_decisions
