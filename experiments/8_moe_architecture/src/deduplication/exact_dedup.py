"""Exact deduplication using hashing."""

import xxhash
from typing import List, Set, Tuple


class ExactDeduplicator:
    """Performs exact deduplication on text chunks."""
    
    def __init__(self, seed: int = 42):
        """Initialize deduplicator with seed."""
        self.seed = seed
        self.seen_hashes: Set[str] = set()
        
    def hash_chunk(self, text: str) -> str:
        """Generate hash for a text chunk."""
        hasher = xxhash.xxh64(seed=self.seed)
        hasher.update(text.encode('utf-8'))
        return hasher.hexdigest()
    
    def deduplicate(self, chunks: List[str]) -> Tuple[List[int], List[int]]:
        """
        Deduplicate chunks.
        
        Returns:
            Tuple of (kept_indices, duplicate_indices)
        """
        kept = []
        duplicates = []
        
        for idx, chunk in enumerate(chunks):
            chunk_hash = self.hash_chunk(chunk)
            if chunk_hash not in self.seen_hashes:
                self.seen_hashes.add(chunk_hash)
                kept.append(idx)
            else:
                duplicates.append(idx)
                
        return kept, duplicates
