"""Near-deduplication using MinHash and SimHash."""

from datasketch import MinHash, MinHashLSH
from typing import List, Tuple, Set


class NearDeduplicator:
    """Performs near-deduplication using MinHash LSH."""
    
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, seed: int = 42):
        """
        Initialize near-deduplicator.
        
        Args:
            threshold: Jaccard similarity threshold
            num_perm: Number of permutations for MinHash
            seed: Random seed
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.seed = seed
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        
    def create_minhash(self, text: str) -> MinHash:
        """Create MinHash signature for text."""
        m = MinHash(num_perm=self.num_perm, seed=self.seed)
        tokens = text.lower().split()
        for token in tokens:
            m.update(token.encode('utf-8'))
        return m
    
    def deduplicate(self, chunks: List[str]) -> Tuple[List[int], List[int]]:
        """
        Deduplicate chunks using LSH.
        
        Returns:
            Tuple of (kept_indices, near_duplicate_indices)
        """
        kept = []
        duplicates = []
        
        for idx, chunk in enumerate(chunks):
            minhash = self.create_minhash(chunk)
            
            # Check if similar document exists
            results = self.lsh.query(minhash)
            
            if not results:
                # No similar documents, keep this one
                self.lsh.insert(f"chunk_{idx}", minhash)
                kept.append(idx)
            else:
                duplicates.append(idx)
                
        return kept, duplicates
