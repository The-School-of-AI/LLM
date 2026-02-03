import hashlib
from typing import List, Set

class MinHash:
    """
    A pure python MinHash implementation to avoid 'datasketch' dependency if missing.
    """
    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        self.max_hash = (1 << 32) - 1
        # Create deterministic seeds
        self.perms = [ (i * 3 + 1, i * 7 + 11) for i in range(num_perm) ]

    def compute(self, text: str) -> List[int]:
        """
        Compute MinHash signature for text.
        Structure: Split by whitespace -> 3-grams -> hash -> min.
        """
        words = text.split()
        if len(words) < 3:
            return [] # Too short
            
        # Generate 3-shingles
        shingles = set()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i+3])
            # Hash to 32-bit int
            h = int(hashlib.md5(shingle.encode('utf-8')).hexdigest()[:8], 16)
            shingles.add(h)
            
        if not shingles:
            return []

        # MinHash calculation
        signature = []
        for i in range(self.num_perm):
            a, b = self.perms[i]
            min_h = self.max_hash
            for h in shingles:
                # Permutation: (a*h + b) % max_hash (Pseudo-random permutation)
                ph = (a * h + b) % 4294967311 # Large prime
                if ph < min_h:
                    min_h = ph
            signature.append(min_h)
            
        return signature

class DedupRegister:
    """
    Maintains a registry of seen signatures (MinHash) and exact hashes (MD5).
    """
    def __init__(self, threshold: float = 0.8):
        self.seen_exact: Set[str] = set()
        # For MinHash query, usually LSH is needed. 
        # Without LSH (complex to implement pure python efficiently), 
        # we might rely on Exact Dedup primarily for this scale,
        # Or just store signatures and do a loose check if strictly needed.
        # Given constraints (200M tokens, local Mac), we might simplify to EXACT dedup only
        # for high speed, or Very Simple LSH.
        self.threshold = threshold

    def is_duplicate(self, text: str, signature: List[int] = None) -> bool:
        """
        Checks if text is a duplicate.
        Currently implements EXACT duplication check efficiently.
        """
        # Exact Hash
        md5_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if md5_hash in self.seen_exact:
            return True
        self.seen_exact.add(md5_hash)
        
        # Note: LSH would go here. For now we skip LSH to keep it fast and pure python.
        return False
