"""
contamination.py — Benchmark Contamination Detection

Detects potential contamination between synthetic data and evaluation benchmarks.
Uses both exact matching (hash) and approximate matching (n-gram similarity).

Supported benchmarks:
- GSM8K (grade school math)
- MATH (competition math)
- MMLU (multitask language understanding)

Usage:
    from validation.contamination import ContaminationChecker

    checker = ContaminationChecker()
    checker.load_benchmarks(["gsm8k", "math", "mmlu"])
    results = checker.check_samples(synthetic_samples)
"""

"NEW CHANGES"
"""
WHAT THIS SCRIPT HAS:
- Text normalization
- Exact hash matching
- N-gram Jaccard similarity
- MinHash + LSH (NEW, for scalability)

WHY MinHash + LSH WAS ADDED:
- Avoid O(N×M) brute-force comparisons
- Keep $0 cost
- Improve scalability and recall safely
"""

'''contamination.py
│
├─ Text normalization                    (unchanged)
├─ Hashing (exact match)                 (unchanged)
├─ N-gram generation                     (unchanged)
│
├─ Disk cache (persistent)               (kept + extended)
│    ├─ hashes
│    ├─ ngrams
│    ├─ questions
│    └─ minhash signatures   ← NEW
│
├─ MinHash + LSH (in-memory index)        ← NEW
│    └─ candidate retrieval only
│
├─ Exact Jaccard similarity               (unchanged)
│    └─ applied ONLY to candidates
│
├─ Threshold decision                    (unchanged)
'''


import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datasketch import MinHash, MinHashLSH


# ==============================
# Data classes 
# ==============================

@dataclass
class ContaminationResult:
    sample_id: str
    is_contaminated: bool
    match_type: Optional[str] = None
    matched_benchmark: Optional[str] = None
    matched_question: Optional[str] = None
    similarity_score: float = 0.0


@dataclass
class ContaminationReport:
    total_samples: int
    contaminated_samples: int
    contamination_rate: float
    exact_matches: int
    high_similarity_matches: int
    partial_matches: int
    by_benchmark: dict = field(default_factory=dict)
    flagged_samples: list = field(default_factory=list)


# ==============================
# Text utilities 
# ==============================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\+\-\*\/\=\(\)\[\]\{\}]", "", text)
    return text.strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()[:16]


def ngram_set(text: str, n: int = 3) -> set:
    words = normalize_text(text).split()
    if len(words) < n:
        return {normalize_text(text)}
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


# ==============================
# Contamination Checker
# ==============================

class ContaminationChecker:

    STRONG_MATCH = 0.75
    PARTIAL_MATCH = 0.50

    MINHASH_PERMUTATIONS = 128
    LSH_THRESHOLD = 0.45

    def __init__(self, cache_dir="./.contamination_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_hashes = {}
        self.benchmark_ngrams = {}
        self.benchmark_questions = {}
        self.benchmark_minhashes = {}   # NEW (cached)

        self.lsh_indexes = {}            # NEW (runtime only)
        self.loaded_benchmarks = []

    # ==============================
    # Benchmark loading WITH CACHE
    # ==============================

    def load_benchmarks(self, benchmarks):
        from datasets import load_dataset

        for bench in benchmarks:
            cache_path = self.cache_dir / f"{bench}.json"

            if cache_path.exists():
                self._load_cache(bench, cache_path)
            else:
                questions = self._download_benchmark(bench, load_dataset)
                self._index_benchmark(bench, questions)
                self._save_cache(bench, cache_path)

            self._build_lsh(bench)

        self.loaded_benchmarks = benchmarks

    def _download_benchmark(self, bench, load_dataset):
        if bench == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            return [r["question"] for r in ds]
        elif bench == "math":
            ds = load_dataset("hendrycks/competition_math", split="test")
            return [r["problem"] for r in ds]
        elif bench == "mmlu":
            ds = load_dataset("cais/mmlu", "all", split="test")
            return [r["question"] for r in ds]
        return []

    # ==============================
    # Indexing + MinHash (CACHED)
    # ==============================

    def _index_benchmark(self, bench, questions):
        self.benchmark_hashes[bench] = set()
        self.benchmark_ngrams[bench] = {}
        self.benchmark_questions[bench] = {}
        self.benchmark_minhashes[bench] = {}

        for q in questions:
            h = text_hash(q)
            ng = ngram_set(q)

            mh = MinHash(num_perm=self.MINHASH_PERMUTATIONS)
            for g in ng:
                mh.update(g.encode())

            self.benchmark_hashes[bench].add(h)
            self.benchmark_ngrams[bench][h] = ng
            self.benchmark_questions[bench][h] = q[:200]
            self.benchmark_minhashes[bench][h] = mh

    # ==============================
    # Disk cache (KEPT)
    # ==============================

    def _save_cache(self, bench, path):
        with open(path, "w") as f:
            json.dump({
                "hashes": list(self.benchmark_hashes[bench]),
                "ngrams": {k: list(v) for k, v in self.benchmark_ngrams[bench].items()},
                "questions": self.benchmark_questions[bench],
                "minhash": {
                    k: v.hashvalues.tolist()
                    for k, v in self.benchmark_minhashes[bench].items()
                }
            }, f)

    def _load_cache(self, bench, path):
        with open(path) as f:
            data = json.load(f)

        self.benchmark_hashes[bench] = set(data["hashes"])
        self.benchmark_ngrams[bench] = {k: set(v) for k, v in data["ngrams"].items()}
        self.benchmark_questions[bench] = data["questions"]

        self.benchmark_minhashes[bench] = {}
        for k, hv in data["minhash"].items():
            mh = MinHash(num_perm=self.MINHASH_PERMUTATIONS)
            mh.hashvalues = hv
            self.benchmark_minhashes[bench][k] = mh

    # ==============================
    # LSH built at runtime
    # ==============================

    def _build_lsh(self, bench):
        lsh = MinHashLSH(
            threshold=self.LSH_THRESHOLD,
            num_perm=self.MINHASH_PERMUTATIONS
        )
        for k, mh in self.benchmark_minhashes[bench].items():
            lsh.insert(k, mh)
        self.lsh_indexes[bench] = lsh

    # ==============================
    # Sample checking
    # ==============================

    def check_sample(self, question, sample_id=""):
        q_hash = text_hash(question)
        q_ngrams = ngram_set(question)

        # Exact match 
        for bench in self.loaded_benchmarks:
            if q_hash in self.benchmark_hashes[bench]:
                return ContaminationResult(sample_id, True, "exact", bench, None, 1.0)

        # MinHash candidate retrieval (NEW)
        q_mh = MinHash(num_perm=self.MINHASH_PERMUTATIONS)
        for g in q_ngrams:
            q_mh.update(g.encode())

        best = ContaminationResult(sample_id, False)

        for bench in self.loaded_benchmarks:
            for h in self.lsh_indexes[bench].query(q_mh):
                sim = jaccard_similarity(q_ngrams, self.benchmark_ngrams[bench][h])
                if sim > best.similarity_score:
                    best = ContaminationResult(
                        sample_id,
                        sim >= self.PARTIAL_MATCH,
                        "high" if sim >= self.STRONG_MATCH else "partial",
                        bench,
                        self.benchmark_questions[bench][h],
                        sim
                    )

        return best
