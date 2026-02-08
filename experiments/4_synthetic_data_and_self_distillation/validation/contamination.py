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

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ContaminationResult:
    """Result of contamination check for a single sample."""

    sample_id: str
    is_contaminated: bool
    match_type: Optional[str] = None  # "exact", "high_similarity", "partial"
    matched_benchmark: Optional[str] = None
    matched_question: Optional[str] = None
    similarity_score: float = 0.0


@dataclass
class ContaminationReport:
    """Aggregated contamination report."""

    total_samples: int
    contaminated_samples: int
    contamination_rate: float
    exact_matches: int
    high_similarity_matches: int
    partial_matches: int
    by_benchmark: dict = field(default_factory=dict)
    flagged_samples: list = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove punctuation except for math symbols
    text = re.sub(r"[^\w\s\+\-\*\/\=\(\)\[\]\{\}]", "", text)
    return text.strip()


def text_hash(text: str) -> str:
    """Create hash of normalized text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def ngram_set(text: str, n: int = 3) -> set:
    """Create set of n-grams from text."""
    normalized = normalize_text(text)
    words = normalized.split()
    if len(words) < n:
        return {normalized}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


class ContaminationChecker:
    """Check synthetic data against benchmark datasets."""

    # Similarity thresholds
    EXACT_MATCH_THRESHOLD = 1.0
    HIGH_SIMILARITY_THRESHOLD = 0.7
    PARTIAL_MATCH_THRESHOLD = 0.4

    def __init__(self, cache_dir: str = "./.contamination_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.benchmark_hashes: dict[str, set] = {}  # benchmark_name -> set of hashes
        self.benchmark_ngrams: dict[str, dict] = (
            {}
        )  # benchmark_name -> {hash: ngram_set}
        self.benchmark_questions: dict[str, dict] = (
            {}
        )  # benchmark_name -> {hash: question}

        self.loaded_benchmarks: list[str] = []

    def load_benchmarks(self, benchmarks: list[str] = None, force_reload: bool = False):
        """Load benchmark datasets for comparison.

        Args:
            benchmarks: List of benchmark names. Default: ["gsm8k", "math", "mmlu"]
            force_reload: Force re-download even if cached
        """
        if benchmarks is None:
            benchmarks = ["gsm8k", "math", "mmlu"]

        try:
            from datasets import load_dataset  # noqa: F401
        except ImportError:
            print(
                "[WARN] 'datasets' library not available. Install with: pip install datasets"
            )
            print("[WARN] Falling back to local cache only.")
            return self._load_from_cache(benchmarks)

        for benchmark in benchmarks:
            cache_path = self.cache_dir / f"{benchmark}_hashes.json"

            if cache_path.exists() and not force_reload:
                print(f"[Load] {benchmark} from cache")
                self._load_benchmark_cache(benchmark, cache_path)
                continue

            print(f"[Download] {benchmark} from HuggingFace...")
            try:
                questions = self._fetch_benchmark(benchmark)
                self._index_questions(benchmark, questions)
                self._save_benchmark_cache(benchmark, cache_path)
                print(f"  Indexed {len(questions)} questions")
            except Exception as e:
                print(f"  [ERROR] Failed to load {benchmark}: {e}")

        self.loaded_benchmarks = list(self.benchmark_hashes.keys())
        print(
            f"[Done] Loaded {len(self.loaded_benchmarks)} benchmarks, "
            f"{sum(len(h) for h in self.benchmark_hashes.values())} total questions"
        )

    def _fetch_benchmark(self, benchmark: str) -> list[str]:
        """Fetch questions from a benchmark dataset."""
        from datasets import load_dataset

        questions = []

        if benchmark == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test")
            questions = [row["question"] for row in ds]

        elif benchmark == "math":
            # Try different MATH dataset variants
            try:
                ds = load_dataset("hendrycks/competition_math", split="test")
                questions = [row["problem"] for row in ds]
            except Exception:
                try:
                    ds = load_dataset("lighteval/MATH", split="test")
                    questions = [row["problem"] for row in ds]
                except Exception as e:
                    print(f"  [WARN] Could not load MATH dataset: {e}")

        elif benchmark == "mmlu":
            ds = load_dataset("cais/mmlu", "all", split="test")
            questions = [row["question"] for row in ds]

        else:
            print(f"  [WARN] Unknown benchmark: {benchmark}")

        return questions

    def _index_questions(self, benchmark: str, questions: list[str]):
        """Index questions for fast lookup."""
        self.benchmark_hashes[benchmark] = set()
        self.benchmark_ngrams[benchmark] = {}
        self.benchmark_questions[benchmark] = {}

        for q in questions:
            h = text_hash(q)
            self.benchmark_hashes[benchmark].add(h)
            self.benchmark_ngrams[benchmark][h] = ngram_set(q)
            self.benchmark_questions[benchmark][h] = q[
                :200
            ]  # Store truncated for reference

    def _save_benchmark_cache(self, benchmark: str, cache_path: Path):
        """Save benchmark index to cache."""
        data = {
            "hashes": list(self.benchmark_hashes.get(benchmark, set())),
            "ngrams": {
                h: list(ng)
                for h, ng in self.benchmark_ngrams.get(benchmark, {}).items()
            },
            "questions": self.benchmark_questions.get(benchmark, {}),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _load_benchmark_cache(self, benchmark: str, cache_path: Path):
        """Load benchmark index from cache."""
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

        self.benchmark_hashes[benchmark] = set(data.get("hashes", []))
        self.benchmark_ngrams[benchmark] = {
            h: set(ng) for h, ng in data.get("ngrams", {}).items()
        }
        self.benchmark_questions[benchmark] = data.get("questions", {})

    def _load_from_cache(self, benchmarks: list[str]):
        """Load only from local cache."""
        for benchmark in benchmarks:
            cache_path = self.cache_dir / f"{benchmark}_hashes.json"
            if cache_path.exists():
                self._load_benchmark_cache(benchmark, cache_path)
                print(f"[Load] {benchmark} from cache")
        self.loaded_benchmarks = list(self.benchmark_hashes.keys())

    def check_sample(self, question: str, sample_id: str = "") -> ContaminationResult:
        """Check a single sample for contamination."""
        q_hash = text_hash(question)
        q_ngrams = ngram_set(question)

        best_match = ContaminationResult(
            sample_id=sample_id,
            is_contaminated=False,
            similarity_score=0.0,
        )

        for benchmark in self.loaded_benchmarks:
            # Exact match check
            if q_hash in self.benchmark_hashes[benchmark]:
                return ContaminationResult(
                    sample_id=sample_id,
                    is_contaminated=True,
                    match_type="exact",
                    matched_benchmark=benchmark,
                    matched_question=self.benchmark_questions[benchmark].get(
                        q_hash, ""
                    ),
                    similarity_score=1.0,
                )

            # Similarity check against all benchmark questions
            for b_hash, b_ngrams in self.benchmark_ngrams[benchmark].items():
                sim = jaccard_similarity(q_ngrams, b_ngrams)

                if sim > best_match.similarity_score:
                    best_match = ContaminationResult(
                        sample_id=sample_id,
                        is_contaminated=sim >= self.PARTIAL_MATCH_THRESHOLD,
                        match_type=(
                            "high_similarity"
                            if sim >= self.HIGH_SIMILARITY_THRESHOLD
                            else "partial"
                        ),
                        matched_benchmark=benchmark,
                        matched_question=self.benchmark_questions[benchmark].get(
                            b_hash, ""
                        ),
                        similarity_score=sim,
                    )

        return best_match

    def check_samples(
        self, samples: list[dict], question_field: str = "question"
    ) -> ContaminationReport:
        """Check multiple samples for contamination.

        Args:
            samples: List of sample dicts
            question_field: Field name containing the question

        Returns:
            ContaminationReport with aggregated results
        """
        results = []
        by_benchmark = {}

        for i, sample in enumerate(samples):
            question = sample.get(question_field) or sample.get("instruction", "")
            sample_id = sample.get("id", f"sample_{i}")

            result = self.check_sample(question, sample_id)
            results.append(result)

            if result.is_contaminated:
                bench = result.matched_benchmark or "unknown"
                by_benchmark[bench] = by_benchmark.get(bench, 0) + 1

        contaminated = [r for r in results if r.is_contaminated]
        exact = [r for r in results if r.match_type == "exact"]
        high_sim = [r for r in results if r.match_type == "high_similarity"]
        partial = [r for r in results if r.match_type == "partial"]

        return ContaminationReport(
            total_samples=len(samples),
            contaminated_samples=len(contaminated),
            contamination_rate=len(contaminated) / len(samples) if samples else 0.0,
            exact_matches=len(exact),
            high_similarity_matches=len(high_sim),
            partial_matches=len(partial),
            by_benchmark=by_benchmark,
            flagged_samples=[
                {
                    "id": r.sample_id,
                    "match_type": r.match_type,
                    "benchmark": r.matched_benchmark,
                    "similarity": r.similarity_score,
                    "matched_question": r.matched_question,
                }
                for r in contaminated
            ],
        )

    def filter_contaminated(
        self, samples: list[dict], question_field: str = "question"
    ) -> tuple[list[dict], ContaminationReport]:
        """Filter out contaminated samples.

        Returns:
            (clean_samples, contamination_report)
        """
        report = self.check_samples(samples, question_field)
        contaminated_ids = {s["id"] for s in report.flagged_samples}

        clean_samples = [
            s
            for s in samples
            if s.get("id", f"sample_{samples.index(s)}") not in contaminated_ids
        ]

        return clean_samples, report


def run_contamination_check(
    data_path: str,
    benchmarks: list[str] = None,
    output_path: str = None,
    filter_contaminated: bool = False,
) -> ContaminationReport:
    """Run contamination check on a data file.

    Args:
        data_path: Path to JSONL file with samples
        benchmarks: Benchmarks to check against
        output_path: Optional path to save clean data
        filter_contaminated: If True, save filtered data to output_path
    """
    # Load samples
    with open(data_path, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"\n[Contamination Check] {len(samples)} samples from {data_path}")

    # Check
    checker = ContaminationChecker()
    checker.load_benchmarks(benchmarks)

    if filter_contaminated and output_path:
        clean_samples, report = checker.filter_contaminated(samples)
        with open(output_path, "w", encoding="utf-8") as f:
            for s in clean_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[Saved] {len(clean_samples)} clean samples to {output_path}")
    else:
        report = checker.check_samples(samples)

    # Print report
    print("\n[Results]")
    print(f"  Total samples: {report.total_samples}")
    print(
        f"  Contaminated: {report.contaminated_samples} ({report.contamination_rate*100:.1f}%)"
    )
    print(f"    Exact matches: {report.exact_matches}")
    print(f"    High similarity: {report.high_similarity_matches}")
    print(f"    Partial matches: {report.partial_matches}")

    if report.by_benchmark:
        print("\n  By Benchmark:")
        for bench, count in report.by_benchmark.items():
            print(f"    {bench}: {count}")

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        run_contamination_check(sys.argv[1])
    else:
        print("Usage: python contamination.py <data.jsonl>")
