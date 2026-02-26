"""Detection layers: N-gram (exact), MinHash (fuzzy), and Semantic (paraphrase)."""

from collections import defaultdict
from typing import Any

from datasketch import MinHash, MinHashLSH
from rich.console import Console
from tqdm import tqdm

console = Console()


class NGramDetector:
    """Exact n-gram overlap detector.

    Builds a set of word n-grams from every benchmark sample and flags any
    training candidate that shares at least one n-gram.  Using 13-grams by
    default makes accidental collisions extremely unlikely while still
    catching verbatim or near-verbatim copies.
    """

    def __init__(self, n: int = 13) -> None:
        """Initialize the n-gram detector.

        Args:
            n: Size of each n-gram (number of consecutive words). Defaults to 13.
        """
        self.n = n
        self.index: dict[str, set[str]] = {}

    def build_index(self, registry: Any) -> None:
        """Build the n-gram index from all loaded benchmarks.

        Args:
            registry: A populated :class:`BenchmarkRegistry` instance.
        """
        console.print(f"[yellow]Building {self.n}-gram index...[/yellow]")

        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)
            ngrams: set[str] = set()

            for text in texts:
                ngrams.update(self._extract(text))

            self.index[name] = ngrams
            console.print(f"✓ {name}: {len(ngrams)} n-grams")

        console.print("[green]✓ N-gram index ready[/green]\n")

    def _extract(self, text: str) -> list[str]:
        """Extract all overlapping n-grams from *text*.

        Args:
            text: Normalized input text.

        Returns:
            List of n-gram strings.  Returns an empty list when the text
            contains fewer words than ``self.n``.
        """
        words = text.split()
        if len(words) < self.n:
            return []
        return [" ".join(words[i : i + self.n]) for i in range(len(words) - self.n + 1)]

    def scan(self, texts: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Scan a list of candidate texts for exact n-gram matches.

        Args:
            texts: Normalized candidate texts to check.

        Returns:
            Mapping of benchmark name → list of match records.  Each record
            contains ``idx`` (position in *texts*), ``text`` (preview), and
            ``count`` (number of matching n-grams).
        """
        matches: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for idx, text in enumerate(tqdm(texts, desc="N-gram")):
            text_ngrams = set(self._extract(text))

            for benchmark, bench_ngrams in self.index.items():
                overlap = text_ngrams & bench_ngrams
                if overlap:
                    matches[benchmark].append(
                        {"idx": idx, "text": text[:150], "count": len(overlap)}
                    )

        return dict(matches)


class MinHashDetector:
    """Fuzzy duplicate detector using MinHash LSH.

    Catches paraphrased or lightly modified benchmark samples that share
    enough word-bigram overlap to be suspicious.  The Jaccard similarity is
    re-verified after the LSH candidate lookup to discard false positives.
    """

    def __init__(self, threshold: float = 0.8, num_perm: int = 128) -> None:
        """Initialize the MinHash detector.

        Args:
            threshold: Minimum Jaccard similarity (0–1) to flag a match.
                Defaults to 0.8.
            num_perm: Number of MinHash permutations.  Higher values improve
                accuracy at the cost of memory and speed.  Defaults to 128.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        # Maps LSH key → {"text": str, "minhash": MinHash}
        self.keys: dict[str, dict[str, Any]] = {}

    def build_index(self, registry: Any) -> None:
        """Build the MinHash LSH index from all loaded benchmarks.

        Args:
            registry: A populated :class:`BenchmarkRegistry` instance.
        """
        console.print("[yellow]Building MinHash index...[/yellow]")

        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)

            for idx, text in enumerate(texts):
                mh = self._hash(text)
                key = f"{name}_{idx}"
                self.lsh.insert(key, mh)
                self.keys[key] = {"text": text[:100], "minhash": mh}

            console.print(f"✓ {name}: {len(texts)} hashes")

        console.print("[green]✓ MinHash index ready[/green]\n")

    def _hash(self, text: str) -> MinHash:
        """Compute a MinHash signature from the word-bigrams of *text*.

        Args:
            text: Normalized input text.

        Returns:
            A :class:`datasketch.MinHash` instance.
        """
        mh = MinHash(num_perm=self.num_perm)
        words = text.split()
        for i in range(len(words) - 1):
            shingle = " ".join(words[i : i + 2])
            mh.update(shingle.encode("utf-8"))
        return mh

    def scan(self, texts: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Scan a list of candidate texts for fuzzy MinHash matches.

        LSH candidates are re-verified with exact Jaccard computation and
        discarded when they fall below ``self.threshold``.

        Args:
            texts: Normalized candidate texts to check.

        Returns:
            Mapping of benchmark name → list of match records.  Each record
            contains ``idx``, ``text`` (preview), ``match`` (matched benchmark
            text preview), and ``jaccard`` (similarity score).
        """
        matches: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for idx, text in enumerate(tqdm(texts, desc="MinHash")):
            mh = self._hash(text)
            similar = self.lsh.query(mh)

            for key in similar:
                benchmark = key.rsplit("_", 1)[0]
                jaccard = mh.jaccard(self.keys[key]["minhash"])
                if jaccard < self.threshold:
                    continue  # LSH false positive — discard
                matches[benchmark].append(
                    {
                        "idx": idx,
                        "text": text[:150],
                        "match": self.keys[key]["text"],
                        "jaccard": round(jaccard, 3),
                    }
                )

        return dict(matches)


class SemanticDetector:
    """Semantic similarity detector using dense embeddings and FAISS.

    Encodes all benchmark samples into a FAISS flat inner-product index
    (cosine similarity on L2-normalized vectors) and queries it per batch
    of candidate texts.  Catches paraphrases and rewrites missed by exact
    or fuzzy methods.

    Requires ``sentence-transformers`` and ``faiss-cpu``.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
    ) -> None:
        """Initialize the semantic detector.

        Args:
            threshold: Minimum cosine similarity (0–1) to flag a match.
                Defaults to 0.9.
            model_name: Sentence-Transformers model identifier.
                Defaults to ``"all-MiniLM-L6-v2"``.
            batch_size: Number of texts to encode in one forward pass.
                Defaults to 512.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Semantic detection requires sentence-transformers. "
                "Install it with:  uv sync"
            )

        self.threshold = threshold
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        self.index = None  # populated by build_index()
        # Parallel list to the FAISS index: meta[i] = {"benchmark": str, "text": str}
        self.meta: list[dict[str, str]] = []

    def build_index(self, registry: Any) -> None:
        """Build the FAISS index from all loaded benchmarks.

        All benchmark texts are encoded in one pass and added to a flat
        inner-product index (equivalent to cosine similarity after L2
        normalization).

        Args:
            registry: A populated :class:`BenchmarkRegistry` instance.
        """
        import faiss

        dim = self.model.get_sentence_embedding_dimension()
        console.print(f"[yellow]Building semantic index ({dim}d)...[/yellow]")

        all_texts: list[str] = []
        all_meta: list[dict[str, str]] = []
        for name in registry.benchmarks.keys():
            for text in registry.get_texts(name):
                all_texts.append(text)
                all_meta.append({"benchmark": name, "text": text[:100]})

        embeddings = self.model.encode(
            all_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        # IndexFlatIP with normalized vectors computes cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.meta = all_meta

        console.print(
            f"[green]✓ Semantic index ready: {len(all_texts)} vectors[/green]\n"
        )

    def scan(self, texts: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Scan a list of candidate texts for semantic near-duplicates.

        Texts are encoded in batches and queried against the FAISS index.
        Only the single nearest neighbour is retrieved (k=1); if its cosine
        similarity meets ``self.threshold`` the sample is flagged.

        Args:
            texts: Normalized candidate texts to check.

        Returns:
            Mapping of benchmark name → list of match records.  Each record
            contains ``idx``, ``text`` (preview), ``match`` (matched benchmark
            text preview), and ``cosine`` (similarity score).
        """
        matches: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for batch_start in tqdm(range(0, len(texts), self.batch_size), desc="Semantic"):
            batch = texts[batch_start : batch_start + self.batch_size]

            embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ).astype("float32")

            scores, indices = self.index.search(embeddings, k=1)

            for i, (score, matched_idx) in enumerate(zip(scores[:, 0], indices[:, 0])):
                if float(score) >= self.threshold:
                    meta = self.meta[int(matched_idx)]
                    global_idx = batch_start + i
                    matches[meta["benchmark"]].append(
                        {
                            "idx": global_idx,
                            "text": texts[global_idx][:150],
                            "match": meta["text"],
                            "cosine": round(float(score), 3),
                        }
                    )

        return dict(matches)

