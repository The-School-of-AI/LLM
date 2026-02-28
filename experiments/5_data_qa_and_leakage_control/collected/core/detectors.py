"""Detection layers: N-gram (exact), MinHash (fuzzy), and Semantic (paraphrase)."""

import pickle
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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

    def __init__(self, n: int = 13, build_workers: int = 1) -> None:
        """Initialize the n-gram detector.

        Args:
            n: Size of each n-gram (number of consecutive words). Defaults to 13.
            build_workers: Number of worker threads used to build the index.
                Set to 1 to keep fully serial behavior.
        """
        self.n = n
        self.build_workers = max(1, int(build_workers))
        self.index: dict[str, set[str]] = {}

    def build_index(self, registry: Any) -> None:
        """Build the n-gram index from all loaded benchmarks.

        Args:
            registry: A populated :class:`BenchmarkRegistry` instance.
        """
        console.print(f"[yellow]Building {self.n}-gram index...[/yellow]")

        for name in registry.benchmarks.keys():
            texts = registry.get_texts(name)
            ngrams = self._extract_ngrams_for_texts(texts)

            self.index[name] = ngrams
            console.print(f"✓ {name}: {len(ngrams)} n-grams")

        console.print("[green]✓ N-gram index ready[/green]\n")

    def save_index(self, filepath: str | Path) -> None:
        """Persist the built n-gram index to disk."""
        payload = {"n": self.n, "index": self.index}
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, filepath: str | Path) -> None:
        """Load a previously persisted n-gram index."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        loaded_n = payload.get("n")
        if loaded_n != self.n:
            raise ValueError(
                f"N-gram cache mismatch: expected n={self.n}, found n={loaded_n}"
            )
        self.index = payload["index"]

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

    def _extract_ngrams_for_texts(self, texts: list[str]) -> set[str]:
        """Build an n-gram set for all texts, optionally in parallel."""
        if self.build_workers <= 1 or len(texts) < 2000:
            ngrams: set[str] = set()
            for text in texts:
                ngrams.update(self._extract(text))
            return ngrams

        ngrams: set[str] = set()
        chunk_size = max(1, len(texts) // self.build_workers)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        with ThreadPoolExecutor(max_workers=self.build_workers) as executor:
            for chunk_ngrams in executor.map(self._extract_ngrams_chunk, chunks):
                ngrams.update(chunk_ngrams)
        return ngrams

    def _extract_ngrams_chunk(self, chunk: list[str]) -> set[str]:
        """Extract n-grams for one chunk of texts."""
        out: set[str] = set()
        for text in chunk:
            out.update(self._extract(text))
        return out

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

    def __init__(
        self, threshold: float = 0.8, num_perm: int = 128, build_workers: int = 1
    ) -> None:
        """Initialize the MinHash detector.

        Args:
            threshold: Minimum Jaccard similarity (0–1) to flag a match.
                Defaults to 0.8.
            num_perm: Number of MinHash permutations.  Higher values improve
                accuracy at the cost of memory and speed.  Defaults to 128.
            build_workers: Number of worker threads used to hash benchmark
                texts during index build. Set to 1 for serial behavior.
        """
        self.threshold = threshold
        self.num_perm = num_perm
        self.build_workers = max(1, int(build_workers))
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

            if self.build_workers <= 1 or len(texts) < 1000:
                for idx, text in enumerate(texts):
                    mh = self._hash(text)
                    key = f"{name}_{idx}"
                    self.lsh.insert(key, mh)
                    self.keys[key] = {"text": text[:100], "minhash": mh}
            else:
                chunk_size = max(1, len(texts) // self.build_workers)
                chunks = [
                    list(enumerate(texts[i : i + chunk_size], start=i))
                    for i in range(0, len(texts), chunk_size)
                ]
                with ThreadPoolExecutor(max_workers=self.build_workers) as executor:
                    for rows in executor.map(self._hash_chunk, chunks):
                        for idx, text_preview, mh in rows:
                            key = f"{name}_{idx}"
                            self.lsh.insert(key, mh)
                            self.keys[key] = {"text": text_preview, "minhash": mh}

            console.print(f"✓ {name}: {len(texts)} hashes")

        console.print("[green]✓ MinHash index ready[/green]\n")

    def save_index(self, filepath: str | Path) -> None:
        """Persist the built MinHash structures to disk."""
        payload = {
            "threshold": self.threshold,
            "num_perm": self.num_perm,
            "lsh": self.lsh,
            "keys": self.keys,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, filepath: str | Path) -> None:
        """Load previously persisted MinHash structures."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        cached_threshold = payload.get("threshold")
        cached_num_perm = payload.get("num_perm")
        if cached_threshold != self.threshold or cached_num_perm != self.num_perm:
            raise ValueError(
                "MinHash cache mismatch: expected "
                f"threshold={self.threshold}, num_perm={self.num_perm}; found "
                f"threshold={cached_threshold}, num_perm={cached_num_perm}"
            )

        self.lsh = payload["lsh"]
        self.keys = payload["keys"]

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

    def _hash_chunk(
        self, chunk: list[tuple[int, str]]
    ) -> list[tuple[int, str, MinHash]]:
        """Hash one chunk of benchmark texts."""
        out: list[tuple[int, str, MinHash]] = []
        for idx, text in chunk:
            out.append((idx, text[:100], self._hash(text)))
        return out

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

    def save_index(self, index_path: str | Path, meta_path: str | Path) -> None:
        """Persist FAISS index + metadata to disk."""
        import faiss

        if self.index is None:
            raise ValueError("Semantic index is empty. Build it before saving.")

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_index(self, index_path: str | Path, meta_path: str | Path) -> None:
        """Load FAISS index + metadata from disk."""
        import faiss

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            self.meta = pickle.load(f)

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
