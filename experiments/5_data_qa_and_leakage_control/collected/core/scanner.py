"""Main scanner pipeline - orchestrates detection layers and produces reports."""

import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from .detectors import MinHashDetector, NGramDetector, SemanticDetector
from .registry import BenchmarkRegistry
from .utils import get_git_info, normalize

console = Console()


class ContaminationScanner:
    """Three-layer data contamination scanner.

    Runs candidate training data through three progressively more expensive
    detection layers:

    1. **N-gram** — exact word-sequence overlap (CRITICAL severity)
    2. **MinHash** — fuzzy Jaccard-based near-duplicate detection (HIGH)
    3. **Semantic** — dense-embedding cosine similarity (MEDIUM)

    Each layer only flags samples not already caught by a stricter layer,
    keeping the output clean and non-redundant.

    Example::

        scanner = ContaminationScanner()
        approved, report = scanner.scan_dataset("data.jsonl", "team-a", "batch-01")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the scanner and build all detection indexes.

        Config keys and their defaults:

        ==================  =======  ============================================
        Key                 Default  Description
        ==================  =======  ============================================
        benchmarks_path     "benchmarks"  Directory with ``*_test.jsonl`` files
        reports_path        "reports"     Directory where reports are written
        ngram_size          13       N-gram width for exact matching
        minhash_threshold   0.8      Jaccard threshold for fuzzy matching
        minhash_permutations 128     MinHash permutation count
        semantic_threshold  0.9      Cosine threshold for semantic matching
        semantic_model      "all-MiniLM-L6-v2"  Sentence-Transformers model
        semantic_batch_size 512      Encoding batch size
        report_sample_limit 50       Max flagged samples shown per layer in reports
        ==================  =======  ============================================

        Args:
            config: Optional configuration dictionary.  Missing keys fall back
                to the defaults listed above.
        """
        self.config: dict[str, Any] = config or {}
        self._sample_limit: int = self.config.get("report_sample_limit", 50)
        self._reports_path = Path(self.config.get("reports_path", "reports"))
        self._git_info = get_git_info()
        self._cache_enabled = bool(self.config.get("cache_indexes", True))
        self._cache_root = Path(self.config.get("cache_dir", ".cache/indexes"))
        self._enable_semantic = bool(self.config.get("enable_semantic", True))
        self._build_workers = max(
            1, int(self.config.get("build_workers", (os.cpu_count() or 1)))
        )

        # Load registry
        self.registry = BenchmarkRegistry(
            self.config.get("benchmarks_path", "benchmarks")
        ).load_all()
        self._index_fingerprint = self._compute_index_fingerprint()
        self._cache_dir = self._cache_root / self._index_fingerprint

        # Build detectors
        self.ngram = NGramDetector(
            n=self.config.get("ngram_size", 13),
            build_workers=self._build_workers,
        )
        self.minhash = MinHashDetector(
            threshold=self.config.get("minhash_threshold", 0.8),
            num_perm=self.config.get("minhash_permutations", 128),
            build_workers=self._build_workers,
        )
        self.has_semantic = False

        ngram_loaded = self._load_ngram_cache()
        if not ngram_loaded:
            self.ngram.build_index(self.registry)
            self._save_ngram_cache()

        minhash_loaded = self._load_minhash_cache()
        if not minhash_loaded:
            self.minhash.build_index(self.registry)
            self._save_minhash_cache()

        if self._enable_semantic:
            try:
                self.semantic = SemanticDetector(
                    threshold=self.config.get("semantic_threshold", 0.9),
                    model_name=self.config.get("semantic_model", "all-MiniLM-L6-v2"),
                    batch_size=self.config.get("semantic_batch_size", 512),
                )
                semantic_loaded = self._load_semantic_cache()
                if not semantic_loaded:
                    self.semantic.build_index(self.registry)
                    self._save_semantic_cache()
                self.has_semantic = True
            except ImportError as e:
                console.print(f"[yellow]⚠ Semantic detector disabled: {e}[/yellow]\n")
                self.has_semantic = False
        else:
            console.print("[yellow]⚠ Semantic detector disabled by config[/yellow]\n")
            self.has_semantic = False

        self._write_cache_manifest()
        console.print("[bold green]✓ Scanner ready![/bold green]\n")

    def scan_dataset(
        self, filepath: str | Path, team_name: str, batch_name: str
    ) -> tuple[bool, dict[str, Any]]:
        """Scan a JSONL file for benchmark contamination.

        Runs all enabled detection layers, aggregates per-sample findings,
        writes a JSON report and (if contamination is found) a JSONL file
        listing the contaminated samples to the ``reports/`` directory.

        Args:
            filepath: Path to the input ``.jsonl`` file.  Each line must be a
                JSON object with at least a ``text`` field.
            team_name: Identifier for the submitting team (stored in report).
            batch_name: Human-readable name for this data batch (used in the
                report filename).

        Returns:
            A ``(is_approved, report)`` tuple.  ``is_approved`` is ``True``
            when no contamination is detected.  ``report`` is the full
            structured findings dictionary.
        """
        run_id = str(uuid.uuid4())
        self._register_run(
            run_id,
            team_name,
            batch_name,
            status="STARTED",
            input_file=str(Path(filepath).resolve()),
            config=self.config,
        )

        return self._run_with_registry(
            run_id, filepath, team_name, batch_name, self._run_scan
        )

    def scan_records(
        self,
        records: list[dict[str, Any]],
        team_name: str,
        batch_name: str,
        input_label: str = "in-memory-records",
    ) -> tuple[bool, dict[str, Any]]:
        """Scan already-loaded JSONL-style records.

        Args:
            records: List of JSON-like objects, each expected to contain a
                ``text`` field.
            team_name: Identifier for the submitting team.
            batch_name: Human-readable batch name (used in report filenames).
            input_label: Provenance string recorded in the run registry
                (for example an S3 URI).
        """
        run_id = str(uuid.uuid4())
        self._register_run(
            run_id,
            team_name,
            batch_name,
            status="STARTED",
            input_file=input_label,
            config=self.config,
        )
        return self._run_with_registry(
            run_id, records, team_name, batch_name, self._run_scan_records
        )

    def _run_with_registry(
        self,
        run_id: str,
        source: Any,
        team_name: str,
        batch_name: str,
        runner: Any,
    ) -> tuple[bool, dict[str, Any]]:
        """Run a scan and ensure FAILED outcomes are recorded."""
        try:
            return runner(run_id, source, team_name, batch_name)
        except FileNotFoundError as exc:
            self._register_run(
                run_id,
                team_name,
                batch_name,
                status="FAILED",
                failure_type="INVALID_INPUT",
                error=str(exc),
            )
            raise
        except MemoryError as exc:
            self._register_run(
                run_id,
                team_name,
                batch_name,
                status="FAILED",
                failure_type="OUT_OF_MEMORY",
                error=str(exc),
            )
            raise
        except Exception as exc:
            self._register_run(
                run_id,
                team_name,
                batch_name,
                status="FAILED",
                failure_type="UNEXPECTED_ERROR",
                error=str(exc),
            )
            raise

    def _run_scan(
        self, run_id: str, filepath: str | Path, team_name: str, batch_name: str
    ) -> tuple[bool, dict[str, Any]]:
        """Internal scan implementation — called by :meth:`scan_dataset`."""
        data = self._load_jsonl(filepath)
        return self._run_scan_records(run_id, data, team_name, batch_name)

    def _run_scan_records(
        self,
        run_id: str,
        data: list[dict[str, Any]],
        team_name: str,
        batch_name: str,
    ) -> tuple[bool, dict[str, Any]]:
        """Internal scan implementation for in-memory records."""
        console.print(f"[bold cyan]Scanning: {batch_name}[/bold cyan]\n")
        texts = [normalize(item.get("text", str(item))) for item in data]
        ids = [item.get("id", f"sample_{i}") for i, item in enumerate(data)]

        console.print(f"Loaded {len(texts)} samples\n")

        # Run all detection layers
        ngram_matches = self.ngram.scan(texts)
        minhash_matches = self.minhash.scan(texts)
        semantic_matches = self.semantic.scan(texts) if self.has_semantic else {}

        # Aggregate per-sample findings (each sample recorded once, by strictest layer)
        contaminated_samples: dict[int, dict[str, Any]] = {}
        findings: list[dict[str, Any]] = []

        if ngram_matches:
            sample_details: list[dict[str, Any]] = []
            for benchmark, matches in ngram_matches.items():
                for match in matches:
                    idx = match["idx"]
                    if idx not in contaminated_samples:
                        contaminated_samples[idx] = {
                            "id": ids[idx],
                            "text": texts[idx],
                            "detection_method": "N-GRAM",
                            "matched_benchmarks": [],
                        }
                    contaminated_samples[idx]["matched_benchmarks"].append(
                        {
                            "benchmark": benchmark,
                            "match_type": "exact",
                            "confidence": "100%",
                        }
                    )
                    sample_details.append(
                        {
                            "sample_id": ids[idx],
                            "sample_index": idx,
                            "text_preview": texts[idx][:200]
                            + ("..." if len(texts[idx]) > 200 else ""),
                            "benchmark": benchmark,
                            "match_count": match["count"],
                        }
                    )

            findings.append(
                {
                    "layer": "N-GRAM",
                    "severity": "CRITICAL",
                    "count": len(
                        {
                            m["idx"]
                            for matches in ngram_matches.values()
                            for m in matches
                        }
                    ),
                    "benchmarks": list(ngram_matches.keys()),
                    "contaminated_samples": sample_details[: self._sample_limit],
                }
            )

        if minhash_matches:
            sample_details = []
            new_contaminated = 0

            for benchmark, matches in minhash_matches.items():
                for match in matches:
                    idx = match["idx"]

                    # Skip samples already caught by the stricter N-gram layer
                    if (
                        idx in contaminated_samples
                        and contaminated_samples[idx]["detection_method"] == "N-GRAM"
                    ):
                        continue

                    if idx not in contaminated_samples:
                        contaminated_samples[idx] = {
                            "id": ids[idx],
                            "text": texts[idx],
                            "detection_method": "MINHASH",
                            "matched_benchmarks": [],
                        }
                        new_contaminated += 1

                    contaminated_samples[idx]["matched_benchmarks"].append(
                        {
                            "benchmark": benchmark,
                            "match_type": "fuzzy",
                            "confidence": f"{match.get('jaccard', 0):.1%}",
                            "similar_to": match.get("match", "")[:100],
                        }
                    )
                    sample_details.append(
                        {
                            "sample_id": ids[idx],
                            "sample_index": idx,
                            "text_preview": texts[idx][:200]
                            + ("..." if len(texts[idx]) > 200 else ""),
                            "benchmark": benchmark,
                            "similar_to": match.get("match", "")[:100],
                        }
                    )

            if new_contaminated > 0:
                findings.append(
                    {
                        "layer": "MINHASH",
                        "severity": "HIGH",
                        "count": new_contaminated,
                        "benchmarks": list(minhash_matches.keys()),
                        "contaminated_samples": sample_details[: self._sample_limit],
                    }
                )

        if semantic_matches:
            sample_details = []
            new_contaminated = 0

            for benchmark, matches in semantic_matches.items():
                for match in matches:
                    idx = match["idx"]

                    # Skip samples already caught by a stricter layer
                    if idx in contaminated_samples:
                        continue

                    contaminated_samples[idx] = {
                        "id": ids[idx],
                        "text": texts[idx],
                        "detection_method": "SEMANTIC",
                        "matched_benchmarks": [],
                    }
                    new_contaminated += 1

                    contaminated_samples[idx]["matched_benchmarks"].append(
                        {
                            "benchmark": benchmark,
                            "match_type": "semantic",
                            "confidence": f"{match.get('cosine', 0):.1%}",
                            "similar_to": match.get("match", "")[:100],
                        }
                    )
                    sample_details.append(
                        {
                            "sample_id": ids[idx],
                            "sample_index": idx,
                            "text_preview": texts[idx][:200]
                            + ("..." if len(texts[idx]) > 200 else ""),
                            "benchmark": benchmark,
                            "similar_to": match.get("match", "")[:100],
                            "cosine": match.get("cosine", 0),
                        }
                    )

            if new_contaminated > 0:
                findings.append(
                    {
                        "layer": "SEMANTIC",
                        "severity": "MEDIUM",
                        "count": new_contaminated,
                        "benchmarks": list(semantic_matches.keys()),
                        "contaminated_samples": sample_details[: self._sample_limit],
                    }
                )

        # Build final report
        report: dict[str, Any] = {
            "run_id": run_id,
            "dataset": batch_name,
            "team": team_name,
            "timestamp": datetime.now().isoformat(),
            "scanner_commit": self._git_info["commit"],
            "repo_dirty": self._git_info["dirty"] == "true",
            "total_samples": len(texts),
            "contaminated_count": len(contaminated_samples),
            "contamination_rate": f"{len(contaminated_samples) / len(texts) * 100:.2f}%",
            "findings": findings,
            "all_contaminated_samples": list(contaminated_samples.values()),
            "status": "APPROVED" if not findings else "REJECTED",
        }

        self._display(report)
        self._save_report(report)
        self._save_contaminated_list(report, contaminated_samples)
        self._register_run(
            run_id,
            team_name,
            batch_name,
            status="COMPLETED",
            result=report["status"],
            contaminated_count=report["contaminated_count"],
            total_samples=report["total_samples"],
        )

        return report["status"] == "APPROVED", report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_run(
        self,
        run_id: str,
        team_name: str,
        batch_name: str,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Append a run record to the persistent run registry.

        The registry file is ``<reports_path>/run_registry.jsonl``.  Each line
        is one JSON record.  A run appears at least twice: once with
        ``status="STARTED"`` (written before detection begins) and once with
        ``status="COMPLETED"`` or ``status="FAILED"``.  This guarantees every
        run is permanently registered regardless of outcome.

        Args:
            run_id: UUID identifying this scan run.
            team_name: Submitting team identifier.
            batch_name: Data batch name.
            status: One of ``"STARTED"``, ``"COMPLETED"``, or ``"FAILED"``.
            **kwargs: Additional fields merged into the record (e.g.
                ``result``, ``contaminated_count``, ``total_samples``).
        """
        self._reports_path.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "team": team_name,
            "dataset": batch_name,
            "scanner_commit": self._git_info["commit"],
            "repo_dirty": self._git_info["dirty"] == "true",
            "status": status,
            **kwargs,
        }
        registry_path = self._reports_path / "run_registry.jsonl"
        with open(registry_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _load_jsonl(self, filepath: str | Path) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of dicts.

        Args:
            filepath: Path to the ``.jsonl`` file.

        Returns:
            List of parsed JSON objects (one per non-empty line).
        """
        data: list[dict[str, Any]] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    data.append(json.loads(stripped))
        return data

    def _display(self, report: dict[str, Any]) -> None:
        """Print a concise scan summary to the terminal.

        Args:
            report: The full report dictionary produced by :meth:`scan_dataset`.
        """
        console.print("\n" + "=" * 60)

        status = report["status"]
        color = "green" if status == "APPROVED" else "red"
        console.print(f"[bold {color}]{status}[/bold {color}]")
        console.print(
            f"Contamination: {report['contaminated_count']}/{report['total_samples']}"
            f" ({report['contamination_rate']})\n"
        )

        for finding in report["findings"]:
            console.print(
                f"[yellow]{finding['layer']}:[/yellow] {finding['count']} samples"
            )
            console.print(f"Benchmarks: {', '.join(finding['benchmarks'])}\n")

        console.print("=" * 60 + "\n")

    def _save_report(self, report: dict[str, Any]) -> None:
        """Persist the full JSON report to ``<reports_path>/<dataset>_<timestamp>.json``.

        Args:
            report: The full report dictionary produced by :meth:`scan_dataset`.
        """
        self._reports_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self._reports_path / f"{report['dataset']}_{timestamp}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        console.print(f"[green]✓ Report: {filename}[/green]\n")

    def _save_contaminated_list(
        self, report: dict[str, Any], contaminated_samples: dict[int, dict[str, Any]]
    ) -> None:
        """Save all contaminated samples to a separate JSONL file.

        The file is written to
        ``reports/<dataset>_CONTAMINATED_<timestamp>.jsonl`` only when at
        least one contaminated sample exists.

        Args:
            report: The full report dictionary (used for the filename).
            contaminated_samples: Mapping of sample index → details dict.
        """
        if not contaminated_samples:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            self._reports_path / f"{report['dataset']}_CONTAMINATED_{timestamp}.jsonl"
        )

        with open(filename, "w", encoding="utf-8") as f:
            for details in contaminated_samples.values():
                f.write(json.dumps(details) + "\n")

        console.print(f"[yellow]⚠ Contaminated samples: {filename}[/yellow]")

    def _compute_index_fingerprint(self) -> str:
        """Create a stable fingerprint for benchmarks + index-relevant config."""
        files = []
        for path in sorted(self.registry.benchmarks_path.glob("*_test.jsonl")):
            stat = path.stat()
            files.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )

        payload = {
            "schema": 1,
            "benchmarks_path": str(self.registry.benchmarks_path.resolve()),
            "benchmark_files": files,
            "ngram_size": self.config.get("ngram_size", 13),
            "minhash_threshold": self.config.get("minhash_threshold", 0.8),
            "minhash_permutations": self.config.get("minhash_permutations", 128),
            "semantic_threshold": self.config.get("semantic_threshold", 0.9),
            "semantic_model": self.config.get("semantic_model", "all-MiniLM-L6-v2"),
            "semantic_batch_size": self.config.get("semantic_batch_size", 512),
            "enable_semantic": self._enable_semantic,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _ensure_cache_dir(self) -> None:
        """Ensure the current cache directory exists."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _write_cache_manifest(self) -> None:
        """Write a small manifest to explain cache provenance."""
        if not self._cache_enabled:
            return
        self._ensure_cache_dir()
        manifest = {
            "fingerprint": self._index_fingerprint,
            "created_at": datetime.now().isoformat(),
            "benchmarks_path": str(self.registry.benchmarks_path.resolve()),
            "config": {
                "ngram_size": self.config.get("ngram_size", 13),
                "minhash_threshold": self.config.get("minhash_threshold", 0.8),
                "minhash_permutations": self.config.get("minhash_permutations", 128),
                "semantic_threshold": self.config.get("semantic_threshold", 0.9),
                "semantic_model": self.config.get("semantic_model", "all-MiniLM-L6-v2"),
                "semantic_batch_size": self.config.get("semantic_batch_size", 512),
                "enable_semantic": self._enable_semantic,
            },
        }
        manifest_path = self._cache_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _load_ngram_cache(self) -> bool:
        """Load cached n-gram index when available."""
        if not self._cache_enabled:
            return False
        path = self._cache_dir / "ngram.pkl"
        if not path.exists():
            return False
        try:
            self.ngram.load_index(path)
            console.print(f"[green]✓ Loaded n-gram index cache: {path}[/green]")
            return True
        except Exception as exc:
            console.print(f"[yellow]⚠ N-gram cache ignored: {exc}[/yellow]")
            return False

    def _save_ngram_cache(self) -> None:
        """Save n-gram index cache."""
        if not self._cache_enabled:
            return
        self._ensure_cache_dir()
        path = self._cache_dir / "ngram.pkl"
        self.ngram.save_index(path)
        console.print(f"[green]✓ Saved n-gram index cache: {path}[/green]")

    def _load_minhash_cache(self) -> bool:
        """Load cached MinHash index when available."""
        if not self._cache_enabled:
            return False
        path = self._cache_dir / "minhash.pkl"
        if not path.exists():
            return False
        try:
            self.minhash.load_index(path)
            console.print(f"[green]✓ Loaded MinHash index cache: {path}[/green]")
            return True
        except Exception as exc:
            console.print(f"[yellow]⚠ MinHash cache ignored: {exc}[/yellow]")
            return False

    def _save_minhash_cache(self) -> None:
        """Save MinHash index cache."""
        if not self._cache_enabled:
            return
        self._ensure_cache_dir()
        path = self._cache_dir / "minhash.pkl"
        self.minhash.save_index(path)
        console.print(f"[green]✓ Saved MinHash index cache: {path}[/green]")

    def _load_semantic_cache(self) -> bool:
        """Load cached semantic FAISS index when available."""
        if not self._cache_enabled:
            return False
        index_path = self._cache_dir / "semantic.faiss"
        meta_path = self._cache_dir / "semantic_meta.pkl"
        if not index_path.exists() or not meta_path.exists():
            return False
        try:
            self.semantic.load_index(index_path, meta_path)
            console.print(f"[green]✓ Loaded semantic index cache: {index_path}[/green]")
            return True
        except Exception as exc:
            console.print(f"[yellow]⚠ Semantic cache ignored: {exc}[/yellow]")
            return False

    def _save_semantic_cache(self) -> None:
        """Save semantic FAISS index cache."""
        if not self._cache_enabled:
            return
        self._ensure_cache_dir()
        index_path = self._cache_dir / "semantic.faiss"
        meta_path = self._cache_dir / "semantic_meta.pkl"
        self.semantic.save_index(index_path, meta_path)
        console.print(f"[green]✓ Saved semantic index cache: {index_path}[/green]")
