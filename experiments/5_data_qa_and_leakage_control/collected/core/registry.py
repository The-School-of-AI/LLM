"""Benchmark registry - loads and manages benchmark datasets."""

import json
from pathlib import Path

from rich.console import Console

from .utils import normalize

console = Console()


class BenchmarkRegistry:
    """Loads and indexes benchmark datasets from a local directory.

    Each benchmark is expected to be a ``*_test.jsonl`` file where each line
    is a JSON object containing at least a ``question`` field.

    Example:
        registry = BenchmarkRegistry("benchmarks").load_all()
        texts = registry.get_texts("mmlu_test")
    """

    def __init__(self, benchmarks_path: str | Path = "benchmarks") -> None:
        """Initialize the registry.

        Args:
            benchmarks_path: Path to the directory containing benchmark
                ``*_test.jsonl`` files. Defaults to ``"benchmarks"``.
        """
        self.benchmarks_path = Path(benchmarks_path)
        self.benchmarks: dict[str, list[dict]] = {}

    def load_all(self) -> "BenchmarkRegistry":
        """Discover and load all ``*_test.jsonl`` files in the benchmarks directory.

        Raises:
            FileNotFoundError: If ``benchmarks_path`` does not exist.

        Returns:
            Self, to allow method chaining (e.g. ``BenchmarkRegistry(...).load_all()``).
        """
        if not self.benchmarks_path.exists():
            raise FileNotFoundError(
                f"Benchmarks directory not found: {self.benchmarks_path.resolve()}\n"
                "Run  python scripts/download_benchmarks.py  to populate it first."
            )

        console.print("[yellow]Loading benchmarks...[/yellow]")

        for file in sorted(self.benchmarks_path.glob("*_test.jsonl")):
            name = file.stem
            data: list[dict] = []

            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        data.append(json.loads(stripped))

            self.benchmarks[name] = data
            console.print(f"✓ {name}: {len(data)} samples")

        console.print(f"[green]✓ Total: {len(self.benchmarks)} benchmarks[/green]\n")
        return self

    def get_texts(self, benchmark_name: str) -> list[str]:
        """Return normalized question strings for a given benchmark.

        Falls back to the full JSON representation for samples that lack a
        ``question`` field.

        Args:
            benchmark_name: Key matching a loaded benchmark (the file stem,
                e.g. ``"mmlu_test"``).

        Returns:
            List of normalized text strings. Returns an empty list if
            ``benchmark_name`` is not found.
        """
        texts: list[str] = []
        for sample in self.benchmarks.get(benchmark_name, []):
            text = sample.get("question", str(sample))
            texts.append(normalize(text))
        return texts
