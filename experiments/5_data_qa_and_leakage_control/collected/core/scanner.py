"""Main scanner pipeline"""

import json
from datetime import datetime
from pathlib import Path

from rich.console import Console

from .detectors import MinHashDetector, NGramDetector, SemanticDetector
from .registry import BenchmarkRegistry

console = Console()


class ContaminationScanner:
    def __init__(self, config=None):
        self.config = config or {}

        # Load registry
        self.registry = BenchmarkRegistry(
            self.config.get("benchmarks_path", "benchmarks")
        ).load_all()

        # Build detectors
        self.ngram = NGramDetector(n=self.config.get("ngram_size", 13))
        self.minhash = MinHashDetector(
            threshold=self.config.get("minhash_threshold", 0.8),
            num_perm=self.config.get("minhash_permutations", 128),
        )

        self.ngram.build_index(self.registry)
        self.minhash.build_index(self.registry)

        try:
            self.semantic = SemanticDetector(
                threshold=self.config.get("semantic_threshold", 0.9),
                model_name=self.config.get("semantic_model", "all-MiniLM-L6-v2"),
                batch_size=self.config.get("semantic_batch_size", 512),
            )
            self.semantic.build_index(self.registry)
            self.has_semantic = True
        except ImportError as e:
            console.print(f"[yellow]⚠ Semantic detector disabled: {e}[/yellow]\n")
            self.has_semantic = False

        console.print("[bold green]✓ Scanner ready![/bold green]\n")

    def scan_dataset(self, filepath, team_name, batch_name):
        console.print(f"[bold cyan]Scanning: {batch_name}[/bold cyan]\n")

        # Load data
        data = self._load_jsonl(filepath)
        texts = [self._normalize(item.get("text", str(item))) for item in data]
        ids = [item.get("id", f"sample_{i}") for i, item in enumerate(data)]

        console.print(f"Loaded {len(texts)} samples\n")

        # Run detectors
        ngram_matches = self.ngram.scan(texts)
        minhash_matches = self.minhash.scan(texts)
        semantic_matches = self.semantic.scan(texts) if self.has_semantic else {}

        # Aggregate with detailed sample info
        contaminated_samples = {}  # idx -> details
        findings = []

        if ngram_matches:
            sample_details = []
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
                        set(
                            m["idx"]
                            for matches in ngram_matches.values()
                            for m in matches
                        )
                    ),
                    "benchmarks": list(ngram_matches.keys()),
                    "contaminated_samples": sample_details[
                        :50
                    ],  # Limit to first 50 for readability
                }
            )

        if minhash_matches:
            sample_details = []
            new_contaminated = 0

            for benchmark, matches in minhash_matches.items():
                for match in matches:
                    idx = match["idx"]

                    # Skip if already caught by n-gram
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
                            "confidence": f"{match.get('jaccard', 0):.0%}",
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
                        "contaminated_samples": sample_details[:50],
                    }
                )

        if semantic_matches:
            sample_details = []
            new_contaminated = 0

            for benchmark, matches in semantic_matches.items():
                for match in matches:
                    idx = match["idx"]

                    # Skip if already caught by a stricter layer
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
                            "confidence": f"{match.get('cosine', 0):.0%}",
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
                        "contaminated_samples": sample_details[:50],
                    }
                )

        # Build report
        report = {
            "dataset": batch_name,
            "team": team_name,
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(texts),
            "contaminated_count": len(contaminated_samples),
            "contamination_rate": f"{len(contaminated_samples)/len(texts)*100:.2f}%",
            "findings": findings,
            "all_contaminated_samples": list(contaminated_samples.values()),
            "status": "APPROVED" if not findings else "REJECTED",
        }

        # Display and save
        self._display(report)
        self._save_report(report)
        self._save_contaminated_list(report, contaminated_samples)

        return report["status"] == "APPROVED", report

    def _load_jsonl(self, filepath):
        data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _normalize(self, text):
        return " ".join(str(text).lower().strip().split())

    def _display(self, report):
        console.print("\n" + "=" * 60)

        status = report["status"]
        color = "green" if status == "APPROVED" else "red"

        console.print(f"[bold {color}]{status}[/bold {color}]")
        console.print(
            f"Contamination: {report['contaminated_count']}/{report['total_samples']} ({report['contamination_rate']})\n"
        )

        for finding in report["findings"]:
            console.print(
                f"[yellow]{finding['layer']}:[/yellow] {finding['count']} samples"
            )
            console.print(f"Benchmarks: {', '.join(finding['benchmarks'])}\n")

        console.print("=" * 60 + "\n")

    def _save_report(self, report):
        Path("reports").mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/{report['dataset']}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"[green]✓ Report: {filename}[/green]\n")

    def _save_contaminated_list(self, report, contaminated_samples):
        """Save a separate file listing all contaminated samples"""
        if not contaminated_samples:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/{report['dataset']}_CONTAMINATED_{timestamp}.jsonl"

        with open(filename, "w") as f:
            for idx, details in contaminated_samples.items():
                f.write(json.dumps(details) + "\n")

        console.print(f"[yellow]⚠ Contaminated samples: {filename}[/yellow]")
