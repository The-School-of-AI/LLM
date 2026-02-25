"""Smoke test for the contamination scanner.

Creates minimal temporary benchmark and input files, runs the scanner
with N-gram + MinHash only (no semantic layer required), and asserts
that the report structure and contamination decisions are correct.

Run from the collected/ directory:
    python tests/smoke_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scanner import ContaminationScanner


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_clean_dataset_is_approved() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        benchmarks_dir = tmp / "benchmarks"
        benchmarks_dir.mkdir()
        reports_dir = tmp / "reports"

        # Benchmark: one question
        write_jsonl(
            benchmarks_dir / "dummy_test.jsonl",
            [{"question": "What is the capital of France?"}],
        )

        # Input: completely unrelated text
        input_file = tmp / "clean.jsonl"
        write_jsonl(
            input_file,
            [
                {"id": "1", "text": "The quick brown fox jumps over the lazy dog."},
                {
                    "id": "2",
                    "text": "Machine learning models require large amounts of data.",
                },
            ],
        )

        scanner = ContaminationScanner(
            {
                "benchmarks_path": str(benchmarks_dir),
                "reports_path": str(reports_dir),
            }
        )
        approved, report = scanner.scan_dataset(input_file, "test-team", "clean-batch")

        assert approved, "Clean dataset should be APPROVED"
        assert report["status"] == "APPROVED"
        assert report["contaminated_count"] == 0
        assert "run_id" in report
        assert "scanner_commit" in report
        print("✓ clean dataset → APPROVED")


def test_contaminated_dataset_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        benchmarks_dir = tmp / "benchmarks"
        benchmarks_dir.mkdir()
        reports_dir = tmp / "reports"

        benchmark_question = (
            "What is the powerhouse of the cell and why is it called that?"
        )

        write_jsonl(
            benchmarks_dir / "dummy_test.jsonl",
            [{"question": benchmark_question}],
        )

        # Input: one verbatim benchmark sample + one clean sample
        input_file = tmp / "contaminated.jsonl"
        write_jsonl(
            input_file,
            [
                {"id": "1", "text": benchmark_question},
                {"id": "2", "text": "An unrelated sentence about cooking pasta."},
            ],
        )

        scanner = ContaminationScanner(
            {
                "benchmarks_path": str(benchmarks_dir),
                "reports_path": str(reports_dir),
            }
        )
        approved, report = scanner.scan_dataset(
            input_file, "test-team", "contaminated-batch"
        )

        assert not approved, "Contaminated dataset should be REJECTED"
        assert report["status"] == "REJECTED"
        assert report["contaminated_count"] >= 1
        print("✓ contaminated dataset → REJECTED")


def test_registry_records_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        benchmarks_dir = tmp / "benchmarks"
        benchmarks_dir.mkdir()
        reports_dir = tmp / "reports"

        write_jsonl(
            benchmarks_dir / "dummy_test.jsonl",
            [{"question": "Sample benchmark question for registry test."}],
        )

        input_file = tmp / "input.jsonl"
        write_jsonl(input_file, [{"id": "1", "text": "Some training text."}])

        scanner = ContaminationScanner(
            {
                "benchmarks_path": str(benchmarks_dir),
                "reports_path": str(reports_dir),
            }
        )
        _, report = scanner.scan_dataset(input_file, "test-team", "registry-batch")

        registry = reports_dir / "run_registry.jsonl"
        assert registry.exists(), "run_registry.jsonl should be created"

        records = [json.loads(line) for line in registry.read_text().splitlines()]
        run_id = report["run_id"]

        statuses = {r["status"] for r in records if r["run_id"] == run_id}
        assert "STARTED" in statuses, "Registry must contain STARTED record"
        assert "COMPLETED" in statuses, "Registry must contain COMPLETED record"

        started = next(
            r for r in records if r["run_id"] == run_id and r["status"] == "STARTED"
        )
        assert "config" in started, "STARTED record must include config"
        assert "input_file" in started, "STARTED record must include input_file"
        print("✓ run registry records STARTED + COMPLETED with config and input_file")


if __name__ == "__main__":
    print("Running smoke tests...\n")
    test_clean_dataset_is_approved()
    test_contaminated_dataset_is_rejected()
    test_registry_records_run()
    print("\nAll smoke tests passed.")
