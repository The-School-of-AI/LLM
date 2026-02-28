#!/usr/bin/env python3
"""
Contamination Scanner (N-gram + MinHash only).

Usage:
    python scripts/scan_no_semantic.py <input_file> <team_name> <batch_name>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scanner import ContaminationScanner
from core.utils import get_git_info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contamination scan without semantic detection."
    )
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("team_name", help="Team name recorded in reports")
    parser.add_argument("batch_name", help="Batch name used in report filenames")
    parser.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory containing *_test.jsonl benchmark files (default: benchmarks)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory to write reports and run registry (default: reports)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/indexes",
        help="Directory root for persisted index caches (default: .cache/indexes)",
    )
    parser.add_argument(
        "--build-workers",
        type=int,
        default=None,
        help="Worker threads for N-gram/MinHash index build (default: CPU count)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)
    if not input_path.is_file():
        print(f"Error: path is not a file: {input_path}")
        sys.exit(1)

    git_info = get_git_info()
    if git_info["dirty"] == "true":
        print(
            "⚠ Warning: repository has uncommitted changes. "
            "The report will be marked dirty."
        )

    scanner_config: dict[str, object] = {
        "benchmarks_path": args.benchmarks_dir,
        "reports_path": args.reports_dir,
        "cache_dir": args.cache_dir,
        "enable_semantic": False,
    }
    if args.build_workers is not None:
        scanner_config["build_workers"] = args.build_workers

    try:
        scanner = ContaminationScanner(scanner_config)
        approved, _ = scanner.scan_dataset(input_path, args.team_name, args.batch_name)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error during scan: {exc}")
        sys.exit(1)

    sys.exit(0 if approved else 1)


if __name__ == "__main__":
    main()
