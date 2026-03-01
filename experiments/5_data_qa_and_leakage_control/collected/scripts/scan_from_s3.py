#!/usr/bin/env python3
"""
Stream a .txt Q&A dataset from S3, convert in memory, and run contamination scan.

Usage:
    python scripts/scan_from_s3.py s3://bucket/path/file.txt "Team 4" "group4_batch_01"
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.scanner import ContaminationScanner


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into (bucket, key)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {uri}")

    without_scheme = uri[5:]
    parts = without_scheme.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid S3 URI (need bucket and key): {uri}")
    return parts[0], parts[1]


def extract_qa_pairs(line: str) -> list[str]:
    """Split one packed Q&A line into individual records."""
    pairs = re.findall(r"([^?.]+\?)\s+([^?.]+\.)", line)
    return [f"{q.strip()} {a.strip()}" for q, a in pairs]


def load_records_from_s3_txt(
    s3_uri: str,
) -> tuple[list[dict[str, str]], dict[str, int], list[dict[str, str | int]]]:
    """Stream a text object from S3 and convert it into JSONL-style records."""
    try:
        import boto3
    except ImportError as exc:
        raise ImportError(
            "boto3 is required for S3 streaming. Install it in the project env first."
        ) from exc

    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)

    records: list[dict[str, str]] = []
    unmatched_lines: list[dict[str, str | int]] = []
    total_lines = 0
    non_empty_lines = 0
    zero_pair_lines = 0
    body = response["Body"]
    for raw_line in body.iter_lines():
        total_lines += 1
        if not raw_line:
            continue
        line = raw_line.decode("utf-8").strip()
        if not line:
            continue
        non_empty_lines += 1
        pairs = extract_qa_pairs(line)
        if not pairs:
            zero_pair_lines += 1
            unmatched_lines.append(
                {
                    "line_number": total_lines,
                    "reason": "no regex match",
                    "line_preview": line[:500],
                }
            )
        for pair in pairs:
            records.append({"id": f"qa_{len(records) + 1}", "text": pair})
    stats = {
        "total_lines": total_lines,
        "non_empty_lines": non_empty_lines,
        "zero_pair_lines": zero_pair_lines,
        "qa_pairs_extracted": len(records),
    }
    return records, stats, unmatched_lines


def write_validation_report(
    reports_dir: str,
    s3_uri: str,
    batch_name: str,
    unmatched_lines: list[dict[str, str | int]],
) -> Path | None:
    """Write a local JSONL validation report for lines with zero extracted pairs."""
    if not unmatched_lines:
        return None

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = reports_path / f"{batch_name}_PARSE_GAPS_{timestamp}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for row in unmatched_lines:
            record = {"s3_uri": s3_uri, **row}
            f.write(json.dumps(record) + "\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run contamination scan directly from an S3 .txt dataset."
    )
    parser.add_argument(
        "s3_uri", help="S3 path to input .txt file (s3://bucket/key.txt)"
    )
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
        "--build-workers",
        type=int,
        default=None,
        help="Worker threads for N-gram/MinHash index build (default: CPU count)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Disable semantic detector and run only N-gram + MinHash",
    )
    args = parser.parse_args()

    if not args.s3_uri.lower().endswith(".txt"):
        print("Error: this wrapper currently expects a .txt file in S3")
        sys.exit(1)

    try:
        records, stats, unmatched_lines = load_records_from_s3_txt(args.s3_uri)
    except Exception as exc:
        print(f"Error reading S3 input: {exc}")
        sys.exit(1)

    if not records:
        print("Error: no Q&A pairs were extracted from the S3 text object")
        sys.exit(1)

    non_empty_lines = stats["non_empty_lines"]
    parsed_lines = non_empty_lines - stats["zero_pair_lines"]
    extraction_rate = (parsed_lines / non_empty_lines) * 100 if non_empty_lines else 0.0

    print("Validation summary")
    print(f"  S3 input:            {args.s3_uri}")
    print(f"  Total lines:         {stats['total_lines']}")
    print(f"  Non-empty lines:     {non_empty_lines}")
    print(f"  Parsed lines:        {parsed_lines}")
    print(f"  Zero-pair lines:     {stats['zero_pair_lines']}")
    print(f"  Q&A pairs extracted: {stats['qa_pairs_extracted']}")
    print(f"  Extraction rate:     {extraction_rate:.2f}%")
    validation_report = write_validation_report(
        args.reports_dir, args.s3_uri, args.batch_name, unmatched_lines
    )
    if validation_report:
        print(f"  Parse gaps report:   {validation_report}")
    print()

    scanner_config: dict[str, object] = {
        "benchmarks_path": args.benchmarks_dir,
        "reports_path": args.reports_dir,
        "enable_semantic": not args.no_semantic,
    }
    if args.build_workers is not None:
        scanner_config["build_workers"] = args.build_workers

    scanner = ContaminationScanner(scanner_config)
    approved, _ = scanner.scan_records(
        records,
        args.team_name,
        args.batch_name,
        input_label=args.s3_uri,
    )
    sys.exit(0 if approved else 1)


if __name__ == "__main__":
    main()
