#!/usr/bin/env python3
"""
Replay a past scan run from the run registry.

Looks up the given run_id in run_registry.jsonl and prints the exact command
needed to reproduce it — including the commit hash, input file, team, and
batch name that were used.

Usage:
    python scripts/replay.py <run_id>
    python scripts/replay.py <run_id> --execute   # also re-runs the scan
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_registry(reports_path: Path) -> list[dict]:
    """Load all records from the run registry."""
    registry_file = reports_path / "run_registry.jsonl"
    if not registry_file.exists():
        print(f"Error: registry not found at {registry_file}")
        sys.exit(1)
    records = []
    for line in registry_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def find_run(records: list[dict], run_id: str) -> dict:
    """Return the STARTED record for the given run_id."""
    started = [
        r for r in records if r.get("run_id") == run_id and r.get("status") == "STARTED"
    ]
    if not started:
        print(f"Error: no STARTED record found for run_id '{run_id}'")
        print("Available run IDs:")
        for r in records:
            if r.get("status") == "STARTED":
                print(f"  {r['run_id']}  {r['timestamp']}  {r['dataset']}")
        sys.exit(1)
    return started[0]


def find_outcome(records: list[dict], run_id: str) -> dict | None:
    """Return the COMPLETED or FAILED record for the given run_id."""
    outcomes = [
        r
        for r in records
        if r.get("run_id") == run_id and r.get("status") in ("COMPLETED", "FAILED")
    ]
    return outcomes[-1] if outcomes else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a past scan run.")
    parser.add_argument("run_id", help="UUID of the run to replay")
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory containing run_registry.jsonl (default: reports/)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually re-run the scan (not just print the command)",
    )
    args = parser.parse_args()

    records = load_registry(args.reports_dir)
    started = find_run(records, args.run_id)
    outcome = find_outcome(records, args.run_id)

    print("=" * 60)
    print(f"Run ID:       {started['run_id']}")
    print(f"Started:      {started['timestamp']}")
    print(f"Team:         {started['team']}")
    print(f"Dataset:      {started['dataset']}")
    print(f"Input file:   {started.get('input_file', 'unknown')}")
    print(f"Commit:       {started['scanner_commit']}")
    print(f"Repo dirty:   {started['repo_dirty']}")
    if outcome:
        print(f"Outcome:      {outcome['status']}", end="")
        if outcome.get("result"):
            print(f" → {outcome['result']}", end="")
        if outcome.get("failure_type"):
            print(f" ({outcome['failure_type']})", end="")
        print()
    if started.get("config"):
        print(f"Config:       {json.dumps(started['config'])}")
    print("=" * 60)

    input_file = started.get("input_file", "UNKNOWN")
    team = started["team"]
    dataset = started["dataset"]
    commit = started["scanner_commit"]

    print("\nTo replay this run on the same commit:")
    print(f"  git checkout {commit}")
    print(f'  python scripts/scan.py "{input_file}" "{team}" "{dataset}"')
    print()

    if commit == "unknown":
        print(
            "Warning: commit was not recorded — exact code state cannot be guaranteed."
        )

    if args.execute:
        if not Path(input_file).exists():
            print(f"Error: input file no longer exists at: {input_file}")
            sys.exit(1)

        from core.scanner import ContaminationScanner

        config = started.get("config", {})
        config["reports_path"] = str(args.reports_dir)

        scanner = ContaminationScanner(config)
        approved, _ = scanner.scan_dataset(input_file, team, dataset)
        sys.exit(0 if approved else 1)


if __name__ == "__main__":
    main()
