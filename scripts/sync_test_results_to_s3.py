#!/usr/bin/env python3
"""
Sync test results folder to S3 while a test is running.

Usage:
  # Sync results for a specific test (run in loop mode)
  uv run sync_test_results_to_s3.py Test_2_20-step_save_init_model
  
  # Run once and exit (useful for cron)
  uv run sync_test_results_to_s3.py Test_2_20-step_save_init_model --once
  
  # Specify full path
  uv run sync_test_results_to_s3.py --test-path LLM/experiments/tests/Test_2_20-step_save_init_model

Cron (every 5 min):
  */5 * * * * /mnt/local-nvme/LLM/LLM/scripts/sync_test_results_to_s3.py Test_2_20-step_save_init_model --once | tee -a /home/ec2-user/LLM/logs/test-results-sync.log 2>&1

Requires: AWS CLI (uses subprocess), no extra Python deps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/mnt/local-nvme/LLM/LLM")
BUCKET = "t-endgame-experiment-logs"
LOOP_INTERVAL_SEC = 5 * 60  # 5 min


def run(cmd: list[str], log_prefix: str = "") -> bool:
    """Run AWS CLI command and return success status."""
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if log_prefix:
            print(f"{log_prefix}: success", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        if log_prefix:
            print(f"{log_prefix} failed: {e.stderr or e}", file=sys.stderr)
        return False


def sync_test_results(test_path: Path) -> bool:
    """Sync the results folder from a test to S3."""
    results_dir = test_path / "results"
    
    if not results_dir.exists():
        print(f"Warning: Results directory does not exist: {results_dir}", file=sys.stderr)
        return False
    
    # Extract test name from path (e.g., "Test_2_20-step_save_init_model")
    test_name = test_path.name
    
    # S3 destination: s3://bucket/experiments/tests/Test_X/results/
    s3_dest = f"s3://{BUCKET}/experiments/tests/{test_name}/results"
    
    return run(
        [
            "aws", "s3", "sync",
            str(results_dir),
            s3_dest,
            "--only-show-errors",
            "--no-progress",
        ],
        log_prefix=f"Sync results for {test_name}",
    )


def resolve_test_path(test_name_or_path: str) -> Path:
    """Resolve test name to full path."""
    # If it's already a full path, use it
    if "/" in test_name_or_path or test_name_or_path.startswith("LLM/"):
        if test_name_or_path.startswith("LLM/"):
            return REPO_ROOT / test_name_or_path.replace("LLM/", "")
        return Path(test_name_or_path)
    
    # Otherwise, assume it's a test name and look in experiments/tests/
    return REPO_ROOT / "experiments" / "tests" / test_name_or_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sync test results folder to S3 while test is running."
    )
    ap.add_argument(
        "test_name",
        nargs="?",
        help="Test name (e.g., 'Test_2_20-step_save_init_model') or path"
    )
    ap.add_argument(
        "--test-path",
        help="Full path to test folder (alternative to test_name)"
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Run one sync cycle and exit (useful for cron)"
    )
    ap.add_argument(
        "--interval",
        type=int,
        default=LOOP_INTERVAL_SEC,
        help=f"Sync interval in seconds (default: {LOOP_INTERVAL_SEC})"
    )
    
    args = ap.parse_args()
    
    # Determine test path
    if args.test_path:
        test_path = Path(args.test_path)
    elif args.test_name:
        test_path = resolve_test_path(args.test_name)
    else:
        ap.error("Must provide either test_name or --test-path")
    
    if not test_path.exists():
        print(f"Error: Test path does not exist: {test_path}", file=sys.stderr)
        sys.exit(1)
    
    if args.once:
        sync_test_results(test_path)
        return
    
    print(
        f"Syncing results for {test_path.name} every {args.interval // 60} min. "
        f"Press Ctrl+C to stop.",
        file=sys.stderr
    )
    
    try:
        while True:
            sync_test_results(test_path)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user", file=sys.stderr)


if __name__ == "__main__":
    main()
