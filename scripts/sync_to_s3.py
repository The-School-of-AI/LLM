#!/usr/bin/env python3
"""
Sync logs and experiments/runs to S3. One loop:
  - Every 5 min: sync logs + runs (excluding checkpoints).
  - Run checkpoint sync only when there are new or changed checkpoint files
    (state in a flat JSON file so we don't run heavy sync unnecessarily).

Usage:
  uv run sync_to_s3.py              # run forever (loop every 5 min)
  uv run sync_to_s3.py --once       # run one cycle and exit (use this in cron)

Cron (every 5 min):
  */5 * * * * /home/ec2-user/LLM/scripts/sync_to_s3.py --once >> /home/ec2-user/LLM/logs/s3-sync.log 2>&1

Requires: AWS CLI (uses subprocess), no extra Python deps.
State: scripts/.sync_state.json (tracked so we only run checkpoint sync when files change).

-----
Example usage:
# One shot (e.g. from cron every 5 min)
uv run /home/ec2-user/LLM/scripts/sync_to_s3.py --once

# Or run in a loop (one process, sleeps 5 min between cycles)
uv run /home/ec2-user/LLM/scripts/sync_to_s3.py

(crontab -l 2>/dev/null; echo "*/5 * * * * /home/ec2-user/LLM/scripts/sync_to_s3.py --once >> /home/ec2-user/LLM/logs/s3-sync.log 2>&1") | crontab -
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/ec2-user/LLM")
BUCKET = "t-endgame-experiment-logs"
STATE_FILE = Path("~/.sync_state.json").expanduser()
LOOP_INTERVAL_SEC = 5 * 60  # 5 min
# Only consider checkpoint files not modified in last N sec (avoid uploading while training is writing)
CHECKPOINT_MIN_AGE_SEC = 120
# Retries for checkpoint sync (handles CONTENT_LENGTH when file was still being written)
CHECKPOINT_SYNC_RETRIES = 3
CHECKPOINT_SYNC_RETRY_DELAY_SEC = 90


def run(cmd: list[str], log_prefix: str = "") -> bool:
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return True
    except subprocess.CalledProcessError as e:
        if log_prefix:
            print(f"{log_prefix} failed: {e.stderr or e}", file=sys.stderr)
        return False


def sync_logs() -> bool:
    return run(
        [
            "aws", "s3", "sync",
            str(REPO_ROOT / "logs"),
            f"s3://{BUCKET}/logs",
            "--only-show-errors", "--no-progress",
        ],
        log_prefix="Sync logs",
    )


def sync_runs_no_checkpoints() -> bool:
    return run(
        [
            "aws", "s3", "sync",
            str(REPO_ROOT / "experiments/runs"),
            f"s3://{BUCKET}/experiments/runs",
            "--exclude", "*checkpoints*",
            "--only-show-errors", "--no-progress",
        ],
        log_prefix="Sync runs (no checkpoints)",
    )


def sync_checkpoints() -> bool:
    cmd = [
        "aws", "s3", "sync",
        str(REPO_ROOT / "experiments/runs"),
        f"s3://{BUCKET}/experiments/runs",
        "--exclude", "*",
        "--include", "*/checkpoints/*",
        "--only-show-errors", "--no-progress",
    ]
    for attempt in range(1, CHECKPOINT_SYNC_RETRIES + 1):
        if run(cmd, log_prefix=f"Sync checkpoints (attempt {attempt}/{CHECKPOINT_SYNC_RETRIES})"):
            return True
        if attempt < CHECKPOINT_SYNC_RETRIES:
            time.sleep(CHECKPOINT_SYNC_RETRY_DELAY_SEC)
    return False


def list_checkpoint_state(runs_dir: Path) -> dict[str, list[float | int]]:
    """Return dict: rel_path -> [mtime, size] for every file under */checkpoints/*.
    Only includes files not modified in the last CHECKPOINT_MIN_AGE_SEC so we don't
    trigger sync (or upload) while training is still writing.
    """
    state = {}
    if not runs_dir.exists():
        return state
    now = time.time()
    for p in runs_dir.rglob("*"):
        if p.is_file() and "/checkpoints/" in str(p):
            try:
                stat = p.stat()
                if now - stat.st_mtime < CHECKPOINT_MIN_AGE_SEC:
                    continue  # skip files still being written
                rel = str(p.relative_to(REPO_ROOT))
                state[rel] = [stat.st_mtime, stat.st_size]
            except (OSError, ValueError):
                continue
    return state


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, sort_keys=True)


def run_one_cycle() -> None:
    sync_logs()
    sync_runs_no_checkpoints()

    runs_dir = REPO_ROOT / "experiments/runs"
    current = list_checkpoint_state(runs_dir)
    saved = load_state().get("checkpoints", {})

    if current != saved:
        if sync_checkpoints():
            save_state({"checkpoints": current})
    # else: no change, skip heavy checkpoint sync


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync logs and runs to S3 (checkpoints only when new).")
    ap.add_argument("--once", action="store_true", help="Run one cycle and exit.")
    args = ap.parse_args()

    if args.once:
        run_one_cycle()
        return

    print(f"Syncing every {LOOP_INTERVAL_SEC // 60} min. State: {STATE_FILE}", file=sys.stderr)
    while True:
        run_one_cycle()
        time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    main()
