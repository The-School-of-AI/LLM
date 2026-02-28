#!/usr/bin/env python3
"""
One-command runner for the S3 contamination scan.

Reads project config from ``config.json`` (and optional ``aws.json``),
ensures benchmarks exist, then runs the scan against the configured S3 URI.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def merge_aws_env(env: dict[str, str], aws_cfg: dict) -> dict[str, str]:
    updated = dict(env)
    mapping = {
        "access_key_id": "AWS_ACCESS_KEY_ID",
        "secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "session_token": "AWS_SESSION_TOKEN",
        "region": "AWS_DEFAULT_REGION",
        "profile": "AWS_PROFILE",
    }
    for src, dst in mapping.items():
        value = aws_cfg.get(src)
        if value:
            updated[dst] = str(value)
    return updated


def ensure_benchmarks(config: dict, env: dict[str, str]) -> None:
    benchmarks_dir = PROJECT_ROOT / config.get("benchmarks_dir", "benchmarks")
    auto_download = bool(config.get("auto_download_benchmarks", True))

    if benchmarks_dir.exists() and any(benchmarks_dir.glob("*_test.jsonl")):
        return

    if not auto_download:
        raise FileNotFoundError(
            f"Benchmarks not found in {benchmarks_dir}. "
            "Enable auto_download_benchmarks or run scripts/download_benchmarks.py first."
        )

    print(f"Benchmarks not found in {benchmarks_dir}. Downloading...")
    subprocess.run(
        [sys.executable, "scripts/download_benchmarks.py"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def build_scan_command(config: dict) -> list[str]:
    required = ["s3_uri", "team_name", "batch_name"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(
            f"Missing required fields in config.json: {', '.join(missing)}"
        )

    cmd = [
        sys.executable,
        "scripts/scan_from_s3.py",
        str(config["s3_uri"]),
        str(config["team_name"]),
        str(config["batch_name"]),
        "--benchmarks-dir",
        str(config.get("benchmarks_dir", "benchmarks")),
        "--reports-dir",
        str(config.get("reports_dir", "reports")),
    ]
    if config.get("build_workers") is not None:
        cmd.extend(["--build-workers", str(config["build_workers"])])
    if config.get("enable_semantic") is False:
        cmd.append("--no-semantic")
    return cmd


def main() -> None:
    start_time = time.perf_counter()

    config_path = PROJECT_ROOT / "config.json"
    aws_cfg_path = PROJECT_ROOT / "aws.json"

    try:
        config = load_json(config_path)
    except Exception as exc:
        print(f"Error loading config.json: {exc}")
        sys.exit(1)

    env = dict(os.environ)

    if config.get("aws_region"):
        env["AWS_DEFAULT_REGION"] = str(config["aws_region"])
    if config.get("aws_profile"):
        env["AWS_PROFILE"] = str(config["aws_profile"])

    if aws_cfg_path.exists():
        try:
            aws_cfg = load_json(aws_cfg_path)
            env = merge_aws_env(env, aws_cfg)
            print("Loaded AWS credentials from aws.json")
        except Exception as exc:
            print(f"Error loading aws.json: {exc}")
            sys.exit(1)

    try:
        ensure_benchmarks(config, env)
        cmd = build_scan_command(config)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        sys.exit(1)

    print("Running S3 scan")
    print(f"  S3 URI:   {config['s3_uri']}")
    print(f"  Team:     {config['team_name']}")
    print(f"  Batch:    {config['batch_name']}")
    print()

    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    elapsed = time.perf_counter() - start_time
    print()
    print(f"Total runtime: {format_duration(elapsed)} ({elapsed:.1f}s)")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
