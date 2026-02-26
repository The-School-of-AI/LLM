#!/usr/bin/env python3
"""
One-command runner for the S3 contamination scan flow.

Reads project config from ``S3_scan.json`` (and optional ``S3_aws.json``),
ensures benchmarks exist, then runs ``scripts/scan_from_s3.py``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    has_benchmarks = benchmarks_dir.exists() and any(benchmarks_dir.glob("*_test.jsonl"))
    if has_benchmarks:
        return

    if not auto_download:
        raise FileNotFoundError(
            f"Benchmarks not found in {benchmarks_dir}. "
            "Enable auto_download_benchmarks or run download_benchmarks.py first."
        )

    print(f"Benchmarks not found in {benchmarks_dir}. Downloading...")
    cmd = [sys.executable, "scripts/download_benchmarks.py"]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def build_scan_command(config: dict) -> list[str]:
    required = ["s3_uri", "team_name", "batch_name"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required fields in S3_scan.json: {', '.join(missing)}")

    cmd = [
        sys.executable,
        "scripts/scan_from_s3.py",
        str(config["s3_uri"]),
        str(config["team_name"]),
        str(config["batch_name"]),
    ]

    cmd.extend(["--benchmarks-dir", str(config.get("benchmarks_dir", "benchmarks"))])
    cmd.extend(["--reports-dir", str(config.get("reports_dir", "reports"))])
    return cmd


def main() -> None:
    scan_cfg_path = PROJECT_ROOT / "S3_scan.json"
    aws_cfg_path = PROJECT_ROOT / "S3_aws.json"

    try:
        config = load_json(scan_cfg_path)
    except Exception as exc:
        print(f"Error loading S3_scan.json: {exc}")
        sys.exit(1)

    env = dict(os.environ)

    # Non-secret AWS settings can live in S3_scan.json
    if config.get("aws_region"):
        env["AWS_DEFAULT_REGION"] = str(config["aws_region"])
    if config.get("aws_profile"):
        env["AWS_PROFILE"] = str(config["aws_profile"])

    # Optional local credentials file (keep local; avoid committing real secrets)
    if aws_cfg_path.exists():
        try:
            aws_cfg = load_json(aws_cfg_path)
            env = merge_aws_env(env, aws_cfg)
            print("Loaded AWS settings from S3_aws.json")
        except Exception as exc:
            print(f"Error loading S3_aws.json: {exc}")
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
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
