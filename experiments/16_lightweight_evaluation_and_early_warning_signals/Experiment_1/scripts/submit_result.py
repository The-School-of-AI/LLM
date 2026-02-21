#!/usr/bin/env python3
"""
Submit a local result JSON file to the central Flask collector.

Usage:
    python scripts/submit_result.py \\
        --file results/raw/step_500_int4_20240301T120000Z.json \\
        --server http://192.168.1.100:5001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Submit result to central collector")
    p.add_argument("--file", type=Path, required=True, help="Path to result JSON file")
    p.add_argument("--server", required=True, help="Base URL of the collector (e.g. http://host:5001)")
    args = p.parse_args()

    if not args.file.exists():
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    try:
        import urllib.request
        import urllib.error

        with open(args.file) as f:
            payload = f.read()

        url = args.server.rstrip("/") + "/submit"
        req = urllib.request.Request(
            url,
            data=payload.encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"Success: {body}")
    except Exception as e:
        print(f"ERROR submitting result: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
