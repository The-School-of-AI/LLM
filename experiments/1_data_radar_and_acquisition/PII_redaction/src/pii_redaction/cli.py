from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .processor import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PII redaction pipeline on JSONL inputs.")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input JSONL files, directories, or glob patterns."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for redacted outputs and metrics.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable execution and reprocess all inputs."
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    manifest = run_pipeline(args.input, Path(args.output_dir), config, resume=not args.no_resume)
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
