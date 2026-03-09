"""CLI: argparse and config overrides, then run_benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark_indicgenbench.runner import run_benchmark


def _build_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.split is not None:
        overrides.setdefault("data", {})["split"] = args.split
    if args.lang is not None:
        overrides.setdefault("data", {})["languages"] = [args.lang] if args.lang != "all" else ["all"]
    if args.max_samples is not None:
        overrides.setdefault("data", {})["max_samples_per_lang"] = args.max_samples
    if args.model_backend is not None:
        overrides.setdefault("model", {})["backend"] = args.model_backend
    if args.model_name is not None:
        overrides.setdefault("model", {})["model_name_or_path"] = args.model_name
    if args.device is not None:
        overrides.setdefault("model", {})["device"] = args.device
    if args.max_new_tokens is not None:
        overrides.setdefault("model", {})["max_new_tokens"] = args.max_new_tokens
    if args.tasks is not None:
        overrides.setdefault("run", {})["tasks"] = args.tasks
    if args.output is not None:
        overrides.setdefault("run", {})["output_file"] = args.output
    if args.output_dir is not None:
        overrides.setdefault("run", {})["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    if args.log_level is not None:
        overrides.setdefault("run", {})["log_level"] = args.log_level
    return overrides


def run() -> int:
    parser = argparse.ArgumentParser(
        description="IndicGenBench: Evaluate LLM generation on Indic languages (summarization, translation, QA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data
    parser.add_argument("--split", choices=["dev", "test"], default=None, help="Dataset split")
    parser.add_argument("--lang", default=None, help="Language code (e.g. 'hi', 'bn') or 'all'")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per language")

    # Model
    parser.add_argument("--model-backend", choices=["small", "hf"], default=None, help="Model backend")
    parser.add_argument("--model-name", default=None, help="HuggingFace model path (e.g. google/gemma-3-1b-it)")
    parser.add_argument("--device", default=None, help="Device: cpu, cuda, cuda:0, mps")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Max tokens to generate")

    # Run
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        choices=["crosssum", "flores", "xquad", "xorqa"],
        help="Tasks to evaluate",
    )
    parser.add_argument("--output", "-o", default=None, help="Output JSON file path")
    parser.add_argument("--output-dir", default=None, help="Output directory (writes results.json)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)

    args = parser.parse_args()
    config_path = Path(args.config) if args.config else None
    overrides = _build_overrides(args)
    run_benchmark(config_path=config_path, overrides=overrides)
    return 0


if __name__ == "__main__":
    sys.exit(run())
