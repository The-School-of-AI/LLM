"""Command-line interface for IndicMT-Eval benchmark."""

from __future__ import annotations

import argparse
import logging
import sys

from benchmark_indic_mt_eval.config import (
    ALL_LANGUAGES,
    BenchmarkConfig,
    load_config,
)
from benchmark_indic_mt_eval.runner import run_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IndicMT-Eval: Meta-evaluate MT metrics for Indian languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config
    p.add_argument("--config", type=str, help="YAML config file path")

    # Data
    p.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help=f"Languages to evaluate. Options: {ALL_LANGUAGES} or 'all'",
    )
    p.add_argument("--split", choices=["train", "val", "test"], default=None)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per language",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local data directory (skips download)",
    )

    # Metrics
    p.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compute (e.g., bleu chrf ter bertscore comet)",
    )
    p.add_argument("--device", default=None, help="Device for neural metrics")

    # Run
    p.add_argument("--output", "-o", default=None, help="Output JSON path")
    p.add_argument(
        "--levels",
        nargs="+",
        default=None,
        choices=["segment", "system"],
    )
    p.add_argument("--verbose", "-v", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build overrides from CLI args
    overrides: dict = {}
    if args.languages:
        overrides.setdefault("data", {})["languages"] = args.languages
    if args.split:
        overrides.setdefault("data", {})["split"] = args.split
    if args.max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = args.max_samples
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.metrics:
        overrides.setdefault("metrics", {})["metrics"] = args.metrics
    if args.device:
        overrides.setdefault("metrics", {})["device"] = args.device
    if args.output:
        overrides.setdefault("run", {})["output"] = args.output
    if args.levels:
        overrides.setdefault("run", {})["levels"] = args.levels
    if args.verbose:
        overrides.setdefault("run", {})["verbose"] = True

    # Load config
    if args.config:
        config = load_config(args.config, overrides=overrides)
    else:
        config = BenchmarkConfig.from_dict(overrides)

    # Run
    results = run_benchmark(config)

    # Print summary
    print("\n=== IndicMT-Eval Results ===")
    for level_key, summary in results.get("summary", {}).items():
        print(f"\n{level_key}:")
        for metric, corrs in summary.items():
            print(
                f"  {metric:12s}  Pearson={corrs['pearson']:.4f}  "
                f"Kendall-\u03c4={corrs['kendall_tau']:.4f}"
            )


if __name__ == "__main__":
    main()
