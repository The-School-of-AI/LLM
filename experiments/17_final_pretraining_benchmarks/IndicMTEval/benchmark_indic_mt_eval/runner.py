"""End-to-end benchmark orchestrator."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from benchmark_indic_mt_eval.config import BenchmarkConfig, LANGUAGE_NAMES
from benchmark_indic_mt_eval.data.loader import load_benchmark_data
from benchmark_indic_mt_eval.evaluation.evaluator import evaluate_language

# Ensure metrics are registered
import benchmark_indic_mt_eval.metrics  # noqa: F401

logger = logging.getLogger(__name__)


def run_benchmark(config: BenchmarkConfig) -> dict:
    start = time.time()

    # Validate requested metrics are available
    from benchmark_indic_mt_eval.metrics.registry import list_metrics

    available = list_metrics()
    for m in config.metrics.metrics:
        if m not in available:
            raise ValueError(
                f"Metric '{m}' not available. "
                f"Installed: {available}. "
                f"Install optional deps for neural metrics."
            )

    # Load data
    logger.info("Loading data for languages: %s", config.data.languages)
    data = load_benchmark_data(
        languages=config.data.languages,
        split=config.data.split,
        data_dir=config.data.data_dir,
        max_samples=config.data.max_samples,
    )

    # Evaluate per language
    results: dict = {
        "config": {
            "languages": config.data.languages,
            "split": config.data.split,
            "max_samples": config.data.max_samples,
            "metrics": config.metrics.metrics,
            "levels": config.run.levels,
        },
        "results": {},
        "summary": {},
    }

    per_lang_results: dict[str, dict] = {}
    for lang, samples in data.items():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        logger.info(
            "Evaluating %s (%s): %d samples", lang_name, lang, len(samples)
        )
        per_lang_results[lang] = evaluate_language(
            samples, config.metrics.metrics, config.run.levels
        )

    # Restructure: level -> lang -> metric -> correlations
    for level in config.run.levels:
        level_key = f"{level}_level"
        results["results"][level_key] = {}
        for lang in config.data.languages:
            if level_key in per_lang_results.get(lang, {}):
                results["results"][level_key][lang] = per_lang_results[lang][
                    level_key
                ]

    # Compute summary (average across languages)
    for level in config.run.levels:
        level_key = f"{level}_level"
        if level_key not in results["results"]:
            continue
        summary: dict[str, dict[str, float]] = {}
        for metric_name in config.metrics.metrics:
            pearson_vals = []
            kendall_vals = []
            for lang_data in results["results"][level_key].values():
                if metric_name in lang_data:
                    pearson_vals.append(lang_data[metric_name]["pearson"])
                    kendall_vals.append(lang_data[metric_name]["kendall_tau"])
            if pearson_vals:
                summary[metric_name] = {
                    "pearson": sum(pearson_vals) / len(pearson_vals),
                    "kendall_tau": sum(kendall_vals) / len(kendall_vals),
                }
        results["summary"][f"{level}_level_avg"] = summary

    elapsed = time.time() - start
    results["config"]["elapsed_seconds"] = round(elapsed, 2)

    # Write output
    output_path = Path(config.run.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s (%.1fs)", output_path, elapsed)

    return results
