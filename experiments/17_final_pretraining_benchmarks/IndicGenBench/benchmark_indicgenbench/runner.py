"""Orchestrates data loading, model creation, task evaluation, and result aggregation."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from benchmark_indicgenbench.config import BenchmarkConfig, load_config
from benchmark_indicgenbench.data.loader import load_task_data
from benchmark_indicgenbench.models.registry import get_model

logger = logging.getLogger(__name__)

TASK_RUNNERS = {
    "crosssum": "benchmark_indicgenbench.tasks.crosssum:run_crosssum",
    "flores": "benchmark_indicgenbench.tasks.flores:run_flores",
    "xquad": "benchmark_indicgenbench.tasks.xquad:run_xquad",
    "xorqa": "benchmark_indicgenbench.tasks.xorqa:run_xorqa",
}


def _import_runner(task: str):
    """Dynamically import and return the task runner function."""
    module_path, func_name = TASK_RUNNERS[task].rsplit(":", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for name in ("httpx", "httpcore", "datasets", "urllib3", "transformers", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def run_benchmark(
    config: BenchmarkConfig | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config, data, model; run requested tasks; return aggregated results."""
    if config is None:
        config = load_config(config_path=config_path, overrides=overrides or {})

    setup_logging(config.run.log_level)
    random.seed(config.run.seed)

    languages = config.resolve_languages()
    max_samples = config.resolve_max_samples()

    logger.info("IndicGenBench Evaluation")
    logger.info("  Tasks: %s", config.run.tasks)
    logger.info("  Languages: %s", languages)
    logger.info("  Split: %s, Max samples/lang: %s", config.data.split, max_samples)
    logger.info("  Model backend: %s", config.model.backend)
    if config.model.model_name_or_path:
        logger.info("  Model: %s", config.model.model_name_or_path)

    # Create model
    model = get_model(
        config.model.backend,
        model_name_or_path=config.model.model_name_or_path,
        device=config.model.device,
        torch_dtype=config.model.torch_dtype,
    )

    results: dict[str, Any] = {
        "config": {
            "split": config.data.split,
            "languages": languages,
            "max_samples_per_lang": max_samples,
            "backend": config.model.backend,
            "model_name_or_path": config.model.model_name_or_path,
            "max_new_tokens": config.model.max_new_tokens,
            "seed": config.run.seed,
        },
        "tasks": {},
    }

    for task in config.run.tasks:
        if task not in TASK_RUNNERS:
            logger.warning("Unknown task: %s, skipping", task)
            continue

        logger.info("Loading data for task: %s", task)
        task_data = load_task_data(
            task=task,
            languages=languages,
            split=config.data.split,
            max_samples_per_lang=max_samples,
            cache_dir=config.data.cache_dir,
        )

        if not task_data:
            logger.warning("No data loaded for task %s, skipping", task)
            continue

        logger.info("Running task: %s", task)
        runner_fn = _import_runner(task)

        # QA tasks use shorter generation
        effective_max_tokens = config.model.max_new_tokens
        if task in ("xquad", "xorqa"):
            effective_max_tokens = min(effective_max_tokens, 64)

        task_results = runner_fn(
            data_by_lang=task_data,
            model=model,
            max_new_tokens=effective_max_tokens,
        )
        results["tasks"][task] = task_results

    # Write output
    out_path = _resolve_output_path(config)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Results written to %s", out_path)

    # Print summary
    _print_summary(results)

    return results


def _resolve_output_path(config: BenchmarkConfig) -> Path | None:
    if config.run.output_file:
        return Path(config.run.output_file)
    if config.run.output_dir:
        return Path(config.run.output_dir) / "results.json"
    return None


def _print_summary(results: dict[str, Any]) -> None:
    """Print a concise summary table."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    for task, task_results in results.get("tasks", {}).items():
        logger.info("")
        logger.info("Task: %s", task)
        logger.info("-" * 40)

        for lang, metrics in task_results.items():
            parts = []
            for k, v in metrics.items():
                if k == "n":
                    continue
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
            n = int(metrics.get("n", 0))
            logger.info("  %s (n=%d): %s", lang, n, ", ".join(parts))

    logger.info("=" * 60)
