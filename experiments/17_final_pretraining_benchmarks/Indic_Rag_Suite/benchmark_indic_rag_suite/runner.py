"""
Orchestrates data load, model creation, evaluation, and result aggregation.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

from benchmark_indic_rag_suite.config import BenchmarkConfig, load_config
from benchmark_indic_rag_suite.data.loader import load_benchmark_data
from benchmark_indic_rag_suite.evaluation.generation import run_generation_eval
from benchmark_indic_rag_suite.evaluation.retrieval import run_retrieval_eval
from benchmark_indic_rag_suite.models.registry import get_generation_model, get_retrieval_model

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    for name in ("httpx", "httpcore", "datasets", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def run_benchmark(
    config: BenchmarkConfig | None = None,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config, data, models; run requested tasks; return aggregated results."""
    if config is None:
        config = load_config(config_path=config_path, overrides=overrides or {})

    setup_logging(config.run.log_level)
    random.seed(config.run.seed)

    max_samples = config.resolve_max_samples()
    languages = config.resolve_languages()

    logger.info(
        "Loading data: dataset=%s languages=%s max_samples=%s shard=%d/%d",
        config.data.dataset_name,
        languages,
        max_samples,
        config.data.shard_index,
        config.data.shard_total,
    )

    data_by_lang = load_benchmark_data(
        dataset_name=config.data.dataset_name,
        languages=languages,
        split=config.data.split,
        max_samples_per_lang=max_samples,
        cache_dir=config.data.cache_dir,
        shard_index=config.data.shard_index,
        shard_total=config.data.shard_total,
    )

    for lang, rows in data_by_lang.items():
        logger.info("  %s: %d samples", lang, len(rows))

    results: dict[str, Any] = {
        "config": {
            "split": config.data.split,
            "max_samples_per_lang": max_samples,
            "languages": list(data_by_lang.keys()),
            "shard_index": config.data.shard_index,
            "shard_total": config.data.shard_total,
            "retrieval_backend": config.model.retrieval_backend,
            "generation_backend": config.model.generation_backend,
            "retrieval_mrr_at_k": config.run.retrieval_mrr_at_k,
        },
        "tasks": {},
    }
    if getattr(config.model, "retrieval_model_name_or_path", None):
        results["config"]["retrieval_model_name_or_path"] = config.model.retrieval_model_name_or_path
    if getattr(config.model, "generation_model_name_or_path", None):
        results["config"]["generation_model_name_or_path"] = config.model.generation_model_name_or_path

    if "retrieval" in config.run.tasks:
        if config.data.split == "dev" and max_samples == 20:
            logger.warning(
                "Retrieval with 20 samples per language (dev default) yields high MRR. "
                "For paper-comparable IndicMSMARCO use: --split test and do not set --max-samples. "
                "Official metric is MRR@10 (monolingual pool per language)."
            )
        logger.info("Running retrieval evaluation (backend=%s, MRR@%d)", config.model.retrieval_backend, config.run.retrieval_mrr_at_k)
        retrieval_model = get_retrieval_model(
            config.model.retrieval_backend,
            device=config.model.device,
            model_name_or_path=getattr(config.model, "retrieval_model_name_or_path", None),
        )
        retrieval_data = data_by_lang
        requested_langs = list(data_by_lang.keys())
        if config.run.retrieval_add_cross_lang_negatives and len(requested_langs) == 1:
            logger.info("Paper-retrieval with single language: loading all languages for distractor pool")
            retrieval_data = load_benchmark_data(
                dataset_name=config.data.dataset_name,
                languages=["all"],
                split=config.data.split,
                max_samples_per_lang=max_samples,
                cache_dir=config.data.cache_dir,
                shard_index=config.data.shard_index,
                shard_total=config.data.shard_total,
            )
        retrieval_results = run_retrieval_eval(
            retrieval_data,
            retrieval_model,
            batch_size=config.model.retrieval_batch_size,
            add_cross_lang_negatives=config.run.retrieval_add_cross_lang_negatives,
            mrr_at_k=config.run.retrieval_mrr_at_k,
            recall_at_k_list=(1, 5, 10),
            ndcg_at_k=10,
        )
        if config.run.retrieval_add_cross_lang_negatives and len(requested_langs) == 1:
            results["tasks"]["retrieval"] = {k: retrieval_results[k] for k in requested_langs if k in retrieval_results}
        else:
            results["tasks"]["retrieval"] = retrieval_results
        if config.run.retrieval_add_cross_lang_negatives:
            results["config"]["retrieval_add_cross_lang_negatives"] = True

    if "generation" in config.run.tasks:
        logger.info("Running generation evaluation (backend=%s)", config.model.generation_backend)
        generation_model = get_generation_model(
            config.model.generation_backend,
            device=config.model.device,
            model_name_or_path=getattr(config.model, "generation_model_name_or_path", None),
        )
        results["tasks"]["generation"] = run_generation_eval(
            data_by_lang,
            generation_model,
            max_new_tokens=config.model.generation_max_new_tokens,
        )

    out_dir = config.run.output_dir
    out_file = config.run.output_file
    if out_file:
        path = Path(out_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", path)
    elif out_dir:
        path = Path(out_dir) / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results written to %s", path)

    return results


def main() -> int:
    from benchmark_indic_rag_suite.cli import run as cli_run
    return cli_run()


if __name__ == "__main__":
    sys.exit(main())
