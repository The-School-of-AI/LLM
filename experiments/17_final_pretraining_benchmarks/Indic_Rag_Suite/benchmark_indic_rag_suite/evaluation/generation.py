"""
Generation evaluation: for each (query, passage), generate answer and compute EM (and optional F1).
"""

from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from benchmark_indic_rag_suite.metrics.generation_metrics import compute_generation_metrics
from benchmark_indic_rag_suite.models.base import GenerationModelBase

logger = logging.getLogger(__name__)


def run_generation_eval(
    data_by_lang: dict[str, list[dict]],
    model: GenerationModelBase,
    max_new_tokens: int = 64,
    use_f1: bool = False,
    **gen_kwargs: Any,
) -> dict[str, dict[str, float]]:
    """Run generation evaluation per language. Each row has query, passage, answer."""
    results: dict[str, dict[str, float]] = {}
    for lang, rows in data_by_lang.items():
        predictions: list[str] = []
        references: list[str] = []
        for item in tqdm(rows, desc=f"gen-{lang}", leave=False):
            query = item.get("query", "")
            passage = (item.get("passage") or "").strip()
            gold = (item.get("answer") or "").strip()
            if not gold:
                continue
            pred = model.generate(query, passage, max_new_tokens=max_new_tokens, **gen_kwargs)
            predictions.append(pred)
            references.append(gold)
        if not references:
            results[lang] = {"exact_match": 0.0, "n": 0.0}
            if use_f1:
                results[lang]["token_f1"] = 0.0
            logger.warning("%s: no valid (query, answer) pairs", lang)
            continue
        metrics = compute_generation_metrics(predictions, references, use_f1=use_f1)
        results[lang] = metrics
        logger.info("%s: EM=%.4f n=%d", lang, metrics["exact_match"], int(metrics["n"]))
    return results
