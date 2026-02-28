"""
Generation evaluation: for each (query, passage), generate answer and compute EM, F1, optional BLEU/ROUGE.
Returns per-language metrics and a flat list of samples for optional RAGAS.
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
    use_f1: bool = True,
    use_squad_normalize: bool = False,
    use_bleu: bool = False,
    use_rouge: bool = False,
    **gen_kwargs: Any,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """
    Run generation evaluation per language. Each row has query, passage, answer.
    Returns (results_by_lang, samples_for_evaluators) where samples_for_evaluators is a flat list
    of {query, passage, prediction, answer, contexts} for optional RAGAS.
    """
    results: dict[str, dict[str, float]] = {}
    all_samples: list[dict[str, Any]] = []
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
            all_samples.append({
                "query": query,
                "passage": passage,
                "prediction": pred,
                "answer": gold,
                "contexts": [passage],
                "language": lang,
            })
        if not references:
            results[lang] = {"exact_match": 0.0, "n": 0.0}
            if use_f1:
                results[lang]["token_f1"] = 0.0
            if use_bleu:
                results[lang]["bleu"] = 0.0
            if use_rouge:
                results[lang]["rouge_l"] = 0.0
            logger.warning("%s: no valid (query, answer) pairs", lang)
            continue
        metrics = compute_generation_metrics(
            predictions,
            references,
            use_f1=use_f1,
            use_squad_normalize=use_squad_normalize,
            use_bleu=use_bleu,
            use_rouge=use_rouge,
        )
        results[lang] = metrics
        extra = f" F1={metrics['token_f1']:.4f}" if use_f1 and "token_f1" in metrics else ""
        logger.info("%s: EM=%.4f n=%d%s", lang, metrics["exact_match"], int(metrics["n"]), extra)
    return results, all_samples
