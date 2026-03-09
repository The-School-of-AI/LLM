"""CrossSum-IN: Cross-lingual summarization (English article -> Indic summary)."""

from __future__ import annotations

import logging
from typing import Any

from benchmark_indicgenbench.config import LANGUAGE_NAMES
from benchmark_indicgenbench.metrics.summarization import compute_summarization_metrics
from benchmark_indicgenbench.models.base import GenerationModelBase

logger = logging.getLogger(__name__)


def _build_prompt(article: str, target_lang: str) -> str:
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    # Truncate article to avoid exceeding context
    article_truncated = article[:3000]
    return (
        f"Summarize the following English article in {lang_name}.\n\n"
        f"Article: {article_truncated}\n\n"
        f"Summary in {lang_name}:"
    )


def run_crosssum(
    data_by_lang: dict[str, list[dict[str, Any]]],
    model: GenerationModelBase,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Run CrossSum-IN evaluation.

    Expected data fields: 'text' or 'document' (English article), 'summary' or 'target' (Indic summary)
    """
    results_by_lang: dict[str, Any] = {}

    for lang, samples in data_by_lang.items():
        predictions = []
        references = []

        for sample in samples:
            # Handle different field naming conventions
            article = sample.get("text") or sample.get("document") or sample.get("source", "")
            reference = sample.get("summary") or sample.get("target", "")

            if not article or not reference:
                continue

            prompt = _build_prompt(article, lang)
            pred = model.generate(prompt, max_new_tokens=max_new_tokens)
            predictions.append(pred)
            references.append(reference)

        if predictions:
            metrics = compute_summarization_metrics(predictions, references)
            results_by_lang[lang] = metrics
            logger.info("  crosssum/%s: ROUGE-L=%.4f, METEOR=%.4f (n=%d)",
                        lang, metrics["rougeL"], metrics["meteor"], len(predictions))

    return results_by_lang
