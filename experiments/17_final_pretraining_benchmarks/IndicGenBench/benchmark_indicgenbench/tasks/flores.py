"""Flores-IN: Machine translation (bidirectional EN <-> Indic)."""

from __future__ import annotations

import logging
from typing import Any

from benchmark_indicgenbench.config import LANGUAGE_NAMES
from benchmark_indicgenbench.metrics.translation import compute_translation_metrics
from benchmark_indicgenbench.models.base import GenerationModelBase

logger = logging.getLogger(__name__)


def _build_prompt(source_text: str, src_lang: str, tgt_lang: str) -> str:
    src_name = LANGUAGE_NAMES.get(src_lang, src_lang)
    tgt_name = LANGUAGE_NAMES.get(tgt_lang, tgt_lang)
    return (
        f"Translate the following from {src_name} to {tgt_name}.\n\n"
        f"{src_name}: {source_text}\n\n"
        f"{tgt_name}:"
    )


def run_flores(
    data_by_lang: dict[str, list[dict[str, Any]]],
    model: GenerationModelBase,
    max_new_tokens: int = 128,
) -> dict[str, Any]:
    """Run Flores-IN evaluation (English -> Indic direction).

    Expected data fields: 'source' or 'sentence_eng_Latn' (English), 'target' or 'sentence_{lang}' (Indic)
    """
    results_by_lang: dict[str, Any] = {}

    for lang, samples in data_by_lang.items():
        predictions = []
        references = []

        for sample in samples:
            # Handle different field conventions
            source = (
                sample.get("source")
                or sample.get("sentence_eng_Latn")
                or sample.get("input", "")
            )
            target = (
                sample.get("target")
                or sample.get("output")
                or sample.get("sentence", "")
            )

            if not source or not target:
                continue

            prompt = _build_prompt(source, "en", lang)
            pred = model.generate(prompt, max_new_tokens=max_new_tokens)
            predictions.append(pred)
            references.append(target)

        if predictions:
            metrics = compute_translation_metrics(predictions, references)
            results_by_lang[lang] = metrics
            logger.info("  flores/%s: BLEU=%.2f, chrF=%.2f, METEOR=%.4f (n=%d)",
                        lang, metrics["bleu"], metrics["chrf"], metrics["meteor"], len(predictions))

    return results_by_lang
