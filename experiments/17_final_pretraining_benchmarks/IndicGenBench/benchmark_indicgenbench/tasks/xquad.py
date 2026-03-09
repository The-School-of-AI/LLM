"""XQuAD-IN: Multilingual QA (Indic question + Indic passage -> answer span)."""

from __future__ import annotations

import logging
from typing import Any

from benchmark_indicgenbench.metrics.qa import compute_qa_metrics
from benchmark_indicgenbench.models.base import GenerationModelBase

logger = logging.getLogger(__name__)


def _extract_answer_text(answer_field: Any) -> str:
    """Extract answer text from various formats: str, dict, or list of dicts."""
    if isinstance(answer_field, str):
        return answer_field
    if isinstance(answer_field, dict):
        return answer_field.get("text", "")
    if isinstance(answer_field, list) and answer_field:
        first = answer_field[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return str(first)
    return ""


def _build_prompt(passage: str, question: str) -> str:
    # Truncate passage to avoid exceeding context
    passage_truncated = passage[:2000]
    return (
        f"Given the context, answer the question with a short answer.\n\n"
        f"Context: {passage_truncated}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def run_xquad(
    data_by_lang: dict[str, list[dict[str, Any]]],
    model: GenerationModelBase,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Run XQuAD-IN evaluation.

    Expected data fields: 'context' or 'passage', 'question', 'answers' or 'answer'
    """
    results_by_lang: dict[str, Any] = {}

    for lang, samples in data_by_lang.items():
        predictions = []
        references = []

        for sample in samples:
            passage = sample.get("context") or sample.get("passage", "")
            question = sample.get("question", "")

            # answers: [{'answer_start': int, 'text': str}, ...]
            answer_field = sample.get("answers") or sample.get("answer", "")
            answer = _extract_answer_text(answer_field)

            if not passage or not question or not answer:
                continue

            prompt = _build_prompt(passage, question)
            pred = model.generate(prompt, max_new_tokens=max_new_tokens)
            # Take only the first line as answer
            pred = pred.split("\n")[0].strip()
            predictions.append(pred)
            references.append(str(answer))

        if predictions:
            metrics = compute_qa_metrics(predictions, references)
            results_by_lang[lang] = metrics
            logger.info("  xquad/%s: EM=%.4f, F1=%.4f (n=%d)",
                        lang, metrics["exact_match"], metrics["token_f1"], len(predictions))

    return results_by_lang
