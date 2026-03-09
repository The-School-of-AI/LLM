"""Summarization metrics: ROUGE-1/2/L and METEOR."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_summarization_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "meteor": 0.0, "n": 0.0}

    result: dict[str, float] = {"n": float(n)}

    # ROUGE via rouge-score
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for key in totals:
                totals[key] += scores[key].fmeasure
        for key in totals:
            result[key] = totals[key] / n
    except ImportError:
        logger.warning("rouge-score not installed, skipping ROUGE")
        result["rouge1"] = 0.0
        result["rouge2"] = 0.0
        result["rougeL"] = 0.0

    # METEOR via nltk
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score as nltk_meteor
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        total_meteor = 0.0
        for pred, ref in zip(predictions, references):
            try:
                total_meteor += nltk_meteor([ref.split()], pred.split())
            except Exception:
                pass
        result["meteor"] = total_meteor / n
    except ImportError:
        logger.warning("nltk not installed, skipping METEOR")
        result["meteor"] = 0.0

    return result
