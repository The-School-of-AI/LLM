"""Overlap-based MT metrics: BLEU, chrF++, TER, ROUGE-L."""

from __future__ import annotations

import sacrebleu
from rouge_score import rouge_scorer

from benchmark_indic_mt_eval.metrics.registry import register_metric


@register_metric("bleu")
def compute_bleu(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score / 100.0


@register_metric("chrf")
def compute_chrf(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_chrf(hypothesis, [reference])
    return result.score / 100.0


@register_metric("ter")
def compute_ter(hypothesis: str, reference: str, **kwargs) -> float:
    result = sacrebleu.sentence_ter(hypothesis, [reference])
    return result.score / 100.0


_rouge_scorer_instance = None


def _get_rouge_scorer():
    global _rouge_scorer_instance
    if _rouge_scorer_instance is None:
        _rouge_scorer_instance = rouge_scorer.RougeScorer(
            ["rougeL"], use_stemmer=False
        )
    return _rouge_scorer_instance


@register_metric("rouge_l")
def compute_rouge_l(hypothesis: str, reference: str, **kwargs) -> float:
    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, hypothesis)
    return scores["rougeL"].fmeasure
