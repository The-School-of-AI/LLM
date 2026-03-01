"""Embedding-based MT metrics: BERTScore."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from bert_score import score as bert_score_fn

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

from benchmark_indic_mt_eval.metrics.registry import register_metric


@register_metric("bertscore")
def compute_bertscore(hypothesis: str, reference: str, **kwargs) -> float:
    if not HAS_BERTSCORE:
        raise ImportError(
            "bert-score not installed. Install with: pip install bert-score"
        )
    P, R, F1 = bert_score_fn(
        [hypothesis],
        [reference],
        lang="hi",  # multilingual model handles all Indic langs
        verbose=False,
    )
    return F1[0].item()
