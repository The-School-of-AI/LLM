"""Correlation metrics: Pearson and Kendall-tau."""

from __future__ import annotations

import logging
from scipy import stats

logger = logging.getLogger(__name__)

MIN_SAMPLES = 3


def compute_pearson(predictions: list[float], human_scores: list[float]) -> float:
    if len(predictions) < MIN_SAMPLES:
        return 0.0
    if len(set(predictions)) <= 1 or len(set(human_scores)) <= 1:
        return 0.0
    r, _ = stats.pearsonr(predictions, human_scores)
    return float(r)


def compute_kendall_tau(
    predictions: list[float], human_scores: list[float]
) -> float:
    if len(predictions) < MIN_SAMPLES:
        return 0.0
    tau, _ = stats.kendalltau(predictions, human_scores)
    return float(tau)


def compute_correlations(
    predictions: list[float], human_scores: list[float]
) -> dict[str, float]:
    return {
        "pearson": compute_pearson(predictions, human_scores),
        "kendall_tau": compute_kendall_tau(predictions, human_scores),
    }
