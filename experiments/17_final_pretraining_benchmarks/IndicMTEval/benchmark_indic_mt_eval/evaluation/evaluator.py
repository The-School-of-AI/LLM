"""Per-language evaluation orchestration."""

from __future__ import annotations

import logging
from collections import defaultdict

from benchmark_indic_mt_eval.data.loader import MTSample
from benchmark_indic_mt_eval.metrics.registry import get_metric
from benchmark_indic_mt_eval.evaluation.correlation import compute_correlations

logger = logging.getLogger(__name__)


def compute_metric_scores(
    samples: list[MTSample], metric_name: str
) -> list[float]:
    metric_fn = get_metric(metric_name)
    scores: list[float] = []
    for sample in samples:
        try:
            score = metric_fn(
                hypothesis=sample.hypothesis,
                reference=sample.reference,
                source=sample.source,
            )
        except Exception as e:
            logger.warning("Metric %s failed on sample: %s", metric_name, e)
            score = 0.0
        scores.append(score)
    return scores


def evaluate_segment_level(
    samples: list[MTSample], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    human_scores = [s.human_score for s in samples]
    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        logger.info("Computing segment-level %s for %d samples", name, len(samples))
        predicted = compute_metric_scores(samples, name)
        corr = compute_correlations(predicted, human_scores)
        corr["n"] = len(samples)
        results[name] = corr
    return results


def evaluate_system_level(
    samples: list[MTSample], metric_names: list[str]
) -> dict[str, dict[str, float]]:
    """Group samples by source sentence, average scores per group.

    In IndicMT-Eval, each source has 7 MT system translations. Samples
    sharing the same source text belong to different systems. We group by
    source, then treat each source group as a data point (average metric
    score vs average human score).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        groups[s.source].append(i)

    results: dict[str, dict[str, float]] = {}
    for name in metric_names:
        logger.info("Computing system-level %s", name)
        predicted = compute_metric_scores(samples, name)
        human_scores = [s.human_score for s in samples]

        avg_predicted: list[float] = []
        avg_human: list[float] = []
        for source, indices in groups.items():
            avg_predicted.append(sum(predicted[i] for i in indices) / len(indices))
            avg_human.append(sum(human_scores[i] for i in indices) / len(indices))

        corr = compute_correlations(avg_predicted, avg_human)
        corr["n"] = len(groups)
        results[name] = corr

    return results


def evaluate_language(
    samples: list[MTSample],
    metric_names: list[str],
    levels: list[str] | None = None,
) -> dict[str, dict]:
    if levels is None:
        levels = ["segment", "system"]

    result: dict[str, dict] = {}
    if "segment" in levels:
        result["segment_level"] = evaluate_segment_level(samples, metric_names)
    if "system" in levels:
        result["system_level"] = evaluate_system_level(samples, metric_names)
    return result
