import pytest
from benchmark_indic_mt_eval.data.loader import MTSample
from benchmark_indic_mt_eval.evaluation.evaluator import (
    compute_metric_scores,
    evaluate_segment_level,
    evaluate_system_level,
    evaluate_language,
)


def _make_samples(n: int = 10) -> list[MTSample]:
    """Create synthetic samples with varying quality."""
    samples = []
    for i in range(n):
        if i < n // 2:
            hyp = f"good translation number {i}"
            ref = f"good translation number {i}"
            score = 0.9
        else:
            hyp = f"bad output {i} wrong"
            ref = f"correct translation number {i}"
            score = 0.3
        samples.append(
            MTSample(
                source=f"source sentence {i}",
                hypothesis=hyp,
                reference=ref,
                human_score=score,
                language="hi",
            )
        )
    return samples


class TestComputeMetricScores:
    def test_returns_list_of_floats(self):
        samples = _make_samples(5)
        scores = compute_metric_scores(samples, "bleu")
        assert len(scores) == 5
        assert all(isinstance(s, float) for s in scores)


class TestSegmentLevel:
    def test_returns_correlations(self):
        samples = _make_samples(20)
        result = evaluate_segment_level(samples, ["bleu", "chrf"])
        assert "bleu" in result
        assert "chrf" in result
        assert "pearson" in result["bleu"]
        assert "kendall_tau" in result["bleu"]
        assert "n" in result["bleu"]
        assert result["bleu"]["n"] == 20


class TestSystemLevel:
    def test_returns_correlations_over_systems(self):
        samples = _make_samples(10)
        result = evaluate_system_level(samples, ["bleu"])
        assert "bleu" in result
        assert "n" in result["bleu"]


class TestEvaluateLanguage:
    def test_returns_both_levels(self):
        samples = _make_samples(20)
        result = evaluate_language(samples, ["bleu"], levels=["segment", "system"])
        assert "segment_level" in result
        assert "system_level" in result
