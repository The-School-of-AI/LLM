import pytest
from benchmark_indic_mt_eval.evaluation.correlation import (
    compute_pearson,
    compute_kendall_tau,
    compute_correlations,
)


class TestPearson:
    def test_perfect_positive(self):
        r = compute_pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert r == pytest.approx(1.0, abs=0.001)

    def test_perfect_negative(self):
        r = compute_pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2])
        assert r == pytest.approx(-1.0, abs=0.001)

    def test_no_correlation(self):
        r = compute_pearson([1, 2, 3, 4, 5], [2, 4, 1, 5, 3])
        assert -0.5 < r < 0.5

    def test_constant_returns_zero(self):
        r = compute_pearson([1, 1, 1], [1, 2, 3])
        assert r == pytest.approx(0.0, abs=0.001)


class TestKendallTau:
    def test_perfect_concordance(self):
        tau = compute_kendall_tau([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert tau == pytest.approx(1.0, abs=0.001)

    def test_perfect_discordance(self):
        tau = compute_kendall_tau([1, 2, 3, 4, 5], [50, 40, 30, 20, 10])
        assert tau == pytest.approx(-1.0, abs=0.001)


class TestComputeCorrelations:
    def test_returns_both_metrics(self):
        result = compute_correlations([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert "pearson" in result
        assert "kendall_tau" in result
        assert result["pearson"] == pytest.approx(1.0, abs=0.001)
        assert result["kendall_tau"] == pytest.approx(1.0, abs=0.001)

    def test_too_few_samples(self):
        result = compute_correlations([1], [2])
        assert result["pearson"] == 0.0
        assert result["kendall_tau"] == 0.0
