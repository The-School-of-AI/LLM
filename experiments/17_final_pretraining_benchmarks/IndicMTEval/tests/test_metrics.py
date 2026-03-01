import pytest
from benchmark_indic_mt_eval.metrics.registry import get_metric, list_metrics
from benchmark_indic_mt_eval.metrics.overlap import (
    compute_bleu,
    compute_chrf,
    compute_ter,
    compute_rouge_l,
)


class TestRegistry:
    def test_list_metrics_includes_overlap(self):
        names = list_metrics()
        assert "bleu" in names
        assert "chrf" in names
        assert "ter" in names

    def test_get_metric_returns_callable(self):
        fn = get_metric("bleu")
        assert callable(fn)

    def test_get_metric_unknown_raises(self):
        with pytest.raises(KeyError):
            get_metric("nonexistent_metric")


class TestBLEU:
    def test_perfect_match(self):
        score = compute_bleu("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_no_match(self):
        score = compute_bleu("aaa bbb ccc ddd", "xxx yyy zzz www")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_partial_match(self):
        score = compute_bleu("the cat sat", "the cat lay")
        assert 0.0 < score < 1.0


class TestChrF:
    def test_perfect_match(self):
        score = compute_chrf("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = compute_chrf("the cat sat", "the cat lay")
        assert 0.0 < score < 1.0


class TestTER:
    def test_perfect_match(self):
        score = compute_ter("hello world", "hello world")
        assert score == pytest.approx(0.0, abs=0.01)

    def test_different(self):
        score = compute_ter("hello world", "goodbye earth")
        assert score > 0.0


class TestROUGEL:
    def test_perfect_match(self):
        score = compute_rouge_l("hello world", "hello world")
        assert score == pytest.approx(1.0, abs=0.01)

    def test_partial_match(self):
        score = compute_rouge_l("the cat sat on mat", "the cat lay on mat")
        assert 0.0 < score < 1.0
