"""Tests for neural metrics. Skipped if dependencies not installed."""

import pytest

try:
    import bert_score

    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

try:
    import comet

    HAS_COMET = True
except ImportError:
    HAS_COMET = False


@pytest.mark.skipif(not HAS_BERTSCORE, reason="bert-score not installed")
class TestBERTScore:
    def test_perfect_match(self):
        from benchmark_indic_mt_eval.metrics.embedding import compute_bertscore

        score = compute_bertscore("hello world", "hello world")
        assert score > 0.9

    def test_different_sentences(self):
        from benchmark_indic_mt_eval.metrics.embedding import compute_bertscore

        score = compute_bertscore("the cat sat", "purple elephants fly")
        assert score < 0.9


@pytest.mark.skipif(not HAS_COMET, reason="unbabel-comet not installed")
class TestCOMET:
    def test_comet_returns_float(self):
        from benchmark_indic_mt_eval.metrics.trained import compute_comet

        score = compute_comet(
            hypothesis="the cat sat",
            reference="the cat sat",
            source="the cat sat",
        )
        assert isinstance(score, float)
