import pytest
from curriculum_tags.metrics.entropy import EntropyMetric


@pytest.fixture
def metric():
    # Config not used by entropy metric
    config = {}
    return EntropyMetric(config)


def test_empty_text(metric):
    result = metric.compute({"text": ""})
    assert result["score"] == 0.0


def test_simple_entropy(metric):
    # "aaaa" -> 1 symbol, prob 1.0 -> log(1) = 0 -> entropy 0
    result = metric.compute({"text": "aaaa"})
    assert result["score"] == 0.0


def test_high_entropy(metric):
    # "abcde" -> 5 symbols, prob 0.2 each
    # -5 * (0.2 * log(0.2)) = -log(0.2) = log(5) ≈ 1.609
    result = metric.compute({"text": "abcde"})
    assert result["score"] > 1.5
