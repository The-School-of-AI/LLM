import pytest
from curriculum_tags.metrics.diversity import DiversityMetric


@pytest.fixture
def metric():
    config = {}
    return DiversityMetric(config)


def test_empty_tokens(metric):
    result = metric.compute({"text": ""})
    assert result["rare_ratio"] == 0.0
    assert result["token_count"] == 0


def test_all_unique(metric):
    # "a b c" -> all unique -> ratio 1.0
    result = metric.compute({"text": "a b c"})
    assert result["rare_ratio"] == 1.0
    assert result["token_count"] == 3


def test_all_repeated(metric):
    # "a a b b" -> no unique -> ratio 0.0
    result = metric.compute({"text": "a a b b"})
    assert result["rare_ratio"] == 0.0
    assert result["token_count"] == 4


def test_mixed(metric):
    # "a b a" -> 'a' is repeated, 'b' is unique
    # tokens: [a, b, a] (len 3)
    # unique: [b] (count 1)
    # ratio: 1/3 ≈ 0.333
    result = metric.compute({"text": "a b a"})
    assert result["rare_ratio"] == pytest.approx(0.333, abs=0.001)
