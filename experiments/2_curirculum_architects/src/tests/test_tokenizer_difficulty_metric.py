import pytest
from curriculum_tags.metrics.tokenizer_difficulty import TokenizerDifficultyMetric


class DummyTokenizer:
    """Simple tokenizer stub that returns a fixed list of token ids."""

    def __init__(self, tokens):
        self._tokens = tokens

    def encode(self, text, add_special_tokens: bool = False):
        return self._tokens


@pytest.fixture
def metric():
    # TokenizerDifficultyMetric only uses config.get for tokenizer settings,
    # so a plain dict is sufficient for tests.
    config = {}
    return TokenizerDifficultyMetric(config)


def test_empty_text_uses_empty_result(metric):
    """Empty text should return the default empty result structure."""
    result = metric.compute({"text": ""})

    assert result["level"] == "T0"
    assert result["score"] == 0.0
    assert result["features"]["token_count"] == 0
    assert result["features"]["level_reason"] == "Empty text or no tokens"


def test_no_tokens_returns_empty_result(metric):
    """If tokenizer produces no tokens, we should also get the empty result."""
    metric.tokenizer = DummyTokenizer([])

    result = metric.compute({"text": "some text"})

    assert result["level"] == "T0"
    assert result["score"] == 0.0
    assert result["features"]["token_count"] == 0
    assert result["features"]["level_reason"] == "Empty text or no tokens"


def test_compute_assigns_T0_for_low_token_ids(metric):
    """Low token ids should map into the simplest band T0."""
    # All ids are very small compared to T0 thresholds.
    tokens = [10, 20, 30, 40, 50]
    metric.tokenizer = DummyTokenizer(tokens)

    result = metric.compute({"text": "simple text"})

    assert result["level"] == "T0"
    assert result["score"] == 0.0
    assert result["features"]["token_count"] == len(tokens)
    # Sanity check on stats: max should match our synthetic tokens.
    assert result["features"]["max_token_id"] == max(tokens)


def test_assign_level_T5_for_very_large_ids(metric):
    """Very large token ids should fall into the highest difficulty band T5."""
    token_stats = {
        "avg_token_id": 200_000.0,
        "max_token_id": 300_000,
        "min_token_id": 150_000,
        "p50_token_id": 200_000.0,
        "p95_token_id": 250_000.0,
        "p99_token_id": 290_000.0,
        "token_count": 10,
    }

    level, meta = metric._assign_level(token_stats)

    assert level == "T5"
    assert meta["level"] == "T5"
    assert "Exceeded all thresholds" not in meta.get("reason", "")
