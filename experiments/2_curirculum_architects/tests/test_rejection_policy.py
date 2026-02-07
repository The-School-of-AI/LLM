"""Tests for RejectionPolicyMetric and its integration in curriculum_tags."""

import tempfile
from pathlib import Path

import pytest
import yaml
from curriculum_tags.core.plugin import MetricPlugin
from curriculum_tags.core.tagger import CurriculumTagger
from curriculum_tags.metrics.rejection_policy import RejectionPolicyMetric
from curriculum_tags.utils.curriculum_loader import CurriculumConfig

# --- Helper Classes & Fixtures ---


class MockConfig:
    def __init__(self, data):
        self.data = data
        self.version = "0.1"

    def get(self, key, default=None):
        keys = key.split(".")
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default


class MockRejectionMetric(MetricPlugin):
    name = "rejection_policy"

    def __init__(self, config, reject=False):
        super().__init__(config)
        self.reject = reject

    def compute(self, sample):
        return {"rejected": self.reject}


class MockOtherMetric(MetricPlugin):
    name = "other_metric"

    def compute(self, sample):
        return {"processed": True}


def make_temp_curriculum(min_tokens=5):
    """Create a temporary curriculum config file."""
    cfg = {
        "version": "0.1",
        "language_and_context": {
            "language_policy": {
                "primary_languages": [{"lang": "en"}],
                "secondary_languages": [{"lang": "indic"}],
            },
            "context_policy": {"min_context_tokens": min_tokens},
        },
        "difficulty_bands": {"bands": [{"id": "B0", "name": "Nursery"}]},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = Path(f.name)
    return path


@pytest.fixture
def temp_curriculum():
    """Fixture to provide a standard temporary curriculum config path."""
    path = make_temp_curriculum()
    yield path
    if path.exists():
        path.unlink()


# --- Unit Tests for RejectionPolicyMetric ---


def test_reject_non_allowed_language():
    cur_path = make_temp_curriculum(min_tokens=1)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        # Language 'fr' is not allowed
        sample = {"id": "1", "text": "This is valid text but lang fr", "language": "fr"}
        res = metric.compute(sample)

        assert res["rejected"] is True
        assert res["rejection_reason"] == "language_not_en_or_indic"
    finally:
        cur_path.unlink()


def test_reject_below_min_tokens():
    # Set min tokens to 10. Logic is approx len(text.split()) / 2 (checked impl).
    # Wait, previous test said /4 but implementation says /2.
    # checking implementation: min_tokens_int = int(min_tokens) / 2
    # if token_count < min_tokens_int: reject.
    # token_count is len(text.split()).
    # So if min_tokens=10, threshold is 5.
    # "short text" = 2 words. 2 < 5 -> Reject.

    cur_path = make_temp_curriculum(min_tokens=10)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        sample = {"id": "2", "text": "short text", "language": "en"}
        res = metric.compute(sample)

        assert res["rejected"] is True
        assert "below_minimum_token_threshold" in res["rejection_reason"]
    finally:
        cur_path.unlink()


def test_accepts_valid_sample():
    # Min tokens 2. Threshold 1.
    cur_path = make_temp_curriculum(min_tokens=2)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        text = "word " * 5  # 5 words -> 5 tokens. 5 > 1.
        sample = {"id": "3", "text": text, "language": "en"}
        res = metric.compute(sample)

        assert res["rejected"] is False
        assert res.get("policy_checked") is True
    finally:
        cur_path.unlink()


def test_language_from_metadata():
    cur_path = make_temp_curriculum(min_tokens=1)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        # Missing top-level language, but present in metadata
        sample = {
            "id": "4",
            "text": "valid text",
            "language": None,
            "metadata": {"lang": "en"},
        }
        res = metric.compute(sample)

        assert res["rejected"] is False
    finally:
        cur_path.unlink()


def test_language_detection_fallback():
    cur_path = make_temp_curriculum(min_tokens=1)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        # Missing language everywhere, but text is clearly English
        sample = {
            "id": "5",
            "text": "This is a clearly English sentence that should be detected.",
            "language": None,
            "metadata": {},
        }
        res = metric.compute(sample)

        # Should detect 'en' and pass
        assert res["rejected"] is False
    finally:
        cur_path.unlink()


def test_indic_detection_fallback():
    cur_path = make_temp_curriculum(min_tokens=1)
    try:
        config = CurriculumConfig(cur_path)
        metric = RejectionPolicyMetric(config)

        # Hindi text (Indic)
        hindi_text = "नमस्ते दुनिया"
        sample = {"id": "6", "text": hindi_text, "language": None, "metadata": {}}
        res = metric.compute(sample)

        assert res["rejected"] is False
    finally:
        cur_path.unlink()


# --- Integration Tests for Short-Circuit Logic ---


def test_rejection_short_circuit(temp_curriculum):
    """Test that metrics after rejection_policy are skipped if rejected."""
    config = CurriculumConfig(temp_curriculum)

    # Case 1: Rejected
    rejection_metric = MockRejectionMetric(config, reject=True)
    other_metric = MockOtherMetric(config)

    tagger = CurriculumTagger(temp_curriculum, metrics=[rejection_metric, other_metric])

    sample = {"id": "1", "text": "test"}
    tagged = tagger.tag_sample(sample)

    assert "rejection_policy" in tagged["curriculum_tags"]
    assert tagged["curriculum_tags"]["rejection_policy"]["rejected"] is True
    # Verify strict short-circuit: other_metric should NOT be present
    assert "other_metric" not in tagged["curriculum_tags"]


def test_no_rejection_continuation(temp_curriculum):
    """Test that metrics continue if not rejected."""
    config = CurriculumConfig(temp_curriculum)

    # Case 2: Not Rejected
    rejection_metric = MockRejectionMetric(config, reject=False)
    other_metric = MockOtherMetric(config)

    tagger = CurriculumTagger(temp_curriculum, metrics=[rejection_metric, other_metric])

    sample = {"id": "1", "text": "test"}
    tagged = tagger.tag_sample(sample)

    assert "rejection_policy" in tagged["curriculum_tags"]
    assert tagged["curriculum_tags"]["rejection_policy"]["rejected"] is False
    # Verify continuation: other_metric SHOULD be present
    assert "other_metric" in tagged["curriculum_tags"]
    assert tagged["curriculum_tags"]["other_metric"]["processed"] is True
