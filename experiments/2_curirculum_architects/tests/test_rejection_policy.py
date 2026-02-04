"""Tests for RejectionPolicyMetric."""

import tempfile
from pathlib import Path

import yaml

from curriculum_extractor.metrics.rejection_policy import RejectionPolicyMetric
from curriculum_extractor.utils.curriculum_loader import CurriculumConfig
from curriculum_extractor.core.plugin import ReadOnlyRecord


def make_temp_curriculum(min_tokens=5):
    cfg = {
        "version": "0.1",
        "language_and_context": {
            "language_policy": {
                "primary_languages": [{"lang": "en"}],
                "secondary_languages": [{"lang": "hi"}],
            },
            "context_policy": {"min_context_tokens": min_tokens},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = Path(f.name)
    return path


def test_reject_non_allowed_language():
    cur_path = make_temp_curriculum(min_tokens=1)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # Language 'fr' is not allowed
    rec = ReadOnlyRecord({"id": "1", "text": "This is valid text but lang fr", "language": "fr"})
    res = metric.extract(rec)
    assert res.rejected is True
    assert res.rejection_reason == "language_not_en_or_indic"

    cur_path.unlink()


def test_reject_below_min_tokens():
    # Set min tokens to 10 for test
    cur_path = make_temp_curriculum(min_tokens=10)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # Language allowed but token count too low
    rec = ReadOnlyRecord({"id": "2", "text": "short text", "language": "en"})
    res = metric.extract(rec)
    assert res.rejected is True
    assert res.rejection_reason == "below_minimum_token_threshold"

    cur_path.unlink()


def test_accepts_valid_sample():
    cur_path = make_temp_curriculum(min_tokens=2)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    rec = ReadOnlyRecord({"id": "3", "text": "this is enough", "language": "en"})
    res = metric.extract(rec)
    assert res.rejected is False
    assert res.metrics.get("policy_checked") is True

    cur_path.unlink()
