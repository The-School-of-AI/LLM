"""Tests for RejectionPolicyMetric in curriculum_tags."""

import tempfile
from pathlib import Path
import pytest
import yaml

# Import from the NEW location
from curriculum_tags.metrics.rejection_policy import RejectionPolicyMetric
from curriculum_tags.core.plugin import MetricPlugin
# We need to mock CurriculumConfig or use the loader if available in tests
# The original test used: from curriculum_extractor.utils.curriculum_loader import CurriculumConfig
# But curriculum_tags likely has its own config loader or uses the same one.
# s3_loader.py uses: from curriculum_tags import CurriculumTagger which uses ..utils.curriculum_loader.CurriculumConfig
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


class MockConfig:
    def __init__(self, data):
        self.data = data
        self.version = "0.1"
    
    def get(self, key, default=None):
        keys = key.split('.')
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default


def make_temp_curriculum(min_tokens=5):
    cfg = {
        "version": "0.1",
        "language_and_context": {
            "language_policy": {
                "primary_languages": [{"lang": "en"}],
                "secondary_languages": [{"lang": "indic"}], # Changed hi -> indic
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
    # New compute signature takes a dict
    sample = {"id": "1", "text": "This is valid text but lang fr", "language": "fr"}
    res = metric.compute(sample)
    
    assert res["rejected"] is True
    assert res["rejection_reason"] == "language_not_en_or_indic"

    cur_path.unlink()


def test_reject_below_min_tokens():
    # Set min tokens to 10 for test
    # Note: New logic is len(text.split()) / 4
    # So for 10 min tokens, we need < 10 tokens. 
    # To fail, token_count < 10. token_count = words / 4.
    # So words < 40.
    
    cur_path = make_temp_curriculum(min_tokens=10)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # "short text" = 2 words. 2/4 = 0.5 tokens. 0.5 < 10 -> Reject.
    sample = {"id": "2", "text": "short text", "language": "en"}
    res = metric.compute(sample)
    
    assert res["rejected"] is True
    assert "below_minimum_token_threshold" in res["rejection_reason"]

    cur_path.unlink()


def test_accepts_valid_sample():
    # Min tokens 2. We need words/4 >= 2 => words >= 8.
    cur_path = make_temp_curriculum(min_tokens=2)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    text = "word " * 10 # 10 words -> 2.5 tokens
    sample = {"id": "3", "text": text, "language": "en"}
    res = metric.compute(sample)
    
    assert res["rejected"] is False
    assert res.get("policy_checked") is True

    cur_path.unlink()


def test_language_from_metadata():
    cur_path = make_temp_curriculum(min_tokens=1)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # Missing top-level language, but present in metadata
    sample = {
        "id": "4", 
        "text": "valid text", 
        "language": None,
        "metadata": {"lang": "en"}
    }
    res = metric.compute(sample)
    
    assert res["rejected"] is False

    cur_path.unlink()


def test_language_detection_fallback():
    cur_path = make_temp_curriculum(min_tokens=1)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # Missing language everywhere, but text is clearly English
    sample = {
        "id": "5", 
        "text": "This is a clearly English sentence that should be detected.", 
        "language": None,
        "metadata": {}
    }
    res = metric.compute(sample)
    
    # Should detect 'en' and pass
    assert res["rejected"] is False

    cur_path.unlink()


def test_indic_detection_fallback():
    cur_path = make_temp_curriculum(min_tokens=1)
    config = CurriculumConfig(cur_path)
    metric = RejectionPolicyMetric(config)

    # Hindi text (Indic)
    # नमसते (Namaste)
    hindi_text = "नमस्ते दुनिया"
    sample = {
        "id": "6", 
        "text": hindi_text, 
        "language": None, 
        "metadata": {}
    }
    res = metric.compute(sample)
    
    # Should detect 'indic' and pass (since 'indic' is secondary allowed)
    assert res["rejected"] is False
    
    cur_path.unlink()
