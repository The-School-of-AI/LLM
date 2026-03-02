"""Tests for OpusConfig dataclass."""

from llm.opus.config import OpusConfig


def test_opus_config_defaults():
    """Verify all default values are set correctly."""
    cfg = OpusConfig()
    assert cfg.enabled is False
    assert cfg.selection_mode == "opus"
    assert cfg.candidate_multiplier == 2
    assert cfg.selection_ratio == 0.5
    assert cfg.score_seq_len == 512
    assert cfg.proxy_batch_size == 8
    assert cfg.sketch_dim == 8192
    assert cfg.temperature == 0.9
    assert cfg.sketch_seed == 42
    assert cfg.fallback_random_on_error is True
    assert cfg.max_selector_time_s == 30.0
    assert cfg.include_embeddings is False
    assert cfg.include_lm_head is False
    assert cfg.track_nonfinite_stats is True
    assert cfg.zero2_exact_global_scoring is True
    assert cfg.strict_shard_preconditioner is False
    assert cfg.log_selection_metrics is True


def test_opus_config_from_dict():
    """Verify from_dict works with a partial dictionary."""
    cfg = OpusConfig.from_dict({"enabled": True, "selection_mode": "random", "sketch_dim": 4096})
    assert cfg.enabled is True
    assert cfg.selection_mode == "random"
    assert cfg.sketch_dim == 4096
    # Other fields should retain defaults
    assert cfg.candidate_multiplier == 2
    assert cfg.temperature == 0.9


def test_opus_config_from_dict_ignores_unknown_keys():
    """Verify unknown keys in the dictionary are silently ignored."""
    cfg = OpusConfig.from_dict({
        "enabled": True,
        "unknown_key": "should_be_ignored",
        "another_unknown": 999,
    })
    assert cfg.enabled is True
    # Defaults still hold for unspecified fields
    assert cfg.selection_mode == "opus"
    assert not hasattr(cfg, "unknown_key")
    assert not hasattr(cfg, "another_unknown")
