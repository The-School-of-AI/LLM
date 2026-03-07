import pytest

from llm.deepspeed_config import apply_runtime_overrides


def test_apply_runtime_overrides_preserves_existing_value_when_unset():
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {"stage": 0, "overlap_comm": True},
    }

    resolved = apply_runtime_overrides(ds_config, None, None)

    assert resolved["zero_optimization"]["overlap_comm"] is True
    assert ds_config["zero_optimization"]["overlap_comm"] is True


def test_apply_runtime_overrides_sets_overlap_comm_when_enabled_or_disabled():
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {"stage": 0, "overlap_comm": True},
    }

    enabled = apply_runtime_overrides(ds_config, True, None)
    disabled = apply_runtime_overrides(ds_config, False, None)

    assert enabled["zero_optimization"]["overlap_comm"] is True
    assert disabled["zero_optimization"]["overlap_comm"] is False


def test_apply_runtime_overrides_sets_reduce_bucket_size_when_present():
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {"stage": 0, "reduce_bucket_size": 50_000_000},
    }

    resolved = apply_runtime_overrides(ds_config, None, 100_000_000)

    assert resolved["zero_optimization"]["reduce_bucket_size"] == 100_000_000
    assert ds_config["zero_optimization"]["reduce_bucket_size"] == 50_000_000


def test_apply_runtime_overrides_errors_when_zero_optimization_missing():
    ds_config = {"train_micro_batch_size_per_gpu": 1}

    with pytest.raises(
        ValueError, match="requires a DeepSpeed 'zero_optimization' config block"
    ):
        apply_runtime_overrides(ds_config, False, None)

    with pytest.raises(
        ValueError, match="requires a DeepSpeed 'zero_optimization' config block"
    ):
        apply_runtime_overrides(ds_config, None, 100_000_000)

    assert "zero_optimization" not in ds_config


def test_apply_runtime_overrides_errors_when_zero_optimization_is_not_mapping():
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": False,
    }

    with pytest.raises(ValueError, match="'zero_optimization' must be a mapping"):
        apply_runtime_overrides(ds_config, False, None)

    with pytest.raises(ValueError, match="'zero_optimization' must be a mapping"):
        apply_runtime_overrides(ds_config, None, 100_000_000)
