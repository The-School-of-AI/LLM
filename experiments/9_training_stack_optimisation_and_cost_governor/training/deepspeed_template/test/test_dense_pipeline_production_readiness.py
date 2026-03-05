"""
Production readiness gates for the Dense training pipeline.

These tests are intentionally strict and are designed to fail until the
recommended production hardening changes are implemented.

They avoid importing training modules directly so they can run in lightweight
CI environments without DeepSpeed/CUDA.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")


def _load_yaml(rel_path: str) -> dict:
    with (PROJECT_ROOT / rel_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_dense_model_config_has_no_active_moe_experts():
    """Dense pipeline must keep MoE experts disabled."""
    src = _read("src/models/recurrence_model_1b.py")
    assert "num_real_experts = 0" in src
    assert "num_null_experts = 0" in src
    assert "top_k = 0" in src


def test_dataloader_uses_distributed_sampler():
    """
    Multi-GPU training must shard dataset per rank.
    """
    src = _read("src/data.py")
    assert "DistributedSampler" in src, (
        "Missing DistributedSampler import/usage. "
        "Without it, each rank processes the full dataset."
    )
    assert "sampler=" in src, (
        "DataLoader is expected to pass sampler=... "
        "for train/val/test in distributed runs."
    )


def test_sampler_epoch_is_set_each_epoch():
    """
    Deterministic distributed shuffling requires sampler.set_epoch(epoch).
    """
    main_src = _read("main.py")
    train_src = _read("src/train.py")
    assert ".set_epoch(" in (
        main_src + "\n" + train_src
    ), "Expected sampler.set_epoch(epoch) call not found."


def test_offline_dataset_loading_is_supported():
    """
    Runtime tokenization on expensive GPU nodes should be avoidable.
    """
    data_src = _read("src/data.py")
    cfg = _load_yaml("config.example.yaml")
    data_cfg = cfg.get("data", {})
    has_offline_key = any(
        k in data_cfg
        for k in ("offline_dataset_path", "tokenized_dataset_path", "dataset_cache_dir")
    )
    assert (
        "load_from_disk" in data_src
    ), "Expected datasets.load_from_disk support for offline tokenized datasets."
    assert (
        has_offline_key
    ), "config.example.yaml is expected to expose an offline dataset path option."


def test_no_per_step_cuda_empty_cache_in_train_loop():
    """
    empty_cache in the hot path hurts throughput and can increase latency.
    """
    train_src = _read("src/train.py")
    assert "torch.cuda.empty_cache()" not in train_src, (
        "Found torch.cuda.empty_cache() in src/train.py; "
        "remove from per-step loops and keep only controlled lifecycle cleanup."
    )


def test_h2d_transfers_use_non_blocking_when_pinned_memory():
    """
    If DataLoader uses pin_memory=True, device transfer should be non_blocking=True.
    """
    data_src = _read("src/data.py")
    train_src = _read("src/train.py")
    assert "pin_memory=True" in data_src, "Expected pin_memory=True in DataLoader."

    to_calls = re.findall(r"\.to\(model_engine\.device[^\)]*\)", train_src)
    assert to_calls, "Expected device transfer calls in src/train.py."
    assert any(
        "non_blocking=True" in call for call in to_calls
    ), "Expected non_blocking=True in .to(model_engine.device, ...)."


def test_precision_policy_validation_exists():
    """
    Mixed precision mode must be validated to avoid bf16/fp16 mismatch bugs.
    """
    main_src = _read("main.py")
    assert (
        "validate_precision_policy" in main_src
    ), "Expected explicit precision policy validator in main.py."
    assert (
        "raise ValueError" in main_src
    ), "Expected startup hard-fail for invalid precision combinations."


def test_required_kernel_fail_fast_exists():
    """
    Expensive runs should fail fast if required fused kernels are unavailable.
    """
    main_src = _read("main.py")
    cfg = _load_yaml("config.example.yaml")
    training_cfg = cfg.get("training", {})

    assert (
        "require_fused_kernels" in training_cfg
    ), "Expected training.require_fused_kernels in config.example.yaml."
    assert (
        "require_fused_kernels" in main_src
    ), "Expected main.py to read require_fused_kernels."
    assert (
        "HAS_TRITON" in main_src or "HAS_FLA" in main_src
    ), "Expected kernel availability checks in main.py."
    assert (
        "RuntimeError" in main_src
    ), "Expected hard failure when required kernels are missing."


def test_structured_logging_is_configured():
    """
    Console-only logs are not enough for expensive production training.
    """
    main_src = _read("main.py")
    train_src = _read("src/train.py")
    combined = main_src + "\n" + train_src
    assert (
        "jsonl" in combined
        or "wandb" in combined
        or "tensorboard" in combined
        or "SummaryWriter" in combined
    ), "Expected structured metrics logging (JSONL/W&B/TensorBoard)."


def test_dense_default_config_does_not_point_to_moe_profile():
    """
    Dense default config should not reference *-moe DeepSpeed profiles.
    """
    cfg = _load_yaml("config.yaml")
    ds_path = cfg.get("deepspeed", {}).get("config_path", "")
    assert (
        "moe" not in ds_path.lower()
    ), f"Dense default config should not use MoE profile, found: {ds_path}"


@pytest.mark.parametrize(
    "cfg_file",
    [
        "config.yaml",
        "config_reversible.yaml",
        "config_fix_oom.yaml",
        "config_profile.yaml",
    ],
)
def test_dense_configs_do_not_expose_model_type_switches(cfg_file: str):
    """
    Dense-only pipeline should keep config surface minimal and remove stale model_type.
    """
    cfg = _load_yaml(cfg_file)
    model_cfg = cfg.get("model", {})
    assert (
        "model_type" not in model_cfg
    ), f"{cfg_file} still has model_type. Remove stale Dense/MoE switch from dense pipeline configs."
