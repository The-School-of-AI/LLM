"""
Shared fixtures for OPUS experiment tests.

Heavy modules are stubbed before experiment code is imported.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure the experiment package is importable
# ---------------------------------------------------------------------------
EXP_ROOT = str(
    __import__("pathlib").Path(__file__).resolve().parent.parent
)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)


# ---------------------------------------------------------------------------
# Stub out ALL heavy modules before importing experiment code.
# ---------------------------------------------------------------------------

def _ensure_module(name: str, attrs: dict | None = None) -> types.ModuleType:
    if name in sys.modules:
        mod = sys.modules[name]
        if attrs:
            for k, v in attrs.items():
                if not hasattr(mod, k):
                    setattr(mod, k, v)
        return mod
    mod = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(mod, k, v)
    sys.modules[name] = mod
    parts = name.rsplit(".", 1)
    if len(parts) == 2:
        parent = _ensure_module(parts[0])
        setattr(parent, parts[1], mod)
    return mod


# triton
_ensure_module("triton", {"jit": MagicMock(), "cdiv": lambda a, b: (a + b - 1) // b})
_ensure_module("triton.language")

# deepspeed
_ensure_module("deepspeed", {
    "DeepSpeedEngine": MagicMock,
    "initialize": MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock(), None)),
})

# transformers
_ensure_module("transformers")
_ensure_module("transformers.tokenization_utils_tokenizers", {
    "TokenizersBackend": MagicMock,
})

# llm package stubs
_ensure_module("llm")
_ensure_module("llm.kernels")
_ensure_module("llm.kernels.triton_cross_entropy", {
    "FusedLinearCrossEntropyLoss": MagicMock,
})
_ensure_module("llm.data", {
    "get_dataloaders": MagicMock(return_value=(MagicMock(), None, None, None)),
    "get_tokenizer": MagicMock(),
})
_ensure_module("llm.models", {
    "KroneckerConfig": MagicMock,
    "KroneckerEmbeddings": MagicMock,
    "Model1B": MagicMock,
    "ModelConfig": MagicMock,
})
_ensure_module("llm.profiler", {
    "PipelineProfiler": MagicMock,
    "StepProfiler": MagicMock,
})
_ensure_module("llm.utils", {
    "print_rank_0": lambda *a, **kw: None,
})


# ---------------------------------------------------------------------------
# Now safe to import experiment code
# ---------------------------------------------------------------------------

from exp.proxy_dataset import ProxyDatasetConfig
from exp.train import Config, DataConfig, OpusConfig, TrainConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def opus_config_enabled():
    return OpusConfig(
        candidate_multiplier=2,
        n_proxy_total=1,
        scoring_seq_len=64,
        train_seq_len=32,
        sketch_dim=64,
        temperature=0.9,
        sketch_seed=42,
        include_proxy_in_training=True,
        strict_shard_preconditioner=False,
        max_selector_time_s=5.0,
        fallback_random_on_error=True,
    )


@pytest.fixture
def opus_config_disabled():
    return OpusConfig(
        candidate_multiplier=1,
        n_proxy_total=1,
        scoring_seq_len=64,
        train_seq_len=32,
        sketch_dim=64,
        temperature=0.9,
        sketch_seed=42,
        include_proxy_in_training=True,
        strict_shard_preconditioner=False,
        max_selector_time_s=5.0,
        fallback_random_on_error=True,
    )


@pytest.fixture
def opus_config_no_proxy_training():
    return OpusConfig(
        candidate_multiplier=2,
        n_proxy_total=1,
        scoring_seq_len=64,
        train_seq_len=32,
        sketch_dim=64,
        temperature=0.9,
        sketch_seed=42,
        include_proxy_in_training=False,
        strict_shard_preconditioner=False,
        max_selector_time_s=5.0,
        fallback_random_on_error=True,
    )


@pytest.fixture
def base_config(opus_config_enabled, tmp_path):
    ds_cfg = tmp_path / "ds.yaml"
    ds_cfg.write_text(
        "train_batch_size: 1\n"
        "train_micro_batch_size_per_gpu: 1\n"
        "gradient_accumulation_steps: 1\n"
        "optimizer:\n"
        "  type: AdamW\n"
        "  params:\n"
        "    lr: 0.0003\n"
        "    betas: [0.9, 0.95]\n"
        "    eps: 1.0e-10\n"
        "    weight_decay: 0\n"
        "bf16:\n"
        "  enabled: false\n"
        "gradient_clipping: 1\n"
        "steps_per_print: 10\n"
    )
    return Config(
        seed=42,
        deepspeed_config=str(ds_cfg),
        tokenizer_dir=".",
        profiler_output_dir=str(tmp_path / "profiler"),
        data=DataConfig(max_length=32, dataset_name="dummy", num_workers=0),
        proxy=ProxyDatasetConfig(local_path=".", seq_len=64, batch_size=4, num_workers=0),
        train=TrainConfig(max_steps=3, log_interval=1),
        opus=opus_config_enabled,
        model=MagicMock(),
    )
