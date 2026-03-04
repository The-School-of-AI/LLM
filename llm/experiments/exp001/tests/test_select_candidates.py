"""
Tests for _select_candidates method — the OPUS selection pipeline.

Covers:
  - 5-tuple return type (local_indices, proxy_ids_for_training, result, lr, opus_timings)
  - opus_timings dict contains all required timing keys
  - proxy_ids_for_training shape matches expected (n_proxy_per_gpu, train_len)
  - Effective scoring length clamping
  - Scoring loss value captured in opus_timings
  - Combined scoring batch is [n_proxy + n_candidates, scoring_len]
"""

from __future__ import annotations

import sys
import time as _time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

from tests.fakes import (
    FakeDeepSpeedEngine,
    FakeFusedCE,
    FakeGhostCollector,
    FakeModel,
    FakeProxyProvider,
    FakeSelector,
)
from exp.opus import AdamWPreconditionerView, CountSketchProjector, SelectionResult
from exp.train import OpusConfig, Trainer


# ---------------------------------------------------------------------------
# Helpers to build a minimal Trainer without __init__
# ---------------------------------------------------------------------------


def _make_trainer_stub(
    n_proxy_per_gpu: int = 1,
    n_to_select: int = 1,
    scoring_seq_len: int = 64,
    train_seq_len: int = 32,
    candidate_multiplier: int = 2,
    include_proxy_in_training: bool = True,
    vocab_size: int = 128,
    hidden_size: int = 32,
) -> Trainer:
    """
    Construct a Trainer-like object with mocked internals,
    bypassing the full __init__ which requires DeepSpeed/CUDA.
    """
    # Bypass __init__
    trainer = object.__new__(Trainer)

    model = FakeModel(hidden_size=hidden_size, vocab_size=vocab_size)
    engine = FakeDeepSpeedEngine(model, device=torch.device("cpu"))

    trainer.engine = engine
    trainer.device = torch.device("cpu")
    trainer.n_proxy_per_gpu = n_proxy_per_gpu
    trainer.n_to_select = n_to_select
    trainer.initial_lr = 3e-4
    trainer.step_prof = None

    # OPUS config
    from exp.train import Config, DataConfig, TrainConfig
    from exp.proxy_dataset import ProxyDatasetConfig

    opus_cfg = OpusConfig(
        candidate_multiplier=candidate_multiplier,
        n_proxy_total=n_proxy_per_gpu,  # single GPU
        scoring_seq_len=scoring_seq_len,
        train_seq_len=train_seq_len,
        sketch_dim=64,
        temperature=0.9,
        sketch_seed=42,
        include_proxy_in_training=include_proxy_in_training,
    )
    trainer.config = Config(
        seed=42,
        deepspeed_config="dummy",
        tokenizer_dir=".",
        profiler_output_dir=".",
        data=DataConfig(),
        proxy=ProxyDatasetConfig(local_path=".", seq_len=64, batch_size=4),
        train=TrainConfig(max_steps=5),
        opus=opus_cfg,
        model=MagicMock(),
    )

    # Mock OPUS components
    trainer.proxy_provider = FakeProxyProvider(vocab_size=vocab_size)
    trainer.sketcher = MagicMock(spec=CountSketchProjector)
    trainer.preconditioner_view = MagicMock(spec=AdamWPreconditionerView)
    trainer.preconditioner_view.refresh = MagicMock()

    # FusedCE
    trainer._fused_ce = FakeFusedCE()

    # Selector — returns n_to_select indices
    trainer.selector = FakeSelector(n_to_select=n_to_select)

    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSelectCandidatesReturnType:
    def test_returns_5_tuple(self):
        """_select_candidates must return a 5-tuple."""
        trainer = _make_trainer_stub()
        candidate_ids = torch.randint(0, 128, (2, 64))

        # Patch OpusGhostCollector to use our fake
        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            result = trainer._select_candidates(candidate_ids)

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_return_types(self):
        """Each element of the 5-tuple has the correct type."""
        trainer = _make_trainer_stub()
        candidate_ids = torch.randint(0, 128, (2, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            local_indices, proxy_ids_train, result, lr, opus_timings = (
                trainer._select_candidates(candidate_ids)
            )

        assert isinstance(local_indices, torch.Tensor)
        assert isinstance(proxy_ids_train, torch.Tensor)
        assert isinstance(result, SelectionResult)
        assert isinstance(lr, float)
        assert isinstance(opus_timings, dict)


class TestOpusTimingsDict:
    def test_has_all_timing_keys(self):
        """opus_timings dict contains all per-phase timing keys."""
        trainer = _make_trainer_stub()
        candidate_ids = torch.randint(0, 128, (2, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, _, _, _, opus_timings = trainer._select_candidates(candidate_ids)

        required_timing_keys = [
            "preconditioner_refresh_ms",
            "proxy_sample_ms",
            "scoring_forward_ms",
            "scoring_backward_ms",
            "zero_grad_ms",
            "boltzmann_select_ms",
        ]
        for key in required_timing_keys:
            assert key in opus_timings, f"Missing timing key: {key}"
            assert isinstance(opus_timings[key], float)
            assert opus_timings[key] >= 0.0

    def test_has_metadata_keys(self):
        """opus_timings has scoring batch metadata."""
        trainer = _make_trainer_stub(scoring_seq_len=64)
        candidate_ids = torch.randint(0, 128, (2, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, _, _, _, opus_timings = trainer._select_candidates(candidate_ids)

        assert "scoring_seq_len" in opus_timings
        assert "combined_scoring_batch" in opus_timings
        assert "n_proxy_sampled" in opus_timings
        assert "n_candidates_scored" in opus_timings
        assert "scoring_loss_val" in opus_timings

    def test_scoring_loss_is_finite(self):
        """Scoring loss captured in opus_timings should be a finite number."""
        trainer = _make_trainer_stub()
        candidate_ids = torch.randint(0, 128, (2, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, _, _, _, opus_timings = trainer._select_candidates(candidate_ids)

        assert "scoring_loss_val" in opus_timings
        assert torch.isfinite(torch.tensor(opus_timings["scoring_loss_val"]))


class TestProxyIdsForTraining:
    def test_shape_matches_config(self):
        """proxy_ids_for_training shape is (n_proxy_per_gpu, train_len)."""
        n_proxy = 2
        train_seq_len = 32
        trainer = _make_trainer_stub(
            n_proxy_per_gpu=n_proxy, train_seq_len=train_seq_len
        )
        candidate_ids = torch.randint(0, 128, (4, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, proxy_ids_train, _, _, _ = trainer._select_candidates(candidate_ids)

        assert proxy_ids_train.shape[0] == n_proxy
        assert proxy_ids_train.shape[1] == min(train_seq_len, candidate_ids.size(1))

    def test_proxy_truncated_to_candidate_len_when_shorter(self):
        """If candidate seq_len < train_seq_len, proxy is truncated to candidate len."""
        trainer = _make_trainer_stub(train_seq_len=128)
        candidate_ids = torch.randint(0, 128, (4, 32))  # seq_len=32 < train_seq_len=128

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, proxy_ids_train, _, _, _ = trainer._select_candidates(candidate_ids)

        assert proxy_ids_train.shape[1] == 32  # min(128, 32) = 32


class TestEffectiveScoringLen:
    def test_clamped_to_min_of_all_sources(self):
        """effective_scoring_len = min(scoring_seq_len, proxy_len, candidate_len)."""
        trainer = _make_trainer_stub(scoring_seq_len=64)
        candidate_ids = torch.randint(0, 128, (4, 48))  # candidate_len=48 < scoring=64

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, _, _, _, opus_timings = trainer._select_candidates(candidate_ids)

        # Should be 48 (candidate_len is the bottleneck)
        assert opus_timings["scoring_seq_len"] == 48.0

    def test_combined_batch_size(self):
        """Combined scoring batch = n_proxy + n_candidates."""
        n_proxy = 1
        trainer = _make_trainer_stub(n_proxy_per_gpu=n_proxy)
        n_candidates = 4
        candidate_ids = torch.randint(0, 128, (n_candidates, 64))

        with patch(
            "exp.train.OpusGhostCollector",
            lambda **kwargs: FakeGhostCollector(
                n_candidates=kwargs["n_candidates"], sketch_dim=64
            ),
        ):
            _, _, _, _, opus_timings = trainer._select_candidates(candidate_ids)

        assert opus_timings["combined_scoring_batch"] == float(
            n_proxy + n_candidates
        )
        assert opus_timings["n_proxy_sampled"] == float(n_proxy)
        assert opus_timings["n_candidates_scored"] == float(n_candidates)
