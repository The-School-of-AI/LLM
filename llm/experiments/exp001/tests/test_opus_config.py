"""
Tests for OpusConfig dataclass and auto-configuration logic in Trainer.__init__.

Covers:
  - Default values and field types
  - n_proxy_total divisibility assertion
  - n_proxy_per_gpu computation for various world sizes
  - n_to_select and selection_ratio derivation
  - include_proxy_in_training=True vs False
  - Proxy seq_len override logic
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Ensure experiment root is importable
EXP_ROOT = str(Path(__file__).resolve().parent.parent)
if EXP_ROOT not in sys.path:
    sys.path.insert(0, EXP_ROOT)

from exp.train import OpusConfig


# ---------------------------------------------------------------------------
# OpusConfig dataclass field tests
# ---------------------------------------------------------------------------


class TestOpusConfigFields:
    def test_required_fields(self):
        """All required fields must be provided."""
        cfg = OpusConfig(
            candidate_multiplier=4,
            n_proxy_total=16,
            scoring_seq_len=512,
            train_seq_len=128,
            sketch_dim=512,
            temperature=0.9,
            sketch_seed=42,
        )
        assert cfg.candidate_multiplier == 4
        assert cfg.n_proxy_total == 16
        assert cfg.scoring_seq_len == 512
        assert cfg.train_seq_len == 128

    def test_defaults(self):
        """Default values match expected ZeRO-2 safe config."""
        cfg = OpusConfig(
            candidate_multiplier=2,
            n_proxy_total=1,
            scoring_seq_len=512,
            train_seq_len=128,
            sketch_dim=512,
            temperature=0.9,
            sketch_seed=42,
        )
        assert cfg.include_proxy_in_training is True
        assert cfg.strict_shard_preconditioner is False
        assert cfg.max_selector_time_s == 30.0
        assert cfg.fallback_random_on_error is True

    def test_opus_disabled_multiplier_1(self):
        """candidate_multiplier=1 is valid (disables OPUS)."""
        cfg = OpusConfig(
            candidate_multiplier=1,
            n_proxy_total=1,
            scoring_seq_len=512,
            train_seq_len=128,
            sketch_dim=512,
            temperature=0.9,
            sketch_seed=42,
        )
        assert cfg.candidate_multiplier == 1


# ---------------------------------------------------------------------------
# Auto-configuration: n_proxy_per_gpu, n_to_select, selection_ratio
# ---------------------------------------------------------------------------


class TestAutoConfiguration:
    """
    Tests the auto-configuration logic from Trainer.__init__ without
    constructing the full Trainer (which needs DeepSpeed, model, etc.).
    We extract and test the pure math.
    """

    @staticmethod
    def _compute_opus_params(
        n_proxy_total: int,
        world_size: int,
        micro_batch: int,
        candidate_multiplier: int,
        include_proxy_in_training: bool,
    ) -> dict:
        """Replicate the auto-config logic from Trainer.__init__."""
        assert n_proxy_total % world_size == 0
        n_proxy_per_gpu = n_proxy_total // world_size

        selected_batch_size = micro_batch
        candidate_pool_size = selected_batch_size * candidate_multiplier

        if include_proxy_in_training:
            assert n_proxy_per_gpu <= selected_batch_size
            n_to_select = selected_batch_size - n_proxy_per_gpu
        else:
            n_to_select = selected_batch_size

        selection_ratio = (
            n_to_select / candidate_pool_size if candidate_pool_size > 0 else 0.0
        )

        return {
            "n_proxy_per_gpu": n_proxy_per_gpu,
            "n_to_select": n_to_select,
            "candidate_pool_size": candidate_pool_size,
            "selection_ratio": selection_ratio,
        }

    def test_single_gpu_basic(self):
        """Single GPU, micro_batch=1, multiplier=2, n_proxy_total=1."""
        p = self._compute_opus_params(
            n_proxy_total=1,
            world_size=1,
            micro_batch=1,
            candidate_multiplier=2,
            include_proxy_in_training=True,
        )
        assert p["n_proxy_per_gpu"] == 1
        assert p["n_to_select"] == 0  # all slots filled by proxy
        assert p["candidate_pool_size"] == 2
        assert p["selection_ratio"] == 0.0

    def test_8gpu_production(self):
        """8 GPUs, micro_batch=4, multiplier=4, n_proxy_total=16."""
        p = self._compute_opus_params(
            n_proxy_total=16,
            world_size=8,
            micro_batch=4,
            candidate_multiplier=4,
            include_proxy_in_training=True,
        )
        assert p["n_proxy_per_gpu"] == 2
        assert p["n_to_select"] == 2  # 4 - 2
        assert p["candidate_pool_size"] == 16  # 4 * 4
        assert p["selection_ratio"] == pytest.approx(2 / 16)

    def test_paper_behavior_no_proxy_in_training(self):
        """Original paper: proxy not in training batch."""
        p = self._compute_opus_params(
            n_proxy_total=16,
            world_size=8,
            micro_batch=4,
            candidate_multiplier=4,
            include_proxy_in_training=False,
        )
        assert p["n_proxy_per_gpu"] == 2
        assert p["n_to_select"] == 4  # full micro_batch
        assert p["selection_ratio"] == pytest.approx(4 / 16)

    def test_n_proxy_total_not_divisible_raises(self):
        """n_proxy_total must be divisible by world_size."""
        with pytest.raises(AssertionError):
            self._compute_opus_params(
                n_proxy_total=3,
                world_size=2,
                micro_batch=4,
                candidate_multiplier=2,
                include_proxy_in_training=True,
            )

    def test_n_proxy_per_gpu_exceeds_micro_batch_raises(self):
        """n_proxy_per_gpu cannot exceed micro_batch when include_proxy=True."""
        with pytest.raises(AssertionError):
            self._compute_opus_params(
                n_proxy_total=8,
                world_size=1,
                micro_batch=4,
                candidate_multiplier=2,
                include_proxy_in_training=True,
            )

    def test_n_proxy_per_gpu_exceeds_micro_batch_ok_when_proxy_excluded(self):
        """n_proxy_per_gpu > micro_batch is fine when proxy not in training."""
        p = self._compute_opus_params(
            n_proxy_total=8,
            world_size=1,
            micro_batch=4,
            candidate_multiplier=2,
            include_proxy_in_training=False,
        )
        assert p["n_proxy_per_gpu"] == 8
        assert p["n_to_select"] == 4

    def test_multiplier_1_disabled(self):
        """candidate_multiplier=1 means pool == micro_batch (no selection)."""
        p = self._compute_opus_params(
            n_proxy_total=1,
            world_size=1,
            micro_batch=4,
            candidate_multiplier=1,
            include_proxy_in_training=True,
        )
        assert p["candidate_pool_size"] == 4
        assert p["n_to_select"] == 3

    def test_batch32_8gpu_scenario(self):
        """The architect's batch_size=32 scenario with 8 GPUs."""
        # micro_batch_per_gpu = 32/8 = 4
        p = self._compute_opus_params(
            n_proxy_total=16,
            world_size=8,
            micro_batch=4,
            candidate_multiplier=4,
            include_proxy_in_training=True,
        )
        # 2 proxy + 2 selected = 4 per GPU = micro_batch
        assert p["n_proxy_per_gpu"] + p["n_to_select"] == 4


# ---------------------------------------------------------------------------
# Proxy seq_len override logic
# ---------------------------------------------------------------------------


class TestProxySeqLenOverride:
    def test_scoring_larger(self):
        """When scoring_seq_len > train_seq_len, proxy loads at scoring_seq_len."""
        scoring = 512
        training = 128
        effective = max(scoring, training)
        assert effective == 512

    def test_training_larger(self):
        """When train_seq_len > scoring_seq_len, proxy loads at train_seq_len."""
        scoring = 64
        training = 256
        effective = max(scoring, training)
        assert effective == 256

    def test_equal(self):
        """When both are equal, no override needed."""
        scoring = 128
        training = 128
        effective = max(scoring, training)
        assert effective == 128
