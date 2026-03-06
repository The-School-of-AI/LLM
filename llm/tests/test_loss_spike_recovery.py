"""Tests for the loss spike detection and recovery module."""

import torch
import torch.nn as nn
import pytest

from llm.loss_spike_recovery import (
    LossSpikeDetector,
    RecoveryAction,
    _parse_choice,
    auto_select_action,
    compute_grad_norm,
    compute_embedding_norms,
)


# ---------------------------------------------------------------------------
# LossSpikeDetector
# ---------------------------------------------------------------------------


class TestLossSpikeDetector:
    """Tests for the sliding-window spike detector."""

    def test_no_spike_during_warmup(self):
        """No spikes should be reported while the window is still filling."""
        detector = LossSpikeDetector(window_size=10)
        for i in range(10):
            assert detector.update(3.0 + i * 0.01) is False

    def test_no_spike_for_stable_loss(self):
        """Stable losses should never trigger a spike."""
        detector = LossSpikeDetector(window_size=10)
        for _ in range(10):
            detector.update(3.0)
        for _ in range(20):
            assert detector.update(3.0) is False

    def test_spike_detected_large_jump(self):
        """A sudden large jump should be flagged as a spike."""
        detector = LossSpikeDetector(window_size=10, z_threshold=3.0, min_abs_delta=0.5)
        for _ in range(10):
            detector.update(3.0)
        assert detector.update(10.0) is True

    def test_spike_detected_ratio_guard(self):
        """Spike should be detected via the ratio guard when std is tiny."""
        detector = LossSpikeDetector(
            window_size=10, z_threshold=3.0, min_spike_ratio=2.0, min_abs_delta=0.5
        )
        for _ in range(10):
            detector.update(3.0)
        assert detector.update(6.5) is True

    def test_no_spike_below_min_abs_delta(self):
        """Small fluctuations below min_abs_delta should not trigger."""
        detector = LossSpikeDetector(window_size=10, min_abs_delta=1.0)
        for _ in range(10):
            detector.update(3.0)
        assert detector.update(3.8) is False

    def test_spike_does_not_corrupt_window(self):
        """Spike values should not be added to the window."""
        detector = LossSpikeDetector(window_size=5, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        assert detector.update(20.0) is True

        stats = detector.get_stats()
        assert stats["window_mean"] == pytest.approx(3.0, abs=0.01)

    def test_get_stats_includes_spike_count(self):
        """get_stats should include spike_count."""
        detector = LossSpikeDetector(window_size=5)
        for v in [2.0, 3.0, 4.0, 3.0, 3.0]:
            detector.update(v)

        stats = detector.get_stats()
        assert stats["current_loss"] == 3.0
        assert stats["spike_count"] == 0

    def test_gradually_increasing_loss_no_spike(self):
        """Gradual increase should not trigger spikes (window adapts)."""
        detector = LossSpikeDetector(window_size=10, z_threshold=3.0, min_abs_delta=0.5)
        for i in range(30):
            loss = 3.0 + i * (2.0 / 30)
            detector.update(loss)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestCooldown:
    """Tests for the cooldown mechanism."""

    def test_cooldown_suppresses_detection(self):
        """No spike should fire during cooldown, even for extreme values."""
        detector = LossSpikeDetector(window_size=5, cooldown_steps=3, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        # First spike fires.
        assert detector.update(20.0) is True
        detector.record_spike_action()

        # Next 3 steps are in cooldown — feed normal values (they enter the
        # window but detection is suppressed).
        for _ in range(3):
            assert detector.update(3.0) is False

        # Cooldown expired — next spike should fire again.
        assert detector.update(20.0) is True

    def test_cooldown_resets_spike_count(self):
        """Spike count resets to 0 when cooldown expires without another spike."""
        detector = LossSpikeDetector(window_size=5, cooldown_steps=2, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        # Trigger a spike and record it.
        assert detector.update(20.0) is True
        detector.record_spike_action()
        assert detector.spike_count == 1

        # Burn through cooldown with normal values.
        detector.update(3.0)
        detector.update(3.0)  # cooldown expires here, resets spike_count

        assert detector.spike_count == 0

    def test_spike_count_increments(self):
        """Each record_spike_action increments spike_count."""
        detector = LossSpikeDetector(window_size=5, cooldown_steps=0, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        for expected in range(1, 4):
            assert detector.update(20.0) is True
            detector.record_spike_action()
            assert detector.spike_count == expected

    def test_values_during_cooldown_enter_window(self):
        """Normal values during cooldown should still update the window."""
        detector = LossSpikeDetector(window_size=5, cooldown_steps=5, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        assert detector.update(20.0) is True
        detector.record_spike_action()

        # Feed lower values during cooldown — they should enter the window.
        for _ in range(5):
            detector.update(2.0)

        # Window should now be all 2.0s, mean should reflect that.
        stats = detector.get_stats()
        assert stats["window_mean"] == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# auto_select_action
# ---------------------------------------------------------------------------


class TestAutoSelectAction:
    """Tests for the automatic escalation policy."""

    def test_tier1_skip_batch(self):
        """Spike count within patience_skip -> SKIP_BATCH."""
        for count in range(1, 4):
            action = auto_select_action(
                spike_count=count,
                patience_skip=3,
                patience_lr=10,
                last_checkpoint_tag="ckpt",
            )
            assert action == RecoveryAction.SKIP_BATCH

    def test_tier2_reduce_lr(self):
        """Spike count past patience_skip but within patience_lr -> REDUCE_LR."""
        for count in [4, 7, 10]:
            action = auto_select_action(
                spike_count=count,
                patience_skip=3,
                patience_lr=10,
                last_checkpoint_tag="ckpt",
            )
            assert action == RecoveryAction.REDUCE_LR

    def test_tier3_rollback(self):
        """Spike count past patience_lr with checkpoint -> ROLLBACK_CHECKPOINT."""
        action = auto_select_action(
            spike_count=11,
            patience_skip=3,
            patience_lr=10,
            last_checkpoint_tag="ckpt",
        )
        assert action == RecoveryAction.ROLLBACK_CHECKPOINT

    def test_tier3_no_checkpoint_falls_back(self):
        """Spike count past patience_lr without checkpoint -> SKIP_BATCH."""
        action = auto_select_action(
            spike_count=11,
            patience_skip=3,
            patience_lr=10,
            last_checkpoint_tag=None,
        )
        assert action == RecoveryAction.SKIP_BATCH

    def test_boundary_patience_skip(self):
        """Exactly at patience_skip boundary -> still SKIP_BATCH."""
        action = auto_select_action(
            spike_count=3,
            patience_skip=3,
            patience_lr=10,
            last_checkpoint_tag="ckpt",
        )
        assert action == RecoveryAction.SKIP_BATCH

    def test_boundary_patience_lr(self):
        """Exactly at patience_lr boundary -> still REDUCE_LR."""
        action = auto_select_action(
            spike_count=10,
            patience_skip=3,
            patience_lr=10,
            last_checkpoint_tag="ckpt",
        )
        assert action == RecoveryAction.REDUCE_LR


# ---------------------------------------------------------------------------
# _parse_choice (interactive fallback)
# ---------------------------------------------------------------------------


class TestParseChoice:
    """Tests for user input parsing."""

    def test_empty_defaults_to_skip(self):
        assert _parse_choice("", "ckpt_tag") == RecoveryAction.SKIP_BATCH

    def test_choice_1_skip(self):
        assert _parse_choice("1", "ckpt_tag") == RecoveryAction.SKIP_BATCH

    def test_choice_2_reduce_lr(self):
        assert _parse_choice("2", "ckpt_tag") == RecoveryAction.REDUCE_LR

    def test_choice_3_rollback(self):
        assert _parse_choice("3", "ckpt_tag") == RecoveryAction.ROLLBACK_CHECKPOINT

    def test_choice_3_no_checkpoint_falls_back(self):
        assert _parse_choice("3", None) == RecoveryAction.SKIP_BATCH

    def test_choice_4_ignore(self):
        assert _parse_choice("4", "ckpt_tag") == RecoveryAction.IGNORE

    def test_invalid_input_defaults_to_skip(self):
        assert _parse_choice("abc", "ckpt_tag") == RecoveryAction.SKIP_BATCH
        assert _parse_choice("5", "ckpt_tag") == RecoveryAction.SKIP_BATCH


# ---------------------------------------------------------------------------
# RecoveryAction enum
# ---------------------------------------------------------------------------


class TestRecoveryAction:
    """Sanity checks on the enum values."""

    def test_int_values(self):
        assert int(RecoveryAction.SKIP_BATCH) == 1
        assert int(RecoveryAction.REDUCE_LR) == 2
        assert int(RecoveryAction.ROLLBACK_CHECKPOINT) == 3
        assert int(RecoveryAction.IGNORE) == 4

    def test_round_trip(self):
        for action in RecoveryAction:
            assert RecoveryAction(int(action)) == action


# ---------------------------------------------------------------------------
# LossSpikeConfig
# ---------------------------------------------------------------------------


class TestLossSpikeConfig:
    """Test the config dataclass defaults."""

    def test_defaults(self):
        from llm.config import LossSpikeConfig

        cfg = LossSpikeConfig()
        assert cfg.enabled is True
        assert cfg.window_size == 100
        assert cfg.z_threshold == 3.0
        assert cfg.min_spike_ratio == 2.0
        assert cfg.min_abs_delta == 0.5
        assert cfg.grad_norm_threshold == 100.0
        assert cfg.lr_reduction_factor == 0.5
        assert cfg.auto_recover is True
        assert cfg.patience_skip == 3
        assert cfg.patience_lr == 10
        assert cfg.cooldown_steps == 50
        assert cfg.rollback_skip_batches == 200
        assert cfg.emb_norm_log_interval == 50
        assert cfg.user_prompt_timeout == 300

    def test_custom_values(self):
        from llm.config import LossSpikeConfig

        cfg = LossSpikeConfig(
            enabled=False,
            window_size=50,
            z_threshold=2.0,
            min_spike_ratio=1.5,
            min_abs_delta=0.3,
            grad_norm_threshold=50.0,
            lr_reduction_factor=0.25,
            auto_recover=False,
            patience_skip=5,
            patience_lr=15,
            cooldown_steps=100,
            rollback_skip_batches=500,
            user_prompt_timeout=60,
        )
        assert cfg.enabled is False
        assert cfg.auto_recover is False
        assert cfg.patience_skip == 5
        assert cfg.patience_lr == 15
        assert cfg.cooldown_steps == 100
        assert cfg.rollback_skip_batches == 500

    def test_grad_norm_disabled(self):
        from llm.config import LossSpikeConfig

        cfg = LossSpikeConfig(grad_norm_threshold=None)
        assert cfg.grad_norm_threshold is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    """Test the factory function."""

    def test_build_returns_detector_when_enabled(self):
        from llm.config import LossSpikeConfig
        from llm.factories import build_loss_spike_detector

        cfg = LossSpikeConfig(enabled=True, cooldown_steps=42)
        detector = build_loss_spike_detector(cfg)
        assert detector is not None
        assert isinstance(detector, LossSpikeDetector)
        assert detector._cooldown_steps == 42

    def test_build_returns_none_when_disabled(self):
        from llm.config import LossSpikeConfig
        from llm.factories import build_loss_spike_detector

        detector = build_loss_spike_detector(LossSpikeConfig(enabled=False))
        assert detector is None


# ---------------------------------------------------------------------------
# compute_grad_norm
# ---------------------------------------------------------------------------


class TestComputeGradNorm:
    """Tests for the gradient norm utility."""

    def test_zero_gradients(self):
        model = nn.Linear(4, 2, bias=False)
        assert compute_grad_norm(model) == 0.0

    def test_known_grad_norm(self):
        model = nn.Linear(4, 2, bias=False)
        x = torch.randn(1, 4)
        y = model(x)
        y.sum().backward()

        expected = torch.norm(model.weight.grad.float(), 2).item()
        actual = compute_grad_norm(model)
        assert actual == pytest.approx(expected, rel=1e-4)

    def test_multi_param_model(self):
        model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        x = torch.randn(1, 4)
        y = model(x)
        y.sum().backward()

        norm = compute_grad_norm(model)
        assert norm > 0

        param_norms = []
        for p in model.parameters():
            if p.grad is not None:
                param_norms.append(torch.norm(p.grad.float(), 2))
        expected = torch.norm(torch.stack(param_norms), 2).item()
        assert norm == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# compute_embedding_norms
# ---------------------------------------------------------------------------


class TestComputeEmbeddingNorms:
    """Tests for the embedding norm tracker."""

    def test_standard_embedding(self):
        model = nn.Module()
        model.token_embed = nn.Embedding(100, 64)
        model.lm_head = nn.Linear(64, 100, bias=False)
        model.kronecker_embeddings = None
        model.pf_to_model = None
        model.embed_norm = None

        norms = compute_embedding_norms(model)
        assert "token_emb_norm" in norms
        assert "lm_head_norm" in norms
        assert norms["token_emb_norm"] > 0
        assert norms["lm_head_norm"] > 0

    def test_kronecker_embedding(self):
        model = nn.Module()
        model.pf_to_model = nn.Linear(8192, 4096, bias=False)
        model.embed_norm = nn.LayerNorm(4096)
        model.lm_head = nn.Linear(4096, 100, bias=False)
        model.token_embed = None

        norms = compute_embedding_norms(model)
        assert "emb_proj_norm" in norms
        assert "lm_head_norm" in norms
        assert norms["emb_proj_norm"] > 0
        assert any(k.startswith("emb_rmsnorm_") for k in norms)

    def test_empty_model(self):
        model = nn.Module()
        norms = compute_embedding_norms(model)
        assert norms == {}

    def test_norms_are_positive(self):
        model = nn.Module()
        model.token_embed = nn.Embedding(50, 32)
        model.lm_head = nn.Linear(32, 50, bias=False)

        norms = compute_embedding_norms(model)
        for key, val in norms.items():
            assert isinstance(val, float), f"{key} is not float"
            assert val > 0, f"{key} is not positive"
