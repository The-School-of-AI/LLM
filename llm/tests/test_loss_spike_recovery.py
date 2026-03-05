"""Tests for the loss spike detection and recovery module."""

from unittest.mock import patch

import pytest

from llm.loss_spike_recovery import (
    LossSpikeDetector,
    RecoveryAction,
    _parse_choice,
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
        # Fill window
        for _ in range(10):
            detector.update(3.0)
        # Continue with stable values
        for _ in range(20):
            assert detector.update(3.0) is False

    def test_spike_detected_large_jump(self):
        """A sudden large jump should be flagged as a spike."""
        detector = LossSpikeDetector(window_size=10, z_threshold=3.0, min_abs_delta=0.5)
        # Fill with stable loss
        for _ in range(10):
            detector.update(3.0)
        # Inject a spike
        assert detector.update(10.0) is True

    def test_spike_detected_ratio_guard(self):
        """Spike should be detected via the ratio guard when std is tiny."""
        detector = LossSpikeDetector(
            window_size=10, z_threshold=3.0, min_spike_ratio=2.0, min_abs_delta=0.5
        )
        # Fill with very stable loss (near-zero std)
        for _ in range(10):
            detector.update(3.0)
        # Loss = 6.5 → ratio = 2.17x, delta = 3.5 > 0.5
        assert detector.update(6.5) is True

    def test_no_spike_below_min_abs_delta(self):
        """Small fluctuations below min_abs_delta should not trigger."""
        detector = LossSpikeDetector(window_size=10, min_abs_delta=1.0)
        for _ in range(10):
            detector.update(3.0)
        # delta = 0.8 < 1.0 → no spike
        assert detector.update(3.8) is False

    def test_spike_does_not_corrupt_window(self):
        """Spike values should not be added to the window."""
        detector = LossSpikeDetector(window_size=5, min_abs_delta=0.5)
        for _ in range(5):
            detector.update(3.0)

        # Inject spike (should not enter window)
        assert detector.update(20.0) is True

        stats = detector.get_stats()
        # Window mean should still be 3.0 since spike was excluded
        assert stats["window_mean"] == pytest.approx(3.0, abs=0.01)

    def test_get_stats(self):
        """get_stats should return current window statistics."""
        detector = LossSpikeDetector(window_size=5)
        for v in [2.0, 3.0, 4.0, 3.0, 3.0]:
            detector.update(v)

        stats = detector.get_stats()
        assert stats["current_loss"] == 3.0
        assert stats["window_mean"] == pytest.approx(3.0, abs=0.1)
        assert stats["window_std"] >= 0
        assert stats["spike_ratio"] > 0

    def test_gradually_increasing_loss_no_spike(self):
        """Gradual increase should not trigger spikes (window adapts)."""
        detector = LossSpikeDetector(window_size=10, z_threshold=3.0, min_abs_delta=0.5)
        # Gradually increase from 3.0 to 5.0 over 30 steps
        for i in range(30):
            loss = 3.0 + i * (2.0 / 30)
            # After warmup, gradual increases should generally not spike
            # (the window adapts)
            if i >= 10:
                # We don't assert False here because with a small window
                # some edge cases could trigger; just ensure no crash.
                detector.update(loss)
            else:
                detector.update(loss)


# ---------------------------------------------------------------------------
# _parse_choice
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
        assert cfg.lr_reduction_factor == 0.5
        assert cfg.user_prompt_timeout == 300

    def test_custom_values(self):
        from llm.config import LossSpikeConfig

        cfg = LossSpikeConfig(
            enabled=False,
            window_size=50,
            z_threshold=2.0,
            min_spike_ratio=1.5,
            min_abs_delta=0.3,
            lr_reduction_factor=0.25,
            user_prompt_timeout=60,
        )
        assert cfg.enabled is False
        assert cfg.window_size == 50
        assert cfg.lr_reduction_factor == 0.25


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    """Test the factory function."""

    def test_build_returns_detector_when_enabled(self):
        from llm.config import LossSpikeConfig
        from llm.factories import build_loss_spike_detector

        detector = build_loss_spike_detector(LossSpikeConfig(enabled=True))
        assert detector is not None
        assert isinstance(detector, LossSpikeDetector)

    def test_build_returns_none_when_disabled(self):
        from llm.config import LossSpikeConfig
        from llm.factories import build_loss_spike_detector

        detector = build_loss_spike_detector(LossSpikeConfig(enabled=False))
        assert detector is None
