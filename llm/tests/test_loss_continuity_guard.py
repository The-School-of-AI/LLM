"""Tests for LossContinuityGuard."""

import torch

from llm.loss_continuity_guard import LossContinuityGuard


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _feed_losses(guard: LossContinuityGuard, values: list[float]) -> None:
    """Feed a sequence of scalar losses into the guard."""
    for v in values:
        guard.observe(torch.tensor(v))


# ------------------------------------------------------------------
# Test 1 — Normal resume: loss is continuous
# ------------------------------------------------------------------


def test_normal_resume_passes_verification():
    """Train → checkpoint → resume with consistent loss → PASS."""
    guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)

    # Simulate pre-checkpoint training
    pre_losses = [2.10, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 2.02, 2.01]
    _feed_losses(guard, pre_losses)

    # Save & restore
    state = guard.state_dict()
    assert state, "state_dict should not be empty after observing losses"

    new_guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)
    new_guard.restore(state)

    # Simulate post-resume training with similar losses
    post_losses = [2.00, 1.99, 1.98, 1.97, 1.96, 1.95, 1.94, 1.93, 1.92, 1.91]
    _feed_losses(new_guard, post_losses)

    # verify() should have been called automatically; call explicitly to get result
    assert new_guard.verify() is True


# ------------------------------------------------------------------
# Test 2 — Optimizer reset: large loss spike after resume
# ------------------------------------------------------------------


def test_optimizer_reset_detected():
    """Resume with broken optimizer state → loss spike → FAIL."""
    guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)

    pre_losses = [2.10, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 2.02, 2.01]
    _feed_losses(guard, pre_losses)

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)
    new_guard.restore(state)

    # Simulate broken resume — loss jumps to ~5.8
    post_losses = [5.80, 5.75, 5.90, 5.85, 5.70, 5.82, 5.88, 5.91, 5.79, 5.83]
    _feed_losses(new_guard, post_losses)

    assert new_guard.verify() is False


# ------------------------------------------------------------------
# Test 3 — LR scheduler reset: moderate but clear loss increase
# ------------------------------------------------------------------


def test_lr_scheduler_reset_detected():
    """Incorrect LR after resume causes elevated loss → FAIL."""
    guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)

    pre_losses = [2.10, 2.09, 2.08, 2.07, 2.06, 2.05, 2.04, 2.03, 2.02, 2.01]
    _feed_losses(guard, pre_losses)

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)
    new_guard.restore(state)

    # Loss jumps to ~3.5 — clearly outside tolerance
    post_losses = [3.50, 3.48, 3.52, 3.49, 3.51, 3.47, 3.53, 3.50, 3.48, 3.52]
    _feed_losses(new_guard, post_losses)

    assert new_guard.verify() is False


# ------------------------------------------------------------------
# Test 4 — RNG reset: small shift within tolerance
# ------------------------------------------------------------------


def test_rng_reset_within_tolerance():
    """RNG reset causes slight loss drift but stays within bounds → PASS."""
    guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)

    # Slightly noisy pre-checkpoint losses
    pre_losses = [2.10, 2.12, 2.08, 2.11, 2.09, 2.13, 2.07, 2.10, 2.12, 2.08]
    _feed_losses(guard, pre_losses)

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=10, tolerance_sigma=3.0)
    new_guard.restore(state)

    # Slightly shifted but within 3σ
    post_losses = [2.14, 2.16, 2.12, 2.15, 2.13, 2.17, 2.11, 2.14, 2.16, 2.12]
    _feed_losses(new_guard, post_losses)

    assert new_guard.verify() is True


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_empty_restore_skips_verification():
    """Restoring empty state should disable verification."""
    guard = LossContinuityGuard(window_size=5)
    guard.restore({})

    _feed_losses(guard, [5.0, 5.0, 5.0, 5.0, 5.0])
    assert guard.verify() is True


def test_fresh_guard_no_state_dict():
    """A guard with no observations returns empty state_dict."""
    guard = LossContinuityGuard()
    assert guard.state_dict() == {}


def test_state_dict_contains_expected_keys():
    """state_dict should contain loss_window, loss_mean, loss_std."""
    guard = LossContinuityGuard(window_size=5)
    _feed_losses(guard, [1.0, 2.0, 3.0, 4.0, 5.0])

    state = guard.state_dict()
    assert "loss_window" in state
    assert "loss_mean" in state
    assert "loss_std" in state
    assert len(state["loss_window"]) == 5
    assert abs(state["loss_mean"] - 3.0) < 1e-6


def test_window_rolls_over():
    """Window should discard oldest values when full."""
    guard = LossContinuityGuard(window_size=3)
    _feed_losses(guard, [1.0, 2.0, 3.0, 4.0, 5.0])

    state = guard.state_dict()
    assert state["loss_window"] == [3.0, 4.0, 5.0]


def test_automatic_verification_trigger():
    """Verification should trigger automatically once window_size losses
    have been observed post-resume."""
    guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    _feed_losses(guard, [2.0, 2.0, 2.0, 2.0, 2.0])

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    new_guard.restore(state)

    # Feed exactly window_size losses — verification should auto-trigger
    _feed_losses(new_guard, [2.01, 1.99, 2.00, 2.02, 1.98])

    # verification_done should be True now
    assert new_guard._verification_done is True
    assert new_guard.verify() is True  # idempotent after done


def test_relative_check_fallback():
    """When std is very small, the relative-difference check (20%) should
    still allow moderate drift."""
    guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    # All identical → std ≈ 0
    _feed_losses(guard, [2.0, 2.0, 2.0, 2.0, 2.0])

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    new_guard.restore(state)

    # 10% increase — within 20% relative threshold
    _feed_losses(new_guard, [2.2, 2.2, 2.2, 2.2, 2.2])
    assert new_guard.verify() is True


def test_relative_check_fails_large_drift():
    """When std is very small and drift exceeds 20%, verification fails."""
    guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    _feed_losses(guard, [2.0, 2.0, 2.0, 2.0, 2.0])

    state = guard.state_dict()
    new_guard = LossContinuityGuard(window_size=5, tolerance_sigma=3.0)
    new_guard.restore(state)

    # 50% increase — exceeds both sigma and relative thresholds
    _feed_losses(new_guard, [3.0, 3.0, 3.0, 3.0, 3.0])
    assert new_guard.verify() is False
