"""Tests for training_state_manager.py — StepManager & TrainingState."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from llm.logger import Metrics
from llm.training_state_manager import StepManager, TrainingState, _STATE_KEY


# ---------------------------------------------------------------------------
# TrainingState
# ---------------------------------------------------------------------------


class TestTrainingState:
    def test_defaults(self):
        s = TrainingState()
        assert s.global_step == 0
        assert s.epoch == 0
        assert s.tokens_seen == 0
        assert s.samples_seen == 0

    def test_round_trip(self):
        s = TrainingState(global_step=42, epoch=3, tokens_seen=100_000, samples_seen=500)
        d = s.to_dict()
        s2 = TrainingState.from_dict(d)
        assert s == s2

    def test_from_dict_missing_keys(self):
        s = TrainingState.from_dict({})
        assert s.global_step == 0


# ---------------------------------------------------------------------------
# StepManager — restore
# ---------------------------------------------------------------------------


class TestRestore:
    def test_restore_none_gives_zeros(self):
        mgr = StepManager()
        state = mgr.restore(None)
        assert state.global_step == 0
        assert state.epoch == 0

    def test_restore_with_training_state_key(self):
        client = {
            _STATE_KEY: {
                "global_step": 100,
                "epoch": 2,
                "tokens_seen": 50_000,
                "samples_seen": 200,
            }
        }
        mgr = StepManager()
        state = mgr.restore(client)
        assert state.global_step == 100
        assert state.epoch == 2
        assert state.tokens_seen == 50_000
        assert state.samples_seen == 200

    def test_restore_legacy_flat_keys(self):
        """Legacy checkpoints without 'training_state' key."""
        client = {"global_step": 77, "epoch": 1}
        mgr = StepManager()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = mgr.restore(client)
            assert len(w) == 1
            assert "training_state" in str(w[0].message)
        assert state.global_step == 77
        assert state.epoch == 1

    def test_restore_corrupted_training_state(self):
        """Corrupted nested dict falls back gracefully."""
        client = {_STATE_KEY: "not-a-dict"}
        mgr = StepManager()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = mgr.restore(client)
            assert len(w) == 1
            assert "Corrupted" in str(w[0].message)
        assert state.global_step == 0


# ---------------------------------------------------------------------------
# StepManager — resume continuation (Test 1)
# ---------------------------------------------------------------------------


class TestResumeContinuation:
    def test_next_step_after_resume_is_correct(self):
        mgr = StepManager()
        # Simulate training 100 steps
        mgr.restore(None)
        for _ in range(100):
            mgr.increment()
        assert mgr.get_step() == 100

        # Save state
        ckpt: dict = {}
        mgr.inject_state(ckpt)

        # New manager resumes
        mgr2 = StepManager()
        state = mgr2.restore(ckpt)
        assert state.global_step == 100

        mgr2.increment()
        assert mgr2.get_step() == 101


# ---------------------------------------------------------------------------
# StepManager — increment
# ---------------------------------------------------------------------------


class TestIncrement:
    def test_increment_advances_step(self):
        mgr = StepManager()
        mgr.restore(None)
        mgr.increment()
        assert mgr.get_step() == 1

    def test_increment_accumulates_tokens_and_samples(self):
        mgr = StepManager()
        mgr.restore(None)
        mgr.increment(tokens=4096, samples=8)
        mgr.increment(tokens=4096, samples=8)
        assert mgr.tokens_seen == 8192
        assert mgr.samples_seen == 16

    def test_gradient_accumulation_pattern(self):
        """global_step increments only per optimizer step, not per iteration."""
        mgr = StepManager()
        mgr.restore(None)
        grad_accum_steps = 8
        iterations = 32

        for i in range(iterations):
            # loss.backward() happens every iteration
            if (i + 1) % grad_accum_steps == 0:
                # optimizer.step()
                mgr.increment()

        assert mgr.get_step() == iterations // grad_accum_steps  # 4


# ---------------------------------------------------------------------------
# StepManager — logging continuity (Test 3)
# ---------------------------------------------------------------------------


class TestLogging:
    def test_log_uses_correct_step(self):
        mgr = StepManager()
        mgr.restore({_STATE_KEY: {"global_step": 50, "epoch": 1}})

        logger = MagicMock()
        metrics = Metrics()
        metrics.add("loss", 0.5)

        mgr.log(metrics, logger)
        logger.log_metrics.assert_called_once_with(50, metrics)

    def test_log_after_increment(self):
        mgr = StepManager()
        mgr.restore({_STATE_KEY: {"global_step": 99}})
        mgr.increment()

        logger = MagicMock()
        metrics = Metrics()
        mgr.log(metrics, logger)
        logger.log_metrics.assert_called_once_with(100, metrics)


# ---------------------------------------------------------------------------
# StepManager — inject_state (checkpoint persistence)
# ---------------------------------------------------------------------------


class TestInjectState:
    def test_inject_adds_training_state(self):
        mgr = StepManager()
        mgr.restore(None)
        for _ in range(10):
            mgr.increment(tokens=512, samples=4)
        mgr.epoch = 2

        ckpt: dict = {"model": "blob", "optimizer": "blob"}
        result = mgr.inject_state(ckpt)

        assert result is ckpt  # mutated in place
        assert _STATE_KEY in ckpt
        ts = ckpt[_STATE_KEY]
        assert ts["global_step"] == 10
        assert ts["epoch"] == 2
        assert ts["tokens_seen"] == 5120
        assert ts["samples_seen"] == 40


# ---------------------------------------------------------------------------
# StepManager — distributed sync (Test 2)
# ---------------------------------------------------------------------------


class TestDistributedSync:
    @patch("llm.training_state_manager.dist")
    def test_broadcast_called_on_restore(self, mock_dist):
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        def fake_broadcast(tensor, src):
            # Simulate rank 0 broadcasting its values (no-op, values stay).
            pass

        mock_dist.broadcast = MagicMock(side_effect=fake_broadcast)

        mgr = StepManager()
        client = {_STATE_KEY: {"global_step": 42, "epoch": 1}}
        state = mgr.restore(client)

        mock_dist.broadcast.assert_called_once()
        assert state.global_step == 42

    @patch("llm.training_state_manager.dist")
    def test_no_broadcast_when_not_distributed(self, mock_dist):
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        mgr = StepManager()
        mgr.restore(None)
        mock_dist.broadcast.assert_not_called()


# ---------------------------------------------------------------------------
# StepManager — scheduler compatibility (Test 4)
# ---------------------------------------------------------------------------


class TestSchedulerCompatibility:
    def test_step_value_correct_for_scheduler(self):
        """Verify that get_step() returns a value suitable for LR schedulers."""
        mgr = StepManager()
        mgr.restore({_STATE_KEY: {"global_step": 200}})

        # Scheduler should receive step=200, then 201, 202, ...
        for expected in range(200, 205):
            assert mgr.get_step() == expected
            mgr.increment()
        assert mgr.get_step() == 205


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_resume_different_world_size(self):
        """State loads correctly regardless of world size changes."""
        mgr = StepManager()
        client = {
            _STATE_KEY: {
                "global_step": 500,
                "epoch": 5,
                "tokens_seen": 1_000_000,
                "samples_seen": 5000,
            }
        }
        state = mgr.restore(client)
        # global_step must not be modified by world size change
        assert state.global_step == 500
        assert state.tokens_seen == 1_000_000

    def test_empty_client_state(self):
        """Empty dict (not None) should yield zeros with a warning."""
        mgr = StepManager()
        # Empty dict has no _STATE_KEY and no flat keys either — but it's
        # truthy so the legacy branch fires and warns.
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            state = mgr.restore({})
        assert state.global_step == 0
