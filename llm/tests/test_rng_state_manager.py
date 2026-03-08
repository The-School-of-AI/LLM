"""Tests for RNGStateManager."""

import random

import numpy as np
import pytest
import torch

from llm.rng_state_manager import RNGStateManager


class TestCapture:
    def test_returns_expected_keys(self):
        state = RNGStateManager.capture()
        assert "python" in state
        assert "numpy" in state
        assert "torch_cpu" in state
        assert "torch_cuda" in state

    def test_python_state_is_tuple(self):
        state = RNGStateManager.capture()
        assert isinstance(state["python"], tuple)

    def test_numpy_state_is_dict(self):
        state = RNGStateManager.capture()
        # np.random.get_state returns a dict (numpy >= 1.17) or tuple
        assert isinstance(state["numpy"], (dict, tuple))

    def test_torch_cpu_state_is_tensor(self):
        state = RNGStateManager.capture()
        assert isinstance(state["torch_cpu"], torch.Tensor)

    def test_torch_cuda_empty_when_no_gpu(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA is available, cannot test no-GPU path")
        state = RNGStateManager.capture()
        assert state["torch_cuda"] == []


class TestRoundTrip:
    def test_python_rng_restored(self):
        random.seed(42)
        state = RNGStateManager.capture()

        # Advance RNG
        _ = [random.random() for _ in range(100)]

        RNGStateManager.restore(state)
        # Should produce same sequence as right after seed(42)
        random.seed(42)
        expected = [random.random() for _ in range(5)]

        RNGStateManager.restore(state)
        actual = [random.random() for _ in range(5)]
        assert actual == expected

    def test_numpy_rng_restored(self):
        np.random.seed(42)
        state = RNGStateManager.capture()

        # Advance RNG
        _ = np.random.rand(100)

        RNGStateManager.restore(state)
        actual = np.random.rand(5)

        # Restore again to get expected
        RNGStateManager.restore(state)
        expected = np.random.rand(5)

        np.testing.assert_array_equal(actual, expected)

    def test_torch_cpu_rng_restored(self):
        torch.manual_seed(42)
        state = RNGStateManager.capture()

        # Advance RNG
        _ = torch.randn(100)

        RNGStateManager.restore(state)
        actual = torch.randn(5)

        RNGStateManager.restore(state)
        expected = torch.randn(5)

        assert torch.equal(actual, expected)

    def test_full_round_trip(self):
        """Capture all states, advance all RNGs, restore, verify all match."""
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)

        state = RNGStateManager.capture()

        # Advance all RNGs
        _ = random.random()
        _ = np.random.rand()
        _ = torch.randn(1)

        # Restore
        RNGStateManager.restore(state)

        # Generate "after restore" values
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.randn(1)

        # Restore again and compare
        RNGStateManager.restore(state)
        assert random.random() == py_val
        np.testing.assert_equal(np.random.rand(), np_val)
        assert torch.equal(torch.randn(1), torch_val)


class TestPartialRestore:
    def test_missing_keys_does_not_raise(self):
        """Restore with partial dict should not error."""
        RNGStateManager.restore({})
        RNGStateManager.restore({"python": random.getstate()})

    def test_empty_cuda_list_does_not_raise(self):
        RNGStateManager.restore({"torch_cuda": []})


class TestCheckpointIntegration:
    def test_state_can_be_merged_into_client_state(self):
        """RNG state dict should be serializable alongside checkpoint metadata."""
        rng_state = RNGStateManager.capture()
        client_state = {
            "epoch": 1,
            "step": 100,
            "global_step": 500,
            "rng_state": rng_state,
        }
        assert "rng_state" in client_state
        assert set(client_state["rng_state"].keys()) == {
            "python",
            "numpy",
            "torch_cpu",
            "torch_cuda",
        }
