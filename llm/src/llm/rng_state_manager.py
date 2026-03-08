"""RNG state manager for reproducible training resume.

Captures and restores random number generator states across all libraries
(Python, NumPy, PyTorch CPU, PyTorch CUDA) so that training after resume
behaves as if the interruption never happened.
"""

import random
import warnings
from typing import Any

import numpy as np
import torch


class RNGStateManager:
    """Captures and restores RNG states for reproducible training resume."""

    @staticmethod
    def capture() -> dict[str, Any]:
        """Capture all RNG states.

        Returns:
            Dict with keys: python, numpy, torch_cpu, torch_cuda.
        """
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else []
            ),
        }
        return state

    @staticmethod
    def restore(rng_state: dict[str, Any]) -> None:
        """Restore all RNG states from a previously captured dict.

        Args:
            rng_state: Dict produced by :meth:`capture`.
        """
        if "python" in rng_state:
            random.setstate(rng_state["python"])

        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])

        if "torch_cpu" in rng_state:
            torch.random.set_rng_state(rng_state["torch_cpu"])

        if "torch_cuda" in rng_state and rng_state["torch_cuda"]:
            if torch.cuda.is_available():
                saved = rng_state["torch_cuda"]
                current_device_count = torch.cuda.device_count()
                if len(saved) == current_device_count:
                    torch.cuda.set_rng_state_all(saved)
                else:
                    warnings.warn(
                        f"CUDA RNG state not restored: checkpoint has {len(saved)} "
                        f"device(s), current run has {current_device_count}. "
                        f"Reproducibility not guaranteed.",
                        UserWarning,
                        stacklevel=2,
                    )
