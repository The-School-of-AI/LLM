from abc import ABC, abstractmethod

from .metrics import Metrics


class Logger(ABC):
    @abstractmethod
    def log_step(
        self,
        step: int,
        metrics: dict,
        context: dict | None = None,
    ):
        """
        Log a training step.

        Parameters
        ----------
        step : int
            Global training step.
        metrics : dict
            Scalar metrics, e.g. {"loss": 0.5, "lr": 3e-4, "tokens_per_second": 12000}.
        context : dict | None
            Optional per-step context, e.g. {"epoch": 1, "phase": "warmup"}.
        """
        pass

    def log_metrics(self, step: int, metrics: Metrics):
        """
        Helper function to log metrics object
        """

        self.log_step(step, metrics._values)
