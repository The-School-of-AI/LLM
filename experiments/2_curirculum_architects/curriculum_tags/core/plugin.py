"""Core plugin system for extensible metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..utils.curriculum_loader import CurriculumConfig


class MetricPlugin(ABC):
    """Base class for all curriculum metric plugins.

    Plugins receive curriculum config for classification thresholds.
    Each plugin sees the sample with accumulated tags from previous plugins.
    """

    name: str = "base_metric"  # Override in subclass

    def __init__(self, config: CurriculumConfig):
        """Initialize plugin with curriculum configuration.

        Args:
            config: Curriculum configuration for thresholds and values
        """
        self.config = config

    @abstractmethod
    def compute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Compute metric tags for a single sample.

        Args:
            sample: Data sample with 'text', 'metadata', and 'curriculum_tags'
                   (curriculum_tags has results from previous plugins)

        Returns:
            Dictionary of computed metric tags to add

        Example:
            >>> plugin.compute({'text': 'Hello world', 'curriculum_tags': {}})
            {'band': 'B0', 'score': 0.12}
        """
        pass
