"""Core plugin system for extensible metrics.

Key Design Principles:
- Records are READ-ONLY: plugins receive immutable copies of data
- No plugin chaining: each metric operates independently on original data  
- Early rejection: if a metric rejects a record, no further metrics run
- Level-based parallelism: metrics at the same level can run in parallel
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from ..utils.curriculum_loader import CurriculumConfig


@dataclass(frozen=True)
class ExtractionResult:
    """Immutable result from a metric computation.

    Attributes:
        metrics: Dictionary of computed metric values (will be flattened to columns)
        rejected: Whether this record should be rejected
        rejection_reason: Reason for rejection (if rejected)
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    rejected: bool = False
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        # Ensure metrics is a regular dict (not frozen)
        if not isinstance(self.metrics, dict):
            object.__setattr__(self, "metrics", dict(self.metrics))


class ReadOnlyRecord(Mapping):
    """Immutable wrapper around a record to prevent modifications.

    Provides dict-like read access but raises TypeError on write attempts.
    """

    def __init__(self, data: Dict[str, Any]):
        self._data = deepcopy(data)  # Deep copy to ensure isolation

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        # Return deep copy of mutable values
        if isinstance(value, (dict, list)):
            return deepcopy(value)
        return value

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return self._data.keys()

    def values(self):
        return [deepcopy(v) for v in self._data.values()]

    def items(self):
        return [(k, deepcopy(v)) for k, v in self._data.items()]

    def to_dict(self) -> Dict[str, Any]:
        """Return a mutable deep copy of the underlying data."""
        return deepcopy(self._data)


class MetricPlugin(ABC):
    """Base class for all curriculum metric plugins.

    Plugins receive curriculum config for classification thresholds.

    IMPORTANT Design Principles:
    - Records are READ-ONLY: never modify input records
    - No plugin chaining: each metric works on original data only
    - Rejection stops pipeline: rejected records skip remaining metrics

    Attributes:
        name: Unique identifier for this metric
        level: Execution level (0-N). Same level metrics can run in parallel.
               Lower levels execute first. Default is 0.
        column_prefix: Prefix for output column names (defaults to name)
    """

    name: str = "base_metric"  # Override in subclass
    level: int = 0  # Execution level for parallelism
    column_prefix: Optional[str] = None

    def __init__(self, config: CurriculumConfig):
        """Initialize plugin with curriculum configuration.

        Args:
            config: Curriculum configuration for thresholds and values
        """
        self.config = config
        if self.column_prefix is None:
            self.column_prefix = self.name

    @abstractmethod
    def compute(self, record: ReadOnlyRecord) -> Dict[str, Any]:
        """Compute metric values for a single record.

        Args:
            record: Read-only data record with 'text' and other fields.
                   DO NOT attempt to modify this record.

        Returns:
            Dictionary of computed metric values

        Example:
            >>> plugin.compute(ReadOnlyRecord({'text': 'Hello world'}))
            {'score': 0.12, 'features': {'count': 2}}
        """
        pass

    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        """Extract metrics with rejection support.

        Override this method to implement rejection logic.
        Default implementation wraps compute() with no rejection.

        Args:
            record: Read-only data record

        Returns:
            ExtractionResult with metrics and optional rejection info
        """
        metrics = self.compute(record)
        return ExtractionResult(metrics=metrics, rejected=False)

    def flatten_metrics(
        self, metrics: Dict[str, Any], prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Flatten nested metric dict to column-friendly format.

        Args:
            metrics: Nested dictionary of metrics
            prefix: Optional prefix for keys (defaults to column_prefix)

        Returns:
            Flattened dictionary with underscored keys

        Example:
            >>> plugin.flatten_metrics({'features': {'count': 10}})
            {'metricname_features_count': 10}
        """
        prefix = prefix or self.column_prefix
        result = {}

        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key

            if isinstance(value, dict):
                nested = self.flatten_metrics(value, full_key)
                result.update(nested)
            else:
                result[full_key] = value

        return result
