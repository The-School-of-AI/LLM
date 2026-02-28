"""Curriculum Tags: Plugin-based metadata tagging for datasets."""

import importlib
import inspect
from pathlib import Path

__version__ = "0.1.0"

from .core.plugin import MetricPlugin
from .core.tagger import CurriculumTagger
from .utils.curriculum_loader import CurriculumConfig

# Auto-discover all metrics from the metrics directory
_metrics_dir = Path(__file__).parent / "metrics"
_discovered_metrics = {}

for _metric_file in _metrics_dir.glob("*.py"):
    if _metric_file.stem.startswith("_"):
        continue  # Skip __init__.py and private modules

    try:
        # Import the module
        _module = importlib.import_module(
            f".metrics.{_metric_file.stem}", package=__package__
        )

        # Find all MetricPlugin subclasses in the module
        for _name, _obj in inspect.getmembers(_module, inspect.isclass):
            if (
                issubclass(_obj, MetricPlugin)
                and _obj is not MetricPlugin
                and _obj.__module__ == _module.__name__
            ):
                _discovered_metrics[_name] = _obj
                globals()[_name] = _obj
    except Exception:
        # Silently skip modules that fail to import
        pass

# Build __all__ dynamically
__all__ = [
    "MetricPlugin",
    "CurriculumTagger",
    "CurriculumConfig",
] + list(_discovered_metrics.keys())

# Clean up temporary variables
del _metrics_dir, _discovered_metrics, _metric_file
