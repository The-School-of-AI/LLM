"""Curriculum Extractor - Distributed metadata extraction pipeline for LLM training data.

This package provides:
- Scalable extraction of curriculum metrics from parquet files
- State management for incremental processing
- Rejection layer for quality control
- Distributed processing with Ray
"""

from .core.extractor import CurriculumExtractor
from .core.plugin import MetricPlugin
from .core.state_manager import StateManager
from .utils.curriculum_loader import CurriculumConfig

__all__ = [
    "CurriculumExtractor",
    "MetricPlugin",
    "StateManager",
    "CurriculumConfig",
]

__version__ = "0.2.0"
