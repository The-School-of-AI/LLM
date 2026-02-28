"""Core modules for curriculum tagging."""

from .plugin import MetricPlugin
from .tagger import CurriculumTagger

__all__ = [
    "MetricPlugin",
    "CurriculumTagger",
]
