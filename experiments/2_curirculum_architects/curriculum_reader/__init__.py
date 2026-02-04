"""Curriculum Reader - Analysis and batch creation from metadata layer.

This package provides:
- Reading and querying the metadata layer
- Deterministic batch creation for training
- Analysis utilities for curriculum statistics
"""

from .core.analyzer import MetadataAnalyzer
from .core.batch_creator import BatchConfig, BatchCreator
from .core.reader import MetadataReader, RejectionReader

__all__ = [
    "MetadataReader",
    "RejectionReader",
    "BatchCreator",
    "BatchConfig",
    "MetadataAnalyzer",
]

__version__ = "0.2.0"
