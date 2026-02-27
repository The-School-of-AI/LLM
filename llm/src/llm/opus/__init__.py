"""OPUS data selection — Optimizer-induced Projected Utility Selection (arXiv:2602.05400)."""
from .config import OpusConfig
from .countsketch import CountSketchProjector
from .data_selector import OpusDataSelector
from .ghost import GhostCollector, LayerCapture, MoERoutedCapture
from .preconditioner import AdamWPreconditionerView
from .proxy import BenchProxyProvider, ProxyProvider, RandomInDistributionProxyProvider
from .selector import OpusSelector, SelectionResult

__all__ = [
    "OpusConfig",
    "OpusDataSelector",
    "CountSketchProjector",
    "GhostCollector",
    "LayerCapture",
    "MoERoutedCapture",
    "AdamWPreconditionerView",
    "ProxyProvider",
    "RandomInDistributionProxyProvider",
    "BenchProxyProvider",
    "OpusSelector",
    "SelectionResult",
]
