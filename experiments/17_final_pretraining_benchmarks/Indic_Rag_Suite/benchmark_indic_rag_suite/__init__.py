"""
Unified benchmark harness for Indic-Rag-Suite and IndicMSMARCO.
Supports retrieval and generation evaluation with dev/test/verify flows.
"""

from benchmark_indic_rag_suite.config import BenchmarkConfig, load_config
from benchmark_indic_rag_suite.runner import run_benchmark

__all__ = ["BenchmarkConfig", "load_config", "run_benchmark"]
