"""IndicMT-Eval: Meta-evaluate MT metrics for Indian languages."""

from benchmark_indic_mt_eval.config import BenchmarkConfig, load_config
from benchmark_indic_mt_eval.runner import run_benchmark

__all__ = ["BenchmarkConfig", "load_config", "run_benchmark"]
