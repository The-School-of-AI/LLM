"""IndicMT-Eval: Meta-evaluate MT metrics for Indian languages."""

__all__ = ["BenchmarkConfig", "load_config", "run_benchmark"]


def __getattr__(name):
    if name == "BenchmarkConfig":
        from benchmark_indic_mt_eval.config import BenchmarkConfig
        return BenchmarkConfig
    if name == "load_config":
        from benchmark_indic_mt_eval.config import load_config
        return load_config
    if name == "run_benchmark":
        from benchmark_indic_mt_eval.runner import run_benchmark
        return run_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
