"""MT metrics for IndicMT-Eval benchmark."""

# Import to trigger registration
import benchmark_indic_mt_eval.metrics.overlap  # noqa: F401

# Optional neural metrics — import only if dependencies available
try:
    import benchmark_indic_mt_eval.metrics.embedding  # noqa: F401
except ImportError:
    pass

try:
    import benchmark_indic_mt_eval.metrics.trained  # noqa: F401
except ImportError:
    pass
