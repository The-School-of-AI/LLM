"""Metrics package for tokenizer evaluation."""

from .compression import (
    tokens_per_byte,
    tokens_per_char,
    compression_ratio,
    compute_compression_metrics,
)
from .fertility import (
    fertility,
    per_language_fertility,
    word_tokenize,
    compute_fertility_metrics,
)
from .speed import (
    encoding_throughput,
    decoding_throughput,
    benchmark_speed,
)
from .code_quality import (
    keyword_preservation_score,
    identifier_quality_score,
    compute_code_quality_metrics,
)

__all__ = [
    # Compression
    'tokens_per_byte',
    'tokens_per_char',
    'compression_ratio',
    'compute_compression_metrics',
    # Fertility
    'fertility',
    'per_language_fertility',
    'word_tokenize',
    'compute_fertility_metrics',
    # Speed
    'encoding_throughput',
    'decoding_throughput',
    'benchmark_speed',
    # Code quality
    'keyword_preservation_score',
    'identifier_quality_score',
    'compute_code_quality_metrics',
]
