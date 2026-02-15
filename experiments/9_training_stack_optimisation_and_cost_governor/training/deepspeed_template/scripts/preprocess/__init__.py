"""
Preprocessing package for tokenizing and sharding multi-source training data.

Components:
    - readers: Format-specific document readers for each data source
    - tokenizer_worker: Parallel tokenization across CPU cores
    - sharder: Uniform re-sharding with multi-source mixing
"""

from scripts.preprocess.readers import (
    DolmaReader,
    IndicNLPReader,
    NCERTReader,
    SangrahaReader,
    get_reader_for_source,
)
from scripts.preprocess.sharder import UniformSharder
from scripts.preprocess.tokenizer_worker import TokenizerWorker, parallel_tokenize

__all__ = [
    "DolmaReader",
    "SangrahaReader",
    "NCERTReader",
    "IndicNLPReader",
    "get_reader_for_source",
    "TokenizerWorker",
    "parallel_tokenize",
    "UniformSharder",
]
