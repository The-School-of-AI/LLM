"""
Speed Metrics - Measure tokenizer throughput performance.

Metrics:
- Encoding throughput (tokens/second, chars/second)
- Decoding throughput (tokens/second, chars/second)
- Latency statistics
"""

import time
from typing import List, Dict, Any, Optional
import sys
import statistics

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
        def decode(self, token_ids: List[int]) -> str: ...
else:
    TokenizerProtocol = Any


def encoding_throughput(
    tokenizer: TokenizerProtocol,
    texts: List[str],
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure encoding throughput.
    
    Args:
        tokenizer: Tokenizer with encode method
        texts: List of texts to encode
        iterations: Number of iterations
        warmup: Warmup iterations (not counted)
    
    Returns:
        Throughput metrics (tokens/sec, chars/sec)
    """
    # Warmup
    for _ in range(warmup):
        for text in texts:
            tokenizer.encode(text)
    
    total_chars = sum(len(t) for t in texts)
    total_tokens = 0
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        for text in texts:
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
        end = time.perf_counter()
        latencies.append(end - start)
    
    total_time = sum(latencies)
    tokens_per_iteration = total_tokens / iterations
    
    return {
        'tokens_per_second': total_tokens / total_time,
        'chars_per_second': (total_chars * iterations) / total_time,
        'mean_latency_ms': (statistics.mean(latencies) * 1000),
        'p50_latency_ms': (statistics.median(latencies) * 1000),
        'p99_latency_ms': (sorted(latencies)[int(len(latencies) * 0.99)] * 1000) if len(latencies) >= 100 else (max(latencies) * 1000),
        'total_tokens': total_tokens,
        'iterations': iterations,
    }


def decoding_throughput(
    tokenizer: TokenizerProtocol,
    token_lists: List[List[int]],
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure decoding throughput.
    
    Args:
        tokenizer: Tokenizer with decode method
        token_lists: List of token ID lists to decode
        iterations: Number of iterations
        warmup: Warmup iterations (not counted)
    
    Returns:
        Throughput metrics (tokens/sec, chars/sec)
    """
    # Warmup
    for _ in range(warmup):
        for tokens in token_lists:
            tokenizer.decode(tokens)
    
    total_tokens = sum(len(t) for t in token_lists)
    total_chars = 0
    latencies = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        for tokens in token_lists:
            text = tokenizer.decode(tokens)
            total_chars += len(text)
        end = time.perf_counter()
        latencies.append(end - start)
    
    total_time = sum(latencies)
    
    return {
        'tokens_per_second': (total_tokens * iterations) / total_time,
        'chars_per_second': total_chars / total_time,
        'mean_latency_ms': (statistics.mean(latencies) * 1000),
        'p50_latency_ms': (statistics.median(latencies) * 1000),
        'p99_latency_ms': (sorted(latencies)[int(len(latencies) * 0.99)] * 1000) if len(latencies) >= 100 else (max(latencies) * 1000),
        'total_tokens': total_tokens * iterations,
        'iterations': iterations,
    }


def benchmark_speed(
    tokenizer: TokenizerProtocol,
    texts: List[str],
    iterations: int = 100,
    warmup: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Run complete speed benchmark (encoding + decoding).
    
    Args:
        tokenizer: Tokenizer with encode and decode methods
        texts: List of texts to benchmark
        iterations: Number of iterations per metric
        warmup: Warmup iterations
    
    Returns:
        Dictionary with 'encoding' and 'decoding' metrics
    """
    # Encoding benchmark
    encoding_metrics = encoding_throughput(tokenizer, texts, iterations, warmup)
    
    # Prepare token lists for decoding
    token_lists = [tokenizer.encode(text) for text in texts]
    
    # Decoding benchmark
    decoding_metrics = decoding_throughput(tokenizer, token_lists, iterations, warmup)
    
    return {
        'encoding': encoding_metrics,
        'decoding': decoding_metrics,
    }


def compare_speed(
    tokenizers: Dict[str, TokenizerProtocol],
    texts: List[str],
    iterations: int = 50,
    warmup: int = 5
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare speed across multiple tokenizers.
    
    Args:
        tokenizers: Dictionary of name -> tokenizer
        texts: List of texts to benchmark
        iterations: Number of iterations
        warmup: Warmup iterations
    
    Returns:
        Nested dict: tokenizer -> metric_type -> values
    """
    results = {}
    
    for name, tokenizer in tokenizers.items():
        try:
            results[name] = benchmark_speed(tokenizer, texts, iterations, warmup)
        except Exception as e:
            results[name] = {'error': str(e)}
    
    return results


def quick_speed_test(
    tokenizer: TokenizerProtocol,
    text: str,
    iterations: int = 10
) -> Dict[str, float]:
    """
    Quick speed test for a single text.
    
    Useful for quick sanity checks.
    
    Args:
        tokenizer: Tokenizer to test
        text: Text to encode/decode
        iterations: Number of iterations
    
    Returns:
        Simple timing results
    """
    # Encoding
    start = time.perf_counter()
    tokens = None
    for _ in range(iterations):
        tokens = tokenizer.encode(text)
    encode_time = time.perf_counter() - start
    
    # Decoding
    start = time.perf_counter()
    for _ in range(iterations):
        tokenizer.decode(tokens)
    decode_time = time.perf_counter() - start
    
    return {
        'encode_ms_per_call': (encode_time / iterations) * 1000,
        'decode_ms_per_call': (decode_time / iterations) * 1000,
        'num_tokens': len(tokens),
        'text_length': len(text),
    }
