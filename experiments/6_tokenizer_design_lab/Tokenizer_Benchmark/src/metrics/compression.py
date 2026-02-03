"""
Compression Metrics - Measure tokenizer compression efficiency.

Metrics:
- Tokens per byte
- Tokens per character  
- Compression ratio (bytes per token)
"""

from typing import List, Dict, Any, Union
import sys

# Type alias for tokenizer interface
if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
else:
    TokenizerProtocol = Any


def tokens_per_byte(tokenizer: TokenizerProtocol, text: str) -> float:
    """
    Calculate tokens per byte ratio.
    
    Lower is better - indicates more efficient compression.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
    
    Returns:
        Tokens per byte ratio
    """
    if not text:
        return 0.0
    
    tokens = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    
    return len(tokens) / max(num_bytes, 1)


def tokens_per_char(tokenizer: TokenizerProtocol, text: str) -> float:
    """
    Calculate tokens per character ratio.
    
    Lower is better - indicates more efficient compression.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
    
    Returns:
        Tokens per character ratio
    """
    if not text:
        return 0.0
    
    tokens = tokenizer.encode(text)
    return len(tokens) / max(len(text), 1)


def compression_ratio(tokenizer: TokenizerProtocol, text: str) -> float:
    """
    Calculate compression ratio (bytes per token).
    
    Higher is better - indicates more bytes encoded per token.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
    
    Returns:
        Bytes per token
    """
    if not text:
        return 0.0
    
    tokens = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    
    return num_bytes / max(len(tokens), 1)


def compute_compression_metrics(
    tokenizer: TokenizerProtocol,
    texts: Union[str, List[str]],
    aggregate: bool = True
) -> Dict[str, Any]:
    """
    Compute all compression metrics for given texts.
    
    Args:
        tokenizer: Tokenizer with encode method
        texts: Single text or list of texts
        aggregate: If True, return aggregated stats; if False, return per-text
    
    Returns:
        Dictionary of compression metrics
    """
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    for text in texts:
        results.append({
            'tokens_per_byte': tokens_per_byte(tokenizer, text),
            'tokens_per_char': tokens_per_char(tokenizer, text),
            'compression_ratio': compression_ratio(tokenizer, text),
            'num_tokens': len(tokenizer.encode(text)),
            'num_chars': len(text),
            'num_bytes': len(text.encode('utf-8')),
        })
    
    if not aggregate:
        return {'per_text': results}
    
    # Aggregate statistics
    n = len(results)
    if n == 0:
        return {
            'tokens_per_byte': {'mean': 0, 'min': 0, 'max': 0},
            'tokens_per_char': {'mean': 0, 'min': 0, 'max': 0},
            'compression_ratio': {'mean': 0, 'min': 0, 'max': 0},
            'total_tokens': 0,
            'total_chars': 0,
            'total_bytes': 0,
        }
    
    def aggregate_metric(key: str) -> Dict[str, float]:
        values = [r[key] for r in results]
        return {
            'mean': sum(values) / n,
            'min': min(values),
            'max': max(values),
        }
    
    return {
        'tokens_per_byte': aggregate_metric('tokens_per_byte'),
        'tokens_per_char': aggregate_metric('tokens_per_char'),
        'compression_ratio': aggregate_metric('compression_ratio'),
        'total_tokens': sum(r['num_tokens'] for r in results),
        'total_chars': sum(r['num_chars'] for r in results),
        'total_bytes': sum(r['num_bytes'] for r in results),
        'num_samples': n,
    }


def compare_tokenizers(
    tokenizers: Dict[str, TokenizerProtocol],
    texts: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple tokenizers on the same texts.
    
    Args:
        tokenizers: Dictionary of name -> tokenizer
        texts: List of texts to evaluate
    
    Returns:
        Dictionary of tokenizer name -> metrics
    """
    results = {}
    for name, tokenizer in tokenizers.items():
        results[name] = compute_compression_metrics(tokenizer, texts)
    
    return results
