"""
Fertility Metrics - Measure tokens per word across languages.

Fertility (tokens per word) is a key metric for multilingual tokenizers,
especially for Indic languages where high fertility indicates poor support.
"""

import re
from typing import List, Dict, Any, Optional, Union
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
else:
    TokenizerProtocol = Any


# Language-specific word splitting patterns
WORD_PATTERNS = {
    'english': r'\b[a-zA-Z]+\b',
    'devanagari': r'[\u0900-\u097F]+',
    'tamil': r'[\u0B80-\u0BFF]+',
    'telugu': r'[\u0C00-\u0C7F]+',
    'kannada': r'[\u0C80-\u0CFF]+',
    'malayalam': r'[\u0D00-\u0D7F]+',
    'bengali': r'[\u0980-\u09FF]+',
    'gujarati': r'[\u0A80-\u0AFF]+',
    'gurmukhi': r'[\u0A00-\u0A7F]+',
    'oriya': r'[\u0B00-\u0B7F]+',
    'generic': r'\S+',  # Fallback: any non-whitespace
}


def word_tokenize(text: str, language: str = 'generic') -> List[str]:
    """
    Split text into words using language-specific patterns.
    
    Args:
        text: Input text
        language: Language/script identifier
    
    Returns:
        List of words
    """
    pattern = WORD_PATTERNS.get(language, WORD_PATTERNS['generic'])
    return re.findall(pattern, text)


def fertility(
    tokenizer: TokenizerProtocol,
    text: str,
    language: str = 'generic'
) -> float:
    """
    Calculate fertility (tokens per word) for text.
    
    Lower fertility is better - indicates words are not being
    excessively fragmented.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
        language: Language for word splitting
    
    Returns:
        Tokens per word ratio
    """
    words = word_tokenize(text, language)
    if not words:
        return 0.0
    
    tokens = tokenizer.encode(text)
    return len(tokens) / len(words)


def per_word_fertility(
    tokenizer: TokenizerProtocol,
    text: str,
    language: str = 'generic'
) -> List[Dict[str, Any]]:
    """
    Calculate fertility for each word individually.
    
    Useful for identifying which words are being fragmented.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
        language: Language for word splitting
    
    Returns:
        List of dicts with word, num_tokens, fertility
    """
    words = word_tokenize(text, language)
    results = []
    
    for word in words:
        tokens = tokenizer.encode(word)
        results.append({
            'word': word,
            'num_tokens': len(tokens),
            'fertility': len(tokens),  # Single word, so fertility = token count
        })
    
    return results


def per_language_fertility(
    tokenizer: TokenizerProtocol,
    corpus_dict: Dict[str, str],
    detect_language: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Calculate fertility metrics per language/script.
    
    Args:
        tokenizer: Tokenizer with encode method
        corpus_dict: Dictionary of language -> text
        detect_language: If True, try to auto-detect script in mixed text
    
    Returns:
        Dictionary of language -> fertility metrics
    """
    results = {}
    
    for language, text in corpus_dict.items():
        words = word_tokenize(text, language)
        if not words:
            results[language] = {'fertility': 0.0, 'num_words': 0, 'num_tokens': 0}
            continue
        
        tokens = tokenizer.encode(text)
        fert = len(tokens) / len(words)
        
        # Calculate per-word statistics
        word_fertilities = [len(tokenizer.encode(w)) for w in words]
        
        results[language] = {
            'fertility': fert,
            'num_words': len(words),
            'num_tokens': len(tokens),
            'min_word_fertility': min(word_fertilities) if word_fertilities else 0,
            'max_word_fertility': max(word_fertilities) if word_fertilities else 0,
            'mean_word_fertility': sum(word_fertilities) / len(word_fertilities) if word_fertilities else 0,
        }
    
    return results


def fertility_comparison(
    tokenizers: Dict[str, TokenizerProtocol],
    corpus_dict: Dict[str, str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compare fertility across multiple tokenizers and languages.
    
    Args:
        tokenizers: Dictionary of name -> tokenizer
        corpus_dict: Dictionary of language -> text
    
    Returns:
        Nested dict: tokenizer -> language -> metrics
    """
    results = {}
    
    for tok_name, tokenizer in tokenizers.items():
        results[tok_name] = per_language_fertility(tokenizer, corpus_dict)
    
    return results


def identify_high_fertility_words(
    tokenizer: TokenizerProtocol,
    text: str,
    language: str = 'generic',
    threshold: int = 3
) -> List[Dict[str, Any]]:
    """
    Identify words with fertility above a threshold.
    
    Useful for diagnosing tokenizer issues with specific words.
    
    Args:
        tokenizer: Tokenizer with encode method
        text: Input text
        language: Language for word splitting
        threshold: Fertility threshold
    
    Returns:
        List of high-fertility words with their details
    """
    word_data = per_word_fertility(tokenizer, text, language)
    return [w for w in word_data if w['fertility'] >= threshold]


def compute_fertility_metrics(
    tokenizer: TokenizerProtocol,
    texts: Union[str, List[str], Dict[str, str]],
    language: str = 'generic'
) -> Dict[str, Any]:
    """
    Compute comprehensive fertility metrics.
    
    Args:
        tokenizer: Tokenizer with encode method
        texts: Text, list of texts, or language->text dict
        language: Default language for word splitting
    
    Returns:
        Comprehensive fertility metrics
    """
    if isinstance(texts, str):
        texts = [texts]
    
    if isinstance(texts, dict):
        return per_language_fertility(tokenizer, texts)
    
    # List of texts - aggregate
    all_words = []
    all_tokens = 0
    fertilities = []
    
    for text in texts:
        words = word_tokenize(text, language)
        tokens = tokenizer.encode(text)
        all_words.extend(words)
        all_tokens += len(tokens)
        if words:
            fertilities.append(len(tokens) / len(words))
    
    if not all_words:
        return {'fertility': 0.0, 'num_words': 0, 'num_tokens': 0}
    
    return {
        'fertility': all_tokens / len(all_words),
        'num_words': len(all_words),
        'num_tokens': all_tokens,
        'mean_per_text_fertility': sum(fertilities) / len(fertilities) if fertilities else 0,
        'min_per_text_fertility': min(fertilities) if fertilities else 0,
        'max_per_text_fertility': max(fertilities) if fertilities else 0,
    }
