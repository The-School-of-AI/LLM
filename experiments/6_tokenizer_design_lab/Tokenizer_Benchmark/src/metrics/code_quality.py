"""
Code Quality Metrics - Measure tokenizer behavior on programming languages.

Metrics:
- Keyword preservation (are keywords single tokens?)
- Identifier handling (reasonable splitting of variable names)
- Whitespace efficiency
"""

from typing import List, Dict, Any, Set
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
        def decode(self, token_ids: List[int]) -> str: ...
else:
    TokenizerProtocol = Any


# Language keywords that should ideally be single tokens
LANGUAGE_KEYWORDS = {
    'python': {
        'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
        'import', 'from', 'try', 'except', 'finally', 'with', 'as',
        'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
        'lambda', 'yield', 'raise', 'pass', 'break', 'continue',
        'async', 'await', 'global', 'nonlocal', 'assert', 'del',
    },
    'javascript': {
        'if', 'else', 'for', 'while', 'do', 'function', 'return',
        'const', 'let', 'var', 'class', 'new', 'this', 'super',
        'true', 'false', 'null', 'undefined', 'typeof', 'instanceof',
        'try', 'catch', 'finally', 'throw', 'async', 'await',
        'import', 'export', 'default', 'from', 'switch', 'case',
        'break', 'continue', 'yield', 'delete', 'void', 'in', 'of',
    },
    'java': {
        'if', 'else', 'for', 'while', 'do', 'class', 'interface',
        'public', 'private', 'protected', 'static', 'final', 'void',
        'return', 'new', 'this', 'super', 'extends', 'implements',
        'try', 'catch', 'finally', 'throw', 'throws', 'import',
        'package', 'abstract', 'synchronized', 'volatile', 'transient',
        'true', 'false', 'null', 'instanceof', 'enum', 'assert',
    },
    'cpp': {
        'if', 'else', 'for', 'while', 'do', 'class', 'struct',
        'public', 'private', 'protected', 'virtual', 'override',
        'return', 'new', 'delete', 'this', 'namespace', 'using',
        'try', 'catch', 'throw', 'template', 'typename', 'const',
        'static', 'inline', 'extern', 'volatile', 'auto', 'nullptr',
        'true', 'false', 'sizeof', 'typedef', 'enum', 'union',
    },
}

# Common operators and syntax
OPERATORS = {
    '==', '!=', '<=', '>=', '&&', '||', '++', '--',
    '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=',
    '->', '=>', '::', '...', '===', '!==',
}


def keyword_preservation_score(
    tokenizer: TokenizerProtocol,
    language: str = 'python'
) -> Dict[str, Any]:
    """
    Measure how well keywords are preserved as single tokens.
    
    A good tokenizer should encode language keywords as single tokens,
    not split them into subwords.
    
    Args:
        tokenizer: Tokenizer to evaluate
        language: Programming language
    
    Returns:
        Score and details of keyword tokenization
    """
    keywords = LANGUAGE_KEYWORDS.get(language, LANGUAGE_KEYWORDS['python'])
    
    preserved = []
    split = []
    
    for keyword in keywords:
        tokens = tokenizer.encode(keyword)
        if len(tokens) == 1:
            preserved.append(keyword)
        else:
            split.append({
                'keyword': keyword,
                'num_tokens': len(tokens),
            })
    
    total = len(keywords)
    score = len(preserved) / total if total > 0 else 0.0
    
    return {
        'score': score,
        'preserved_count': len(preserved),
        'split_count': len(split),
        'total_keywords': total,
        'preserved': preserved,
        'split': split,
    }


def operator_preservation_score(tokenizer: TokenizerProtocol) -> Dict[str, Any]:
    """
    Measure how well operators are preserved as single tokens.
    
    Args:
        tokenizer: Tokenizer to evaluate
    
    Returns:
        Score and details
    """
    preserved = []
    split = []
    
    for op in OPERATORS:
        tokens = tokenizer.encode(op)
        if len(tokens) == 1:
            preserved.append(op)
        else:
            split.append({
                'operator': op,
                'num_tokens': len(tokens),
            })
    
    total = len(OPERATORS)
    score = len(preserved) / total if total > 0 else 0.0
    
    return {
        'score': score,
        'preserved_count': len(preserved),
        'split_count': len(split),
        'total_operators': total,
    }


def identifier_quality_score(
    tokenizer: TokenizerProtocol,
    identifiers: List[str] = None
) -> Dict[str, Any]:
    """
    Measure tokenizer behavior on identifiers.
    
    Good tokenizers should split identifiers at reasonable boundaries
    (e.g., camelCase, snake_case) rather than arbitrary positions.
    
    Args:
        tokenizer: Tokenizer to evaluate
        identifiers: List of identifiers to test (uses defaults if None)
    
    Returns:
        Quality metrics for identifier handling
    """
    if identifiers is None:
        identifiers = [
            # camelCase
            'getUserName', 'processDataItem', 'calculateTotalAmount',
            'isValidInput', 'handleButtonClick', 'fetchApiResponse',
            # snake_case
            'get_user_name', 'process_data_item', 'calculate_total_amount',
            'is_valid_input', 'handle_button_click', 'fetch_api_response',
            # PascalCase
            'UserProfile', 'DataProcessor', 'HttpClient', 'DatabaseConnection',
            # SCREAMING_SNAKE_CASE
            'MAX_VALUE', 'API_BASE_URL', 'DEFAULT_TIMEOUT', 'ERROR_MESSAGES',
            # Mixed
            'XMLHttpRequest', 'getElementById', 'innerHTML', 'JSONParser',
        ]
    
    results = []
    total_tokens = 0
    
    for ident in identifiers:
        tokens = tokenizer.encode(ident)
        results.append({
            'identifier': ident,
            'num_tokens': len(tokens),
            'tokens_per_char': len(tokens) / len(ident),
        })
        total_tokens += len(tokens)
    
    # Calculate average tokens per identifier
    avg_tokens = total_tokens / len(identifiers) if identifiers else 0
    
    # Score: lower average is better (more efficient)
    # Normalize: 1 token = 1.0, 5+ tokens = 0.0
    score = max(0.0, 1.0 - (avg_tokens - 1) / 4)
    
    return {
        'score': score,
        'avg_tokens_per_identifier': avg_tokens,
        'total_identifiers': len(identifiers),
        'details': results,
    }


def whitespace_efficiency(
    tokenizer: TokenizerProtocol,
    indent_sizes: List[int] = None
) -> Dict[str, Any]:
    """
    Measure tokenizer efficiency on whitespace/indentation.
    
    Args:
        tokenizer: Tokenizer to evaluate
        indent_sizes: List of indentation sizes to test
    
    Returns:
        Whitespace tokenization metrics
    """
    if indent_sizes is None:
        indent_sizes = [2, 4, 8, 12, 16]
    
    results = []
    
    for size in indent_sizes:
        spaces = ' ' * size
        tokens = tokenizer.encode(spaces)
        results.append({
            'spaces': size,
            'tokens': len(tokens),
            'efficiency': size / len(tokens) if tokens else 0,  # spaces per token
        })
    
    # Also test tabs
    for num_tabs in [1, 2, 4]:
        tabs = '\t' * num_tabs
        tokens = tokenizer.encode(tabs)
        results.append({
            'tabs': num_tabs,
            'tokens': len(tokens),
            'efficiency': num_tabs / len(tokens) if tokens else 0,
        })
    
    return {
        'results': results,
        'avg_spaces_per_token': sum(r.get('efficiency', 0) for r in results if 'spaces' in r) / len(indent_sizes),
    }


def compute_code_quality_metrics(
    tokenizer: TokenizerProtocol,
    languages: List[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive code quality metrics.
    
    Args:
        tokenizer: Tokenizer to evaluate
        languages: List of languages to test (defaults to Python, JS)
    
    Returns:
        Complete code quality assessment
    """
    if languages is None:
        languages = ['python', 'javascript']
    
    # Keyword preservation per language
    keyword_scores = {}
    for lang in languages:
        keyword_scores[lang] = keyword_preservation_score(tokenizer, lang)
    
    # Overall keyword score (average)
    avg_keyword_score = sum(
        keyword_scores[lang]['score'] for lang in languages
    ) / len(languages)
    
    return {
        'keyword_scores': keyword_scores,
        'avg_keyword_score': avg_keyword_score,
        'operator_score': operator_preservation_score(tokenizer),
        'identifier_score': identifier_quality_score(tokenizer),
        'whitespace_efficiency': whitespace_efficiency(tokenizer),
    }
