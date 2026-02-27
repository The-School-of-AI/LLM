"""
Combined Data Processing & Metrics Computation Glue Job

Combines Team 1 (data transformation) and Team 2 (metrics computation) into a single job.
Reads raw data once, processes for both outputs simultaneously.

Benefits:
- Read data only once (major I/O savings)
- Process in memory (no intermediate storage)
- Faster overall pipeline
- Lower cost (fewer DPU hours)

Outputs:
1. Team 1: Transformed parquet data (original format)
2. Team 2: Separate metrics parquet with rejection info
"""

import sys
import re
import zlib
import uuid
from typing import Dict, Tuple, Optional
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    FloatType, BooleanType, TimestampType
)

# ============================================================================
# GLUE JOB SETUP
# ============================================================================

# args = getResolvedOptions(
#     sys.argv,
#     [
#         "JOB_NAME",
#         "INPUT_PATH",              # s3://bucket/raw/dolma/*.json.gz (raw JSONL)
#         "TEAM1_OUTPUT_PATH",       # s3://bucket/parquet/dolma/ (transformed data)
#         "TEAM2_METRICS_PATH",      # s3://bucket/metrics/dolma/ (metrics)
#         "DOMAIN",                  # e.g. web
#         "EXTERNAL_SOURCE",         # e.g. books
#         "VERSION",                 # e.g. 1.7
#         "NUM_PARTITIONS",          # 400 (tune based on cluster)
#     ],
# )

sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init('t123_test')

# Configuration
INPUT_PATH = "s3://t1-dataacquisition-datasets/datasets_prod/huggingface_dolma/books/small.json.gzz"
TEAM1_OUTPUT = 's3://t1-dataacquisition-datasets/processed_dataset/h1/t1/'
TEAM2_METRICS = 's3://t1-dataacquisition-datasets/processed_dataset/h1/t2-3/'
DOMAIN = "web"
EXTERNAL_SOURCE = "c4"
VERSION = "1.7"
NUM_PARTITIONS = 4

# ============================================================================
# METRIC COMPUTATION FUNCTIONS (Team 2)
# ============================================================================

# Compiled regex patterns
PATTERNS = {
    'url': re.compile(r'https?://[^\s]+'),
    'sentence': re.compile(r'[.!?]+\s+'),
    'reasoning': re.compile(r'\b(therefore|thus|hence|because|since|consequently)\b', re.IGNORECASE),
    'math_expr': re.compile(r'[\$\^\{\}\\\[\]]|\\[a-zA-Z]+'),
    'step': re.compile(r'\b(step\s+\d+|first|second|third|next|finally)\b', re.IGNORECASE),
    'list_marker': re.compile(r'^\s*[\d\-\*\+]+[\.\)]\s+', re.MULTILINE),
    'truncation': re.compile(r'(\.\.\.|…|\[truncated\]|\[cut\])', re.IGNORECASE),
    'code_fence': re.compile(r'```'),
    'heading': re.compile(r'^#{1,6}\s+|\n={3,}|\n-{3,}', re.MULTILINE),
    'citation': re.compile(r'\[[0-9]+\]|\([A-Za-z]+\s+\d{4}\)'),
}

# Boilerplate markers for detecting low-quality web scrapes
BOILERPLATE_MARKERS = [
    'cookie policy', 'privacy policy', 'terms of service',
    'all rights reserved', '© copyright', 'click here',
    'subscribe to', 'sign up', 'newsletter', 'unsubscribe',
    'contact us', 'about us', 'follow us on',
    'accept cookies', 'manage preferences'
]

# Comment patterns for code detection (language-agnostic approach)
COMMENT_PATTERNS = {
    'single_line': re.compile(r'^\s*(?:#|//|--|%)', re.MULTILINE),  # Python, Java, C++, SQL, MATLAB
    'block_comment': re.compile(r'/\*.*?\*/|""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL),  # C-style, Python docstrings
    'html_comment': re.compile(r'<!--.*?-->', re.DOTALL)
}

# Thread/reply markers for detecting orphaned forum fragments
THREAD_MARKERS = [
    '>>',              # 4chan/8chan style: ">>12345"
    'replied to:',     # Forum style: "replied to: username"
    'in response to',  # Formal: "in response to the above comment"
    're:',             # Email/forum: "re: your post"
    'replying to',     # Twitter/Reddit: "replying to @user"
    'quote from',      # Quote indicators: "quote from original poster"
    'responding to',   # Another common variant
]

# URL spam detection patterns and lists
URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
    'buff.ly', 'adf.ly', 'bc.vc', 'shorte.st', 'clck.ru', 'short.link',
    'tiny.cc', 'lnkd.in', 'rebrand.ly', 'cutt.ly', 'bitly.com'
]

RISKY_TLDS = [
    '.tk', '.ml', '.ga', '.cf', '.gq',  # Free domains often used for spam
    '.xyz', '.top', '.club', '.work', '.info',  # High-risk TLDs
    '.loan', '.win', '.bid', '.download', '.stream',  # Spam-associated
    '.review', '.click', '.link', '.trade', '.date'
]

# Compiled patterns for URL spam detection
URL_SPAM_PATTERNS = {
    'ip_domain': re.compile(r'https?://(?:\d{1,3}\.){3}\d{1,3}'),  # IP as domain
    'at_symbol': re.compile(r'https?://[^/]*@'),  # Credential injection
    'extract_domain': re.compile(r'https?://([^/]+)'),  # Extract domain for analysis
}


def generate_uuid() -> str:
    """Generate UUID for metric record"""
    return str(uuid.uuid4())


def compute_basic_metrics(text: str) -> Dict:
    """PRIORITY 1 metrics - fastest, most fundamental checks"""
    if text is None or not isinstance(text, str):
        return {
            'byte_length': 0,
            'char_length': 0,
            'token_count_estimate': 0,
            'non_printable_ratio': 1.0,
            'line_count': 0,
        }
    
    byte_length = len(text.encode('utf-8'))
    char_length = len(text)
    line_count = text.count('\n') + 1
    
    non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) == 127)
    non_printable_ratio = non_printable / max(char_length, 1)
    token_count_estimate = max(1, char_length // 4)
    
    return {
        'byte_length': byte_length,
        'char_length': char_length,
        'token_count_estimate': token_count_estimate,
        'non_printable_ratio': round(non_printable_ratio, 6),
        'line_count': line_count,
    }


def check_priority1_rejection(metrics: Dict) -> Tuple[bool, Optional[str]]:
    """Check Priority 1 rejection criteria"""
    if metrics['byte_length'] < 50:
        return True, "byte_length too short (<50): lacks context"
    if metrics['byte_length'] > 1_000_000:
        return True, "byte_length too long (>1M): exceeds processing limits"
    if metrics['char_length'] < 20:
        return True, "char_length too short (<20): insufficient learning signal"
    if metrics['char_length'] > 500_000:
        return True, "char_length too long (>500K): exceeds single-pass processing"
    if metrics['token_count_estimate'] < 10:
        return True, "token_count too low (<10): noise/meaningless"
    if metrics['token_count_estimate'] > 128_000:
        return True, "token_count too high (>128K): exceeds context window"
    if metrics['non_printable_ratio'] > 0.01:
        return True, "non_printable_ratio too high (>1%): encoding corruption"
    return False, None


def is_emoji(char: str) -> bool:
    """Fast emoji detection using Unicode ranges"""
    code = ord(char)
    return (
        0x1F600 <= code <= 0x1F64F or  # Emoticons
        0x1F300 <= code <= 0x1F5FF or  # Misc symbols
        0x1F680 <= code <= 0x1F6FF or  # Transport
        0x1F900 <= code <= 0x1F9FF or  # Supplemental
        0x2600 <= code <= 0x26FF or    # Misc symbols
        0x2700 <= code <= 0x27BF or    # Dingbats
        0xFE00 <= code <= 0xFE0F       # Variation selectors
    )


def compute_low_effort_post_score(
    text: str,
    char_length: int,
    capitalization_ratio: float,
    sentence_count: int
) -> float:
    """
    Compute low-effort post score for conversational content (Reddit/forums).
    Composite score combining 4 indicators (threshold: >0.6)
    """
    if not text or char_length == 0:
        return 0.0
    
    # 1. Short with exclamations (0.3 weight)
    if char_length < 100:
        exclamation_count = text.count('!')
        short_exclamatory_ratio = 1.0 if exclamation_count > 2 else 0.0
    else:
        short_exclamatory_ratio = 0.0
    
    # 2. Emoji ratio (0.2 weight)
    try:
        emoji_count = sum(1 for c in text if is_emoji(c))
        emoji_ratio = min(emoji_count / max(char_length, 1), 1.0)
    except:
        emoji_ratio = 0.0
    
    # 3. All caps fragments (0.3 weight) - reuse existing metric
    all_caps_fragments = capitalization_ratio
    
    # 4. Single word sentences (0.2 weight)
    if sentence_count > 0:
        sentences = PATTERNS['sentence'].split(text)
        single_word_sentences = sum(
            1 for s in sentences if len(s.split()) <= 2 and len(s.strip()) > 0
        )
        single_word_ratio = single_word_sentences / max(sentence_count, 1)
    else:
        single_word_ratio = 0.0
    
    # Compute weighted composite score
    low_effort_score = (
        short_exclamatory_ratio * 0.3 +
        emoji_ratio * 0.2 +
        all_caps_fragments * 0.3 +
        single_word_ratio * 0.2
    )
    
    return low_effort_score


def compute_url_spam_score(text: str) -> Tuple[float, list]:
    """
    Compute URL spam score based on 6 research-backed indicators.
    Returns (score, indicators_list) where score is sum of all URL spam signals.
    
    Indicators (each adds to score):
    1. Long URLs (>200 chars): +1 per URL
    2. Excessive subdomains (>5 dots): +1.5 per URL
    3. IP address as domain: +2 per URL
    4. @ symbol (credential injection): +2 per URL
    5. URL shortener: +1 per URL
    6. High-risk TLD: +1.5 per URL
    """
    if not text:
        return 0.0, []
    
    urls = PATTERNS['url'].findall(text)
    if not urls:
        return 0.0, []
    
    total_score = 0.0
    indicators = []
    
    for url in urls:
        url_score = 0
        url_indicators = []
        
        # 1. Long URL (>200 chars)
        if len(url) > 200:
            url_score += 1
            url_indicators.append('LONG-URL')
        
        # 2. IP address as domain
        if URL_SPAM_PATTERNS['ip_domain'].match(url):
            url_score += 2
            url_indicators.append('IP-DOMAIN')
        
        # 3. @ symbol (credential injection)
        if URL_SPAM_PATTERNS['at_symbol'].match(url):
            url_score += 2
            url_indicators.append('@-INJECT')
        
        # Extract domain for further checks
        domain_match = URL_SPAM_PATTERNS['extract_domain'].search(url)
        if domain_match:
            domain = domain_match.group(1).lower()
            
            # 4. Excessive subdomains (>5 dots in domain)
            dot_count = domain.count('.')
            if dot_count > 5:
                url_score += 1.5
                url_indicators.append('SUBDOMAINS')
            
            # 5. URL shortener
            if any(shortener in domain for shortener in URL_SHORTENERS):
                url_score += 1
                url_indicators.append('SHORTENER')
            
            # 6. High-risk TLD
            if any(domain.endswith(tld) for tld in RISKY_TLDS):
                url_score += 1.5
                url_indicators.append('RISKY-TLD')
        
        if url_indicators:
            total_score += url_score
            indicators.extend(url_indicators)
    
    return total_score, list(set(indicators))  # Remove duplicates from indicators


def compute_lexical_metrics(text: str, char_length: int) -> Dict:
    """PRIORITY 2 metrics - lexical diversity and noise detection"""
    if not text or char_length == 0:
        return {
            'unique_token_ratio': 0.0,
            'vocab_size': 0,
            'compression_ratio': 0.0,
            'capitalization_ratio': 0.0,
            'whitespace_ratio': 0.0,
            'symbol_density': 0.0,
            'boilerplate_ratio': 0.0,
            'url_spam_score': 0.0,
            'url_spam_indicators': None,
            'low_effort_post_score': 0.0,
            'html_tag_density': 0.0,
            'thread_fragment_marker_count': 0,
        }
    
    tokens = text.split()
    token_count = len(tokens)
    unique_tokens = len(set(tokens)) if tokens else 0
    unique_token_ratio = unique_tokens / max(token_count, 1)
    
    try:
        compressed = zlib.compress(text.encode('utf-8'), level=6)
        compression_ratio = len(compressed) / max(len(text.encode('utf-8')), 1)
    except:
        compression_ratio = 0.5
    
    uppercase = sum(1 for c in text if c.isupper())
    whitespace = sum(1 for c in text if c.isspace())
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    
    capitalization_ratio = uppercase / max(char_length, 1)
    
    # Compute boilerplate ratio
    text_lower = text.lower()
    boilerplate_token_count = sum(
        text_lower.count(marker) * len(marker.split())
        for marker in BOILERPLATE_MARKERS
    )
    boilerplate_ratio = boilerplate_token_count / max(token_count, 1)
    
    # Compute URL spam score
    url_spam_score, url_spam_indicators = compute_url_spam_score(text)
    url_spam_indicators_str = ', '.join(url_spam_indicators) if url_spam_indicators else None
    
    # Compute sentence count for low_effort_post_score (reuse in Priority 2 check)
    sentence_count = len(PATTERNS['sentence'].split(text))
    
    # Compute low effort post score
    low_effort_post_score = compute_low_effort_post_score(
        text, char_length, capitalization_ratio, sentence_count
    )
    
    # Compute HTML tag density (fast count-based approach)
    html_tag_count = text.count('<')
    html_tag_density = html_tag_count / max(char_length, 1)
    
    # Compute thread fragment markers (fast count-based approach)
    text_lower = text.lower()
    thread_fragment_marker_count = sum(
        text_lower.count(marker) for marker in THREAD_MARKERS
    )
    
    return {
        'unique_token_ratio': round(unique_token_ratio, 6),
        'vocab_size': unique_tokens,
        'compression_ratio': round(compression_ratio, 6),
        'capitalization_ratio': round(capitalization_ratio, 6),
        'whitespace_ratio': round(whitespace / max(char_length, 1), 6),
        'symbol_density': round(symbols / max(char_length, 1), 6),
        'boilerplate_ratio': round(boilerplate_ratio, 6),
        'url_spam_score': round(url_spam_score, 6),
        'url_spam_indicators': url_spam_indicators_str,
        'low_effort_post_score': round(low_effort_post_score, 6),
        'html_tag_density': round(html_tag_density, 6),
        'thread_fragment_marker_count': thread_fragment_marker_count,
        'sentence_count_estimate': sentence_count,  # Store for reuse
    }


def check_priority2_rejection(metrics: Dict, text: str, token_count: int) -> Tuple[bool, Optional[str]]:
    """Check Priority 2 rejection criteria"""
    if metrics.get('unique_token_ratio', 1.0) < 0.1:
        return True, "unique_token_ratio too low (<0.1): template/repetitive content"
    if metrics.get('compression_ratio', 0.0) > 0.95:
        return True, "compression_ratio too high (>0.95): random/encrypted/binary data"
    if metrics.get('capitalization_ratio', 0.0) > 0.5:
        return True, "capitalization_ratio too high (>50%): ALL CAPS spam/shouting"
    if metrics.get('whitespace_ratio', 0.0) > 0.6:
        return True, "whitespace_ratio too high (>60%): mostly empty/formatting artifacts"
    if metrics.get('boilerplate_ratio', 0.0) > 0.15:
        return True, f"boilerplate_ratio too high (>15%): navigation menus/cookie notices/footer spam (ratio={metrics.get('boilerplate_ratio', 0.0):.2%})"
    if metrics.get('url_spam_score', 0.0) > 7:
        indicators = metrics.get('url_spam_indicators', 'unknown')
        return True, f"url_spam_score too high (>7): malicious URL pattern detected (score={metrics.get('url_spam_score', 0.0):.2f}) with indicators: [{indicators}]"
    if metrics.get('low_effort_post_score', 0.0) > 0.6:
        return True, f"low_effort_post_score too high (>0.6): short, low-signal conversational fragments (score={metrics.get('low_effort_post_score', 0.0):.2f})"
    if metrics.get('html_tag_density', 0.0) > 0.05:
        return True, f"html_tag_density too high (>5%): raw HTML dump or markup-heavy document (density={metrics.get('html_tag_density', 0.0):.2%})"
    
    # Thread fragment check: orphaned replies without context
    thread_marker_count = metrics.get('thread_fragment_marker_count', 0)
    if thread_marker_count > 2 and token_count < 200:
        return True, f"thread_fragment_indicator detected (>{thread_marker_count} markers, <200 tokens): orphaned thread fragment without context"
    
    truncation_count = len(PATTERNS['truncation'].findall(text))
    if truncation_count > 2:
        return True, f"truncation_indicators too high (>2): incomplete content ({truncation_count} signals)"
    
    # Reuse sentence_count from lexical metrics if available
    sentence_count = metrics.get('sentence_count_estimate', len(PATTERNS['sentence'].split(text)))
    if sentence_count < 2 and token_count > 100:
        return True, "sentence_count low (<2) with high token_count: parsing failure"
    
    noise_score = (
        metrics.get('capitalization_ratio', 0.0) * 0.3 +
        metrics.get('whitespace_ratio', 0.0) * 0.3 +
        (1.0 - metrics.get('unique_token_ratio', 1.0)) * 0.2 +
        metrics.get('non_printable_ratio', 0.0) * 0.2
    )
    
    if noise_score > 0.6:
        return True, f"noise_score too high (>0.6): low-quality content (score={noise_score:.3f})"
    
    metrics['truncation_indicators'] = truncation_count
    metrics['sentence_count_estimate'] = sentence_count
    metrics['noise_score'] = round(noise_score, 6)
    
    return False, None


def compute_code_comment_ratio(text: str) -> float:
    """Compute ratio of comment lines to total lines (for code detection)"""
    if not text:
        return 0.0
    
    lines = text.split('\n')
    if not lines:
        return 0.0
    
    total_lines = len(lines)
    comment_lines = 0
    
    # Remove block comments first to avoid double counting
    text_no_blocks = COMMENT_PATTERNS['block_comment'].sub('', text)
    text_no_blocks = COMMENT_PATTERNS['html_comment'].sub('', text_no_blocks)
    
    # Count single-line comments
    lines_no_blocks = text_no_blocks.split('\n')
    for line in lines_no_blocks:
        stripped = line.strip()
        if stripped and COMMENT_PATTERNS['single_line'].match(line):
            comment_lines += 1
    
    # Count lines removed by block comment removal
    block_comment_lines = total_lines - len(lines_no_blocks)
    comment_lines += max(0, block_comment_lines)
    
    return comment_lines / max(total_lines, 1)


def compute_structural_metrics(text: str, sentence_count: int, char_length: int) -> Dict:
    """PRIORITY 3 metrics - complex structural analysis"""
    if not text or char_length == 0:
        return {
            'avg_line_length': 0.0,
            'avg_sentence_length': 0.0,
            'punctuation_density': 0.0,
            'avg_word_length': 0.0,
            'code_comment_ratio': 0.0,
        }
    
    line_count = text.count('\n') + 1
    avg_line_length = char_length / max(line_count, 1)
    avg_sentence_length = char_length / max(sentence_count, 1)
    
    punctuation = sum(1 for c in text if c in '.,;:!?')
    punctuation_density = punctuation / max(char_length, 1)
    
    words = text.split()
    avg_word_length = sum(len(w) for w in words) / max(len(words), 1) if words else 0.0
    
    # Compute code comment ratio
    code_comment_ratio = compute_code_comment_ratio(text)
    
    return {
        'avg_line_length': round(avg_line_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'punctuation_density': round(punctuation_density, 6),
        'avg_word_length': round(avg_word_length, 2),
        'code_comment_ratio': round(code_comment_ratio, 6),
    }


def compute_pattern_metrics(text: str, token_count: int) -> Dict:
    """Pattern-based metrics"""
    if not text:
        return {
            'url_count': 0,
            'question_density': 0.0,
            'citation_count': 0,
            'reasoning_marker_density': 0.0,
            'math_expression_count': 0,
            'step_indicator_count': 0,
            'list_marker_count': 0,
            'code_block_count': 0,
            'heading_count': 0,
        }
    
    url_count = len(PATTERNS['url'].findall(text))
    questions = text.count('?')
    
    return {
        'url_count': url_count,
        'question_density': round(questions / max(token_count, 1), 6),
        'citation_count': len(PATTERNS['citation'].findall(text)),
        'reasoning_marker_density': round(len(PATTERNS['reasoning'].findall(text)) / max(token_count, 1), 6),
        'math_expression_count': len(PATTERNS['math_expr'].findall(text)),
        'step_indicator_count': len(PATTERNS['step'].findall(text)),
        'list_marker_count': len(PATTERNS['list_marker'].findall(text)),
        'code_block_count': len(PATTERNS['code_fence'].findall(text)),
        'heading_count': len(PATTERNS['heading'].findall(text)),
    }


def check_priority3_rejection(metrics: Dict, text: str, token_count: int) -> Tuple[bool, Optional[str]]:
    """Check Priority 3 rejection criteria"""
    if metrics.get('avg_sentence_length', 0.0) > 500:
        return True, "avg_sentence_length too high (>500): run-on sentences/parsing errors"
    
    url_ratio = metrics.get('url_count', 0) / max(token_count, 1)
    if url_ratio > 0.3:
        return True, f"url_ratio too high (>0.3): link spam/scraping artifacts ({url_ratio:.2%})"
    
    if metrics.get('code_comment_ratio', 0.0) > 0.8:
        return True, f"code_comment_ratio too high (>80%): mostly comments/TODOs, not actual code (ratio={metrics.get('code_comment_ratio', 0.0):.2%})"
    
    sentences = max(metrics.get('sentence_count_estimate', 1), 1)
    words = max(token_count, 1)
    syllables = words * 1.5
    flesch = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
    metrics['flesch_reading_ease'] = round(flesch, 2)
    
    if flesch < 0 or flesch > 120:
        return True, f"flesch_reading_ease out of range (0-120): calculation error (score={flesch:.1f})"
    
    max_depth = 0
    current_depth = 0
    for c in text[:10000]:
        if c in '([{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif c in ')]}':
            current_depth = max(0, current_depth - 1)
    
    metrics['dependency_depth_estimate'] = max_depth
    if max_depth > 20:
        return True, f"dependency_depth too high (>20): malformed code/data corruption (depth={max_depth})"
    
    sentences_text = PATTERNS['sentence'].split(text)
    valid_endings = sum(1 for s in sentences_text[:100] if len(s.strip()) > 5)
    coherence = valid_endings / max(len(sentences_text[:100]), 1)
    metrics['sentence_boundary_coherence'] = round(coherence, 6)
    
    if coherence < 0.5:
        return True, f"sentence_boundary_coherence too low (<0.5): parsing/extraction failures (score={coherence:.2f})"
    
    alpha_chars = sum(1 for c in text if c.isalpha())
    info_density = alpha_chars / max(len(text), 1)
    metrics['information_density'] = round(info_density, 6)
    
    if info_density < 0.2:
        return True, f"information_density too low (<0.2): mostly filler/function words (density={info_density:.2%})"
    
    return False, None


def compute_derived_metrics(all_metrics: Dict) -> Dict:
    """Compute derived/composite metrics"""
    structural_score = (
        min(all_metrics.get('sentence_count_estimate', 0) / 100, 1.0) * 0.3 +
        min(all_metrics.get('avg_sentence_length', 0) / 100, 1.0) * 0.2 +
        min(all_metrics.get('dependency_depth_estimate', 0) / 10, 1.0) * 0.3 +
        all_metrics.get('symbol_density', 0.0) * 0.2
    )
    
    code_score = (
        all_metrics.get('code_block_count', 0) * 0.4 +
        all_metrics.get('symbol_density', 0.0) * 100 * 0.3 +
        all_metrics.get('avg_line_length', 0) / 100 * 0.3
    )
    
    math_score = all_metrics.get('math_expression_count', 0) * 0.6
    dialogue_score = all_metrics.get('question_density', 0.0) * 1000 * 0.5
    
    domain_scores = {
        'code': code_score,
        'math': math_score,
        'dialogue': dialogue_score,
        'general': 1.0
    }
    domain_signal = max(domain_scores, key=domain_scores.get)
    
    return {
        'structural_complexity_score': round(structural_score, 6),
        'domain_signal': domain_signal,
    }


def process_record_with_early_rejection(
    record_id: str,
    text: str,
    source_file: str
) -> Dict:
    """Main processing function with early rejection optimization"""
    result = {
        'metric_record_uuid': generate_uuid(),
        'source_record_id': record_id,
        'source_file_path': source_file,
        'is_rejected': False,
        'rejection_reason': None,
    }
    
    # Priority 1
    basic_metrics = compute_basic_metrics(text)
    result.update(basic_metrics)
    
    is_rejected, reason = check_priority1_rejection(basic_metrics)
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P1] {reason}"
        return result
    
    # Priority 2
    lexical_metrics = compute_lexical_metrics(text, basic_metrics['char_length'])
    result.update(lexical_metrics)
    
    is_rejected, reason = check_priority2_rejection(
        result, text, basic_metrics['token_count_estimate']
    )
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P2] {reason}"
        return result
    
    # Priority 3
    structural_metrics = compute_structural_metrics(
        text, result['sentence_count_estimate'], basic_metrics['char_length']
    )
    result.update(structural_metrics)
    
    is_rejected, reason = check_priority3_rejection(
        result, text, basic_metrics['token_count_estimate']
    )
    if is_rejected:
        result['is_rejected'] = True
        result['rejection_reason'] = f"[P3] {reason}"
        return result
    
    # Non-rejection metrics
    pattern_metrics = compute_pattern_metrics(text, basic_metrics['token_count_estimate'])
    result.update(pattern_metrics)
    
    derived_metrics = compute_derived_metrics(result)
    result.update(derived_metrics)
    
    # Add placeholder metrics
    result.update({
        'mtld': None, 'fertility': None, 'script_distribution': None,
        'code_language_hint': None, 'rare_word_ratio': None,
        'num_numeric_tokens': None, 'num_entities_estimate': None,
        'ellipsis_count': None, 'table_count_estimate': None,
        'dialogue_turn_count': None, 'visual_placeholder_count': None,
        'equation_density': None, 'table_complexity': None,
        'few_shot_potential': None, 'cross_domain_analogy_markers': None,
        'domain_specificity': None, 'concept_density': None,
        'example_density': None, 'prerequisite_density': None,
        'hedging_language_ratio': None, 'counterargument_presence': None,
        'instruction_complexity': None,
    })
    
    return result


# ============================================================================
# SPARK UDF FOR METRICS (Team 2)
# ============================================================================

metrics_schema = StructType([
    StructField("metric_record_uuid", StringType(), False),
    StructField("source_record_id", StringType(), False),
    StructField("source_file_path", StringType(), False),
    StructField("is_rejected", BooleanType(), False),
    StructField("rejection_reason", StringType(), True),
    StructField("byte_length", IntegerType(), True),
    StructField("char_length", IntegerType(), True),
    StructField("token_count_estimate", IntegerType(), True),
    StructField("non_printable_ratio", FloatType(), True),
    StructField("line_count", IntegerType(), True),
    StructField("unique_token_ratio", FloatType(), True),
    StructField("vocab_size", IntegerType(), True),
    StructField("compression_ratio", FloatType(), True),
    StructField("capitalization_ratio", FloatType(), True),
    StructField("whitespace_ratio", FloatType(), True),
    StructField("symbol_density", FloatType(), True),
    StructField("boilerplate_ratio", FloatType(), True),
    StructField("url_spam_score", FloatType(), True),
    StructField("url_spam_indicators", StringType(), True),
    StructField("low_effort_post_score", FloatType(), True),
    StructField("html_tag_density", FloatType(), True),
    StructField("thread_fragment_marker_count", IntegerType(), True),
    StructField("truncation_indicators", IntegerType(), True),
    StructField("sentence_count_estimate", IntegerType(), True),
    StructField("noise_score", FloatType(), True),
    StructField("avg_line_length", FloatType(), True),
    StructField("avg_sentence_length", FloatType(), True),
    StructField("punctuation_density", FloatType(), True),
    StructField("avg_word_length", FloatType(), True),
    StructField("code_comment_ratio", FloatType(), True),
    StructField("flesch_reading_ease", FloatType(), True),
    StructField("dependency_depth_estimate", IntegerType(), True),
    StructField("sentence_boundary_coherence", FloatType(), True),
    StructField("information_density", FloatType(), True),
    StructField("url_count", IntegerType(), True),
    StructField("question_density", FloatType(), True),
    StructField("citation_count", IntegerType(), True),
    StructField("reasoning_marker_density", FloatType(), True),
    StructField("math_expression_count", IntegerType(), True),
    StructField("step_indicator_count", IntegerType(), True),
    StructField("list_marker_count", IntegerType(), True),
    StructField("code_block_count", IntegerType(), True),
    StructField("heading_count", IntegerType(), True),
    StructField("structural_complexity_score", FloatType(), True),
    StructField("domain_signal", StringType(), True),
    StructField("mtld", FloatType(), True),
    StructField("fertility", FloatType(), True),
    StructField("script_distribution", StringType(), True),
    StructField("code_language_hint", StringType(), True),
    StructField("rare_word_ratio", FloatType(), True),
    StructField("num_numeric_tokens", IntegerType(), True),
    StructField("num_entities_estimate", IntegerType(), True),
    StructField("ellipsis_count", IntegerType(), True),
    StructField("table_count_estimate", IntegerType(), True),
    StructField("dialogue_turn_count", IntegerType(), True),
    StructField("visual_placeholder_count", IntegerType(), True),
    StructField("equation_density", FloatType(), True),
    StructField("table_complexity", FloatType(), True),
    StructField("few_shot_potential", FloatType(), True),
    StructField("cross_domain_analogy_markers", IntegerType(), True),
    StructField("domain_specificity", FloatType(), True),
    StructField("concept_density", FloatType(), True),
    StructField("example_density", FloatType(), True),
    StructField("prerequisite_density", FloatType(), True),
    StructField("hedging_language_ratio", FloatType(), True),
    StructField("counterargument_presence", BooleanType(), True),
    StructField("instruction_complexity", FloatType(), True),
])


@F.udf(returnType=metrics_schema)
def compute_metrics_udf(record_id: str, text: str, source_file: str):
    """Spark UDF wrapper for metrics computation"""
    return process_record_with_early_rejection(record_id, text, source_file)


# ============================================================================
# MAIN PROCESSING - COMBINED PIPELINE
# ============================================================================

def main():
    """Main Glue job - processes data for both Team 1 and Team 2"""
    
    print("=" * 80)
    print("COMBINED DATA PROCESSING & METRICS COMPUTATION JOB")
    print("=" * 80)
    
    # Define schema for raw input
    input_schema = (
        StructType()
        .add("id", StringType())
        .add("text", StringType())
        .add("metadata", StringType())
        .add("added", TimestampType())
        .add("created", TimestampType())
    )
    
    # ========== READ RAW DATA (Once) ==========
    print(f"\n📥 Reading raw data from: {INPUT_PATH}")
    
    df_raw = (
        spark.read
        .schema(input_schema)
        .option("compression", "gzip")
        .json(INPUT_PATH)
    )
    
    # Add input file path (needed for Team 2)
    df_raw = df_raw.withColumn("input_file_path", F.input_file_name())
    
    # Cache the dataframe since we'll use it for both outputs
    df_raw.cache()
    
    record_count = df_raw.count()
    print(f"✓ Records read: {record_count:,}")
    
    # ========== TEAM 1: Transform and Write Main Data ==========
    print(f"\n🔄 Team 1: Transforming data...")
    
    df_team1 = (
        df_raw
        .withColumn("hash", F.sha2(F.col("text"), 256))
        .withColumn("dataset", F.lit("dolma"))
        .withColumn("domain", F.lit(DOMAIN))
        .withColumn("source", F.lit(EXTERNAL_SOURCE))
        .withColumn("language", F.lit("en"))
        .withColumn("metadata", F.col("metadata").cast("string"))
        .withColumn("version", F.lit(VERSION))
        .select(
            "id", "hash", "dataset", "domain", "source",
            "text", "language", "metadata", "added", "created", "version"
        )
    )
    
    print(f"📤 Team 1: Writing to {TEAM1_OUTPUT}")
    
    (
        df_team1
        .repartition(NUM_PARTITIONS)
        .write
        .mode("overwrite")
        .option("compression", "zstd")
        .parquet(TEAM1_OUTPUT)
    )
    
    print("✅ Team 1: Data transformation complete!")
    
    # ========== TEAM 2: Compute Metrics and Write ==========
    print(f"\n📊 Team 2: Computing metrics with early rejection...")
    
    df_metrics = df_raw.select(
        compute_metrics_udf(
            F.col("id"),
            F.col("text"),
            F.col("input_file_path")
        ).alias("metrics")
    ).select("metrics.*")
    
    # Add processing timestamp
    df_metrics = df_metrics.withColumn("processed_at", F.current_timestamp())
    
    # Show statistics
    print("\n📈 Rejection Statistics:")
    rejection_stats = df_metrics.groupBy("is_rejected").count().collect()
    
    total = sum(row['count'] for row in rejection_stats)
    for row in rejection_stats:
        status = "Rejected" if row['is_rejected'] else "Accepted"
        count = row['count']
        pct = (count / total * 100) if total > 0 else 0
        print(f"   {status}: {count:,} ({pct:.1f}%)")
    
    print("\n🔝 Top 10 Rejection Reasons:")
    rejection_reasons = (
        df_metrics
        .filter(F.col("is_rejected") == True)
        .groupBy("rejection_reason")
        .count()
        .orderBy(F.desc("count"))
        .limit(10)
    )
    
    for row in rejection_reasons.collect():
        print(f"   • {row['rejection_reason']}: {row['count']:,}")
    
    print(f"\n📤 Team 2: Writing metrics to {TEAM2_METRICS}")
    
    (
        df_metrics
        .repartition(NUM_PARTITIONS)
        .write
        .mode("overwrite")
        .option("compression", "zstd")
        .parquet(TEAM2_METRICS)
    )
    
    print("✅ Team 2: Metrics computation complete!")
    
    # Unpersist cache
    df_raw.unpersist()
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("JOB COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"Input:  {INPUT_PATH}")
    print(f"Team 1: {TEAM1_OUTPUT}")
    print(f"Team 2: {TEAM2_METRICS}")
    print(f"Records: {record_count:,}")
    print("=" * 80)
    
    job.commit()


if __name__ == "__main__":
    main()
