
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


def test():
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


"""

'boilerplate_ratio': round(boilerplate_ratio, 6),
'url_spam_score': round(url_spam_score, 6),
'url_spam_indicators': url_spam_indicators_str,
'low_effort_post_score': round(low_effort_post_score, 6),
'html_tag_density': round(html_tag_density, 6),
'thread_fragment_marker_count': thread_fragment_marker_count,
'sentence_count_estimate': sentence_count,  # Store for reuse


"""