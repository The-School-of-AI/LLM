import math
import re
import zlib
from collections import Counter
from statistics import mean, pstdev
from typing import Dict, List

# ------------------------
# Regex buckets
# ------------------------

RE_REASONING = re.compile(r"\b(therefore|hence|implies|assume|thus|because)\b", re.I)
RE_STEPS = re.compile(r"\b(step\s+\d+|first,|second,|third,)\b", re.I)
RE_AGENT = re.compile(r"\b(tool_call|observation|action|final_answer)\b", re.I)
RE_API = re.compile(r"\b(function|class|import|def|return|api)\b", re.I)
RE_PEDAGOGY = re.compile(r"\b(explain|example|exercise|solution)\b", re.I)

RE_HEADER = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
RE_BULLET = re.compile(r"^\s*[-*+]\s+", re.M)
RE_CODE_BLOCK = re.compile(r"```")
RE_INLINE_CODE = re.compile(r"`[^`]+`")
RE_JSON_LIKE = re.compile(r'\{\s*"[^"]+"\s*:')

MATH_SYMBOLS = set("=<>±*/^∑√≈≠≤≥")

VOWELS = re.compile(r"[aeiouy]+", re.I)

CODE_TOKENS = {
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ";",
    "::",
    "==",
    "!=",
    "<=",
    ">=",
    "+=",
    "-=",
    "*=",
    "/=",
    "->",
    "=>",
}

RE_CITATION = re.compile(r"\[\d+\]|\([A-Z][a-z]+ et al\.,? \d{4}\)|doi:|arxiv:", re.I)

RE_REFERENCES_HEADER = re.compile(r"^\s*(references|bibliography)\s*$", re.I | re.M)


# ------------------------
# Helpers
# ------------------------


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def split_sentences(text: str) -> List[str]:
    # Deliberately simple sentence splitter; precision not required for banding
    return re.split(r"[.!?]\s+", text)


# ------------------------
# Feature extractors
# ------------------------


def extract_basic_shape(text: str) -> Dict:
    return {
        "char_count": len(text),
        "line_count": text.count("\n") + 1,
        "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
    }


def extract_sentence_stats(text: str) -> Dict:
    sentences = [s for s in split_sentences(text) if s.strip()]
    tokenized = [simple_tokenize(s) for s in sentences]

    lengths = [len(t) for t in tokenized] or [0]
    lengths_sorted = sorted(lengths)
    p95 = lengths_sorted[int(0.95 * len(lengths_sorted))] if lengths else 0

    return {
        "sentence_count": len(sentences),
        "sentence_len_avg": mean(lengths),
        "sentence_len_std": pstdev(lengths) if len(lengths) > 1 else 0.0,
        "sentence_len_max": max(lengths),
        "sentence_len_p95": p95,
    }


def extract_structural(text: str, tokens: List[str]) -> Dict:
    bullet_lines = RE_BULLET.findall(text)
    code_blocks = len(RE_CODE_BLOCK.findall(text)) // 2

    math_count = sum(1 for t in tokens if t in MATH_SYMBOLS)

    return {
        "header_count": len(RE_HEADER.findall(text)),
        "bullet_count": len(bullet_lines),
        "bullet_depth_max": max((line.count("  ") for line in bullet_lines), default=0),
        "code_block_count": code_blocks,
        "inline_code_ratio": len(RE_INLINE_CODE.findall(text)) / max(len(tokens), 1),
        "json_like_ratio": len(RE_JSON_LIKE.findall(text)) / max(len(tokens), 1),
        "math_symbol_ratio": math_count / max(len(tokens), 1),
    }


def extract_lexical(tokens: List[str]) -> Dict:
    token_count = len(tokens)
    if token_count == 0:
        return {}

    counts = Counter(tokens)
    types = len(counts)
    rare_tokens = sum(1 for c in counts.values() if c == 1)

    uppercase = sum(1 for t in tokens if t.isupper())
    numeric = sum(1 for t in tokens if t.isdigit())

    return {
        "doc_token_count": token_count,
        "unique_token_ratio": types / token_count,
        "rare_token_ratio": rare_tokens / token_count,
        "uppercase_ratio": uppercase / token_count,
        "numeric_ratio": numeric / token_count,
        "token_entropy": compute_token_entropy(tokens),
        "code_token_ratio": compute_code_token_ratio(tokens),
        "repetition_score": compute_repetition_score(tokens),
    }


def extract_semantic_flags(text: str) -> Dict:
    return {
        "has_reasoning_markers": bool(RE_REASONING.search(text)),
        "has_step_markers": bool(RE_STEPS.search(text)),
        "has_agent_markers": bool(RE_AGENT.search(text)),
        "has_api_terms": bool(RE_API.search(text)),
        "has_pedagogy_markers": bool(RE_PEDAGOGY.search(text)),
    }


def extract_compression_ratio(text: str) -> float:
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def compute_token_entropy(tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    probs = [c / total for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def count_syllables(word: str) -> int:
    word = word.lower()
    syllables = VOWELS.findall(word)
    return max(1, len(syllables))


def compute_flesch_kincaid(text: str, tokens: List[str], sentence_count: int) -> float:
    words = [t for t in tokens if t.isalpha()]
    if not words or sentence_count == 0:
        return 0.0

    syllables = sum(count_syllables(w) for w in words)

    return (
        0.39 * (len(words) / sentence_count) + 11.8 * (syllables / len(words)) - 15.59
    )


def compute_code_token_ratio(tokens: List[str]) -> float:
    if not tokens:
        return 0.0

    code_like = sum(
        1 for t in tokens if (t in CODE_TOKENS or (t.isidentifier() and "_" in t))
    )

    return code_like / len(tokens)


def compute_citation_count(text: str) -> int:
    count = len(RE_CITATION.findall(text))
    if RE_REFERENCES_HEADER.search(text):
        count += 3  # structural boost
    return count


def compute_repetition_score(tokens: List[str], n: int = 5) -> float:
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))

    # ToDo: This can go high, cap it later if needed. min(score, 0.95)
    return 1.0 - (unique / total)


# ------------------------
# Main entry point
# ------------------------


def extract_document_features(text: str) -> Dict:
    tokens = simple_tokenize(text)

    features = {}
    features.update(extract_basic_shape(text))
    features.update(extract_structural(text, tokens))
    features.update(extract_lexical(tokens))
    features.update(extract_semantic_flags(text))

    sentence_stats = extract_sentence_stats(text)
    features.update(sentence_stats)
    features["flesch_kincaid_grade"] = compute_flesch_kincaid(
        text, tokens, sentence_stats.get("sentence_count", 1)
    )

    features["gzip_compression_ratio"] = extract_compression_ratio(text)
    features["citation_count"] = compute_citation_count(text)

    # Optional placeholders (filled later)
    features["teacher_ppl"] = None
    features["entropy_variance"] = None

    # Derived flags
    features["is_code_heavy"] = features.get("code_block_count", 0) >= 2
    features["is_math_heavy"] = features.get("math_symbol_ratio", 0) >= 0.01
    features["is_long_form"] = features.get("doc_token_count", 0) >= 1500
    features["is_potential_cot"] = features.get(
        "has_reasoning_markers"
    ) or features.get("has_step_markers")
    features["is_agentic_like"] = features.get("has_agent_markers")

    return features
