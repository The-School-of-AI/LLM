# isort: skip_file
"""
Utilities for prompt-only curriculum dataset generation.

Target dataset format:
- A flat JSON array (list) of strings
- Each string is a natural-language question or instruction
- No answers, no QA mappings, no dict-like structures
"""

from __future__ import annotations

import json
import re
import random
from typing import Iterable, List
import regex

# Pre-compile regexes for performance
_RE_COMMA_SEPARATED_LETTERS = re.compile(r"\b[a-z](?:,\s*[a-z])+\b", re.IGNORECASE)
_RE_JSONISH_KEY_VALUE = re.compile(r'"\s*[^"]+\s*"\s*:\s*"', re.IGNORECASE)
_RE_ARROW_DELIM = re.compile(r"\s->\s")
_RE_LEADING_QA_LABEL = re.compile(r"^[QA]\d*[:.)\s-]*", re.IGNORECASE)

# Corrected regexes for ensure_query_punctuation
_RE_QUERY_IF = re.compile(r"\.\s+If\s")
_RE_QUERY_HOW = re.compile(r"\.\s+How\s+(many|much|do|does|is|are|can)", re.IGNORECASE)
_RE_QUERY_WHAT = re.compile(r"\.\s+What(\'s|'s| is| do| does| can|\s)", re.IGNORECASE)
_RE_QUERY_WHICH = re.compile(r"\.\s+Which\s", re.IGNORECASE)
_RE_QUERY_TELL = re.compile(r"\.\s+Tell\s+me\s", re.IGNORECASE)


def get_marathi_grapheme_clusters(word: str) -> List[str]:
    """
    Split a Marathi word into grapheme clusters (syllables).
    Uses regex \\X which is Unicode UAX#29 compliant.
    """
    return regex.findall(r"\X", word)


def normalize_prompt(p: str) -> str:
    """
    Normalize whitespace/punctuation lightly without changing meaning.
    """
    p = p.strip()
    # Collapse internal whitespace runs
    p = re.sub(r"\s+", " ", p)
    return p


def dedupe_preserve_order(prompts: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def is_valid_prompt(p: str) -> bool:
    """
    Reject strings that look like answers/QA pairs or artificial formatting.
    This is intentionally conservative; generators should aim to emit clean prompts.
    """
    if not isinstance(p, str):
        return False
    if not p:
        return False
    if "\n" in p or "\r" in p:
        return False

    # Obvious QA delimiters / dump formats
    if _RE_ARROW_DELIM.search(p):
        return False
    if _RE_JSONISH_KEY_VALUE.search(p):
        return False
    if _RE_LEADING_QA_LABEL.search(p):
        return False
    if p.lstrip().startswith("{") or p.lstrip().startswith("["):
        return False

    # Forbid comma-separated spellings like "c, a, t"
    if _RE_COMMA_SEPARATED_LETTERS.search(p):
        return False

    # Colons are common in worksheet prompts (e.g., "What comes next: ...", "True or false: ...").
    # We only reject obvious labeled QA formats above; generator code should never append answers.

    return True


def filter_and_normalize_prompts(prompts: Iterable[str]) -> List[str]:
    """
    Normalize, validate, and dedupe prompts.
    """
    normalized: list[str] = []
    for p in prompts:
        p2 = normalize_prompt(p)
        if is_valid_prompt(p2):
            normalized.append(p2)
    return dedupe_preserve_order(normalized)


def save_prompts_json(path: str, prompts: List[str]) -> None:
    """
    Save prompts as a JSON array.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def remove_quotes(text: str) -> str:
    """
    Remove all single and double quotes from text.
    """
    return text.replace("'", "").replace('"', "")


def ensure_answer_period(answer: str) -> str:
    """
    Ensure answer ends with a period.
    """
    answer = answer.strip()
    if not answer.endswith("."):
        return answer + "."
    return answer


# Kannada digits ೦-೯ (U+0CE6 to U+0CEF)
_KANNADA_DIGITS = "೦೧೨೩೪೫೬೭೮೯"


def int_to_kannada(n: int) -> str:
    """Convert integer to Kannada numeral string (೦, ೧, ೨, ... ೯, ೧೦, ...)."""
    if n < 0:
        return "-" + int_to_kannada(-n)
    s = str(n)
    return "".join(_KANNADA_DIGITS[int(d)] for d in s)


# Kannada number words (for "ಕೇವಲ ಮೂರು ಅಕ್ಷರಗಳಿವೆ" style phrases)
_KANNADA_NUM_WORDS = {
    1: "ಒಂದು",
    2: "ಎರಡು",
    3: "ಮೂರು",
    4: "ನಾಲ್ಕು",
    5: "ಐದು",
    6: "ಆರು",
    7: "ಏಳು",
    8: "ಎಂಟು",
    9: "ಒಂಬತ್ತು",
    10: "ಹತ್ತು",
}


def int_to_kannada_word(n: int) -> str:
    """Convert small integer to Kannada word form (ಮೂರು, ನಾಲ್ಕು, etc.) for prose."""
    if 1 <= n <= 10:
        return _KANNADA_NUM_WORDS[n]
    return int_to_kannada(n)


def format_qa_pair_kannada(query: str, answer: str) -> str:
    """
    Format a query-answer pair for Kannada TXT output.
    - Preserves quotes around target words/sequences
    - Ensures query ends with ?
    - Ensures answer ends with period (.)
    - Returns formatted string: "query? answer."

    CRITICAL: Queries MUST end with "?", answers MUST end with "."
    """
    query_clean = query.strip()
    query_clean = ensure_query_punctuation(query_clean)
    answer_clean = ensure_answer_period(answer)
    return f"{query_clean} {answer_clean}"


def ensure_query_punctuation(query: str) -> str:
    """
    Ensure query ends with a question mark.
    - Fixes internal periods before question words (e.g., "Compare X and Y. Which is less?" -> "Compare X and Y, which is less?")
    - If it doesn't end with '?', replace any trailing punctuation with '?' or add '?'.
    """
    query = query.strip()
    if not query:
        return query

    # Fix patterns like "Compare X and Y. Which is less?" -> "Compare X and Y, which is less?"
    # Fix patterns like "You have X. If..." -> "You have X, if..."
    # Fix patterns like "Add X. What's..." -> "Add X, what's..."

    # Replace ". If" with ", if"
    query = _RE_QUERY_IF.sub(", if ", query)
    # Replace ". How" with ", how" (when followed by question word)
    query = _RE_QUERY_HOW.sub(r", how \1", query)
    # Replace ". What" with ", what" (handles both "What " and "What's", "What's", etc.)
    query = _RE_QUERY_WHAT.sub(r", what\1", query)
    # Replace ". Which" with ", which"
    query = _RE_QUERY_WHICH.sub(", which ", query)
    # Replace ". Tell" with ", tell" (when it's "tell me")
    query = _RE_QUERY_TELL.sub(", tell me ", query)

    # If already ends with '?', return as-is
    if query.endswith("?"):
        return query

    # Remove any trailing punctuation (. ! , ; :) and add '?'
    query = query.rstrip(".!?,;:")
    return query + "?"


def count_tokens(text: str) -> int:
    """
    Count tokens using LLM-like tokenization.

    Tokenization rules:
    - For Devanagari/Hindi: Each Unicode character counts as 1 token (matches spelling format)
    - For other scripts: Word units (sequences of letters/digits) count as 1 token
    Optimized token counter for Marathi/English text.
    - Each Devanagari/Kannada character = 1 token
    - Each alphanumeric word (including internal apostrophes) = 1 token
    - Each symbol/punctuation = 1 token
    - Whitespace skipped
    """
    # Regex to capture all token types:
    # 1. Devanagari/Kannada characters: [\u0900-\u097f\u0c80-\u0cff]
    # 2. English words with optional apostrophes: [a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*
    # 3. Any other non-whitespace character (symbols): [^\s\u0900-\u097f\u0c80-\u0cff]
    # Note: we use regex.findall for speed
    tokens = regex.findall(
        r"[\u0900-\u097f\u0c80-\u0cff]|[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)*|[^\s\u0900-\u097f\u0c80-\u0cff]",
        text,
    )
    return len(tokens)


def format_qa_pair(query: str, answer: str) -> str:
    """
    Format a query-answer pair for TXT output.
    - Preserves quotes around target words/sequences (as required by professor)
    - Ensures query ends with ? or .
    - Ensures answer ends with period
    - Returns formatted string: "query? answer."
    """
    # Don't remove quotes - professor wants quotes around target words/sequences
    # Only strip whitespace
    query_clean = query.strip()
    query_clean = ensure_query_punctuation(query_clean)
    answer_clean = ensure_answer_period(answer)
    return f"{query_clean} {answer_clean}"


def ensure_answer_purna_viraam(answer: str) -> str:
    """
    Ensure answer ends with purna-viraam (।) for Hindi.
    """
    answer = answer.strip()
    if not answer.endswith("।"):
        return answer + "।"
    return answer


def format_qa_pair_hindi(query: str, answer: str) -> str:
    """
    Format a query-answer pair for Hindi TXT output.
    - Preserves quotes around target words/sequences
    - Ensures query ends with ? (NEVER use । in queries)
    - Ensures answer ends with purna-viraam (।)
    - Returns formatted string: "query? answer।"

    CRITICAL: Queries MUST end with "?", answers MUST end with "।"
    """
    # Don't remove quotes - preserve quotes around target words/sequences
    # Only strip whitespace
    query_clean = query.strip()
    # Ensure query ends with ? (critical for LLM training)
    query_clean = ensure_query_punctuation(query_clean)
    # Ensure answer ends with । (purna-viraam)
    answer_clean = ensure_answer_purna_viraam(answer)
    return f"{query_clean} {answer_clean}"


def ensure_answer_full_stop(answer: str) -> str:
    """
    Ensure answer ends with a full stop (.) for Marathi.
    """
    answer = answer.strip()
    if not answer.endswith("."):
        return answer + "."
    return answer


def format_qa_pair_marathi(query: str, answer: str) -> str:
    """
    Format a query-answer pair for Marathi TXT output.
    - Preserves quotes around target words/sequences
    - Ensures query ends with ?
    - Ensures answer ends with full stop (.)
    - Returns formatted string: "query? answer."

    CRITICAL: Queries MUST end with "?", answers MUST end with "."
    """
    query_clean = query.strip()
    # Ensure query ends with ? (critical for LLM training)
    query_clean = ensure_query_punctuation(query_clean)
    # Ensure answer ends with . (full stop)
    answer_clean = ensure_answer_full_stop(answer)
    return f"{query_clean} {answer_clean}"


def combine_qa_pairs_to_reach_min_tokens(
    qa_pairs: list[tuple[str, str]], min_tokens: int = 512
) -> list[str]:
    """
    Combine QA pairs into samples where all questions have answers.
    Format: "Q1? A1. Q2? A2. Q3? A3. ..." (all questions with answers)
    until reaching min_tokens per sample.

    Args:
        qa_pairs: List of (query, answer) tuples
        min_tokens: Minimum tokens per sample

    Returns:
        List of formatted sample strings, each with >= min_tokens
    """
    if not qa_pairs:
        return []

    samples = []
    i = 0

    while i < len(qa_pairs):
        current_sample_parts = []
        current_sample_qa_pairs = set()  # Track QA pairs to avoid duplicates
        current_tokens = 0

        # Add QA pairs (all with answers) until we reach min_tokens
        while current_tokens < min_tokens and i < len(qa_pairs):
            query, answer = qa_pairs[i]
            qa_key = (query, answer)  # Use tuple as key for deduplication

            # Skip if this QA pair already in current sample
            if qa_key not in current_sample_qa_pairs:
                qa_formatted = format_qa_pair(query, answer)
                current_sample_parts.append(qa_formatted)
                current_sample_qa_pairs.add(qa_key)
                current_tokens += count_tokens(qa_formatted)

            i += 1

        # Join all parts with spaces
        sample = " ".join(current_sample_parts)
        samples.append(sample)

    return samples


def combine_qa_pairs_to_reach_min_tokens_marathi(
    qa_pairs: list[tuple[str, str]], min_tokens: int = 512
) -> list[str]:
    """
    Optimized version: Pre-calculates formatting and tokens to avoid O(N*M) overhead.
    """
    if not qa_pairs:
        return []

    # PRE-CALCULATE formatting and tokens (once per pair)
    # This is the single biggest performance win for large datasets
    formatted_data = []
    print(f"Pre-processing {len(qa_pairs)} pairs for combination...")
    for q, a in qa_pairs:
        fmt = format_qa_pair_marathi(q, a)
        cnt = count_tokens(fmt)
        formatted_data.append((fmt, cnt, (q, a)))

    samples = []
    used_indices = set()
    total_pairs = len(formatted_data)
    current_idx = 0

    while len(used_indices) < total_pairs:
        current_sample_parts = []
        current_sample_qa_keys = set()
        current_tokens = 0

        # sequential consumption
        while current_tokens < min_tokens and current_idx < total_pairs:
            if current_idx not in used_indices:
                fmt, cnt, qa_key = formatted_data[current_idx]
                if qa_key not in current_sample_qa_keys:
                    # check for huge pair
                    if current_tokens + cnt <= min_tokens * 3:
                        current_sample_parts.append(fmt)
                        current_sample_qa_keys.add(qa_key)
                        current_tokens += cnt
                        used_indices.add(current_idx)
            current_idx += 1

        # Check if we reached the goal. If not, we might be at the end of the list.
        # Use a small set of random samples to top off if needed (MUCH faster than full scan)
        if current_tokens < min_tokens and current_sample_parts:
            # Try a limited number of random pairs to fill up
            for _ in range(100):
                rj = random.randint(0, total_pairs - 1)
                fmt, cnt, qa_key = formatted_data[rj]
                if qa_key not in current_sample_qa_keys:
                    if current_tokens + cnt <= min_tokens * 3:
                        current_sample_parts.append(fmt)
                        current_sample_qa_keys.add(qa_key)
                        current_tokens += cnt
                        used_indices.add(rj)
                        if current_tokens >= min_tokens:
                            break

        if current_sample_parts:
            samples.append(" ".join(current_sample_parts))
        else:
            # safeguard
            break

    return samples


def combine_qa_pairs_to_reach_min_tokens_kannada(
    qa_pairs: list[tuple[str, str]], min_tokens: int = 512
) -> list[str]:
    """
    Super-optimized version of combining QA pairs for Kannada.
    """
    if not qa_pairs:
        return []

    # Pre-format all pairs and calculate their tokens once
    formatted_pairs = []
    for q, a in qa_pairs:
        fmt = format_qa_pair_kannada(q, a)
        formatted_pairs.append((fmt, count_tokens(fmt)))

    samples = []
    current_sample_parts = []
    current_tokens = 0

    for fmt, tokens in formatted_pairs:
        current_sample_parts.append(fmt)
        current_tokens += tokens

        if current_tokens >= min_tokens:
            samples.append(" ".join(current_sample_parts) + "\n")
            current_sample_parts = []
            current_tokens = 0

    # Add any remaining pairs to the last sample if it's too small, or as a new sample
    if current_sample_parts:
        # If there are already samples, append to the last one, otherwise create a new one
        if samples:
            samples[-1] = (
                samples[-1].rstrip("\n") + " " + " ".join(current_sample_parts) + "\n"
            )
        else:
            samples.append(" ".join(current_sample_parts) + "\n")

    return samples


def get_kannada_grapheme_clusters(word: str) -> list[str]:
    """
    Get grapheme clusters for Kannada word (for counting/length/position).
    Uses regex library's \\X pattern (Unicode UAX#29 compliant).
    Each grapheme cluster = 1 अक्षर (akshara) for counting/position questions.
    """
    import regex

    return regex.findall(r"\X", word)


def get_kannada_characters(word: str) -> list[str]:
    """
    Break down a Kannada word into its constituent Unicode characters.
    Each Unicode character (consonant, vowel, matra, nukta, virama) is separate.
    This matches the spelling format where each character is shown separately.
    """
    # Simply return each Unicode character separately
    return list(word)
