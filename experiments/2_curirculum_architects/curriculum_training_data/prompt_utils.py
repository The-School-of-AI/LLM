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
from typing import Iterable, List


_RE_COMMA_SEPARATED_LETTERS = re.compile(r"\b[a-z](?:,\s*[a-z])+\b", re.IGNORECASE)
_RE_JSONISH_KEY_VALUE = re.compile(r'"\s*[^"]+\s*"\s*:\s*"', re.IGNORECASE)
_RE_ARROW_DELIM = re.compile(r"\s->\s")
_RE_LEADING_QA_LABEL = re.compile(
    r"^\s*(?:q|a|question|answer|prompt)\s*:\s*", re.IGNORECASE
)


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
    import re

    # Replace ". If" with ", if"
    query = re.sub(r"\.\s+If\s", r", if ", query)
    # Replace ". How" with ", how" (when followed by question word)
    query = re.sub(
        r"\.\s+How\s+(many|much|do|does|is|are|can)",
        r", how \1",
        query,
        flags=re.IGNORECASE,
    )
    # Replace ". What" with ", what" (handles both "What " and "What's", "What's", etc.)
    query = re.sub(
        r"\.\s+What(\'s|\'s| is| do| does| can|\s)",
        r", what\1",
        query,
        flags=re.IGNORECASE,
    )
    # Replace ". Which" with ", which"
    query = re.sub(r"\.\s+Which\s", r", which ", query)
    # Replace ". Tell" with ", tell" (when it's "tell me")
    query = re.sub(r"\.\s+Tell\s+me\s", r", tell me ", query, flags=re.IGNORECASE)

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
    - Symbol units: punctuation, quotes, and other symbols each count as 1 token
    - Whitespace is skipped (not counted)

    Examples:
    - "c, a, t." → 6 tokens (c, comma, a, comma, t, period)
    - "What is the spelling of cat?" → 7 tokens (What, is, the, spelling, of, cat, ?)
    - "पानी" → 4 tokens (प, ा, न, ी) - each Unicode char is 1 token
    - "प, ा, न, ी" → 7 tokens (प, comma, space, ा, comma, space, न, comma, space, ी)

    Args:
        text: Input text to tokenize

    Returns:
        Number of tokens according to LLM-like tokenization rules
    """
    count = 0
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Check if character is Devanagari (U+0900 to U+097F)
        is_devanagari = "\u0900" <= ch <= "\u097f"

        if is_devanagari:
            # For Devanagari: each Unicode character = 1 token
            # This matches the spelling format where each character is shown separately
            count += 1
            i += 1
            continue

        # Word unit: letters/digits (for non-Devanagari), allowing internal apostrophes
        if ch.isalnum():
            count += 1
            i += 1
            while i < n:
                # Don't group Devanagari characters with other alphanumeric
                next_ch = text[i]
                if "\u0900" <= next_ch <= "\u097f":
                    break
                if next_ch.isalnum():
                    i += 1
                elif next_ch == "'" and i + 1 < n and text[i + 1].isalnum():
                    i += 1  # keep apostrophe inside word
                else:
                    break
            continue

        # Symbol unit: everything else (punctuation, quotes, etc.)
        count += 1
        i += 1

    return count


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


def combine_qa_pairs_to_reach_min_tokens_hindi(
    qa_pairs: list[tuple[str, str]], min_tokens: int = 512
) -> list[str]:
    """
    Super-optimized version of combining QA pairs for Hindi/Punjabi.
    """
    if not qa_pairs:
        return []

    # Pre-format all pairs and calculate their tokens once
    formatted_pairs = []
    for q, a in qa_pairs:
        fmt = format_qa_pair_hindi(q, a)
        formatted_pairs.append((fmt, count_tokens(fmt)))

    samples = []
    current_sample_parts = []
    current_tokens = 0
    
    for fmt, tokens in formatted_pairs:
        current_sample_parts.append(fmt)
        current_tokens += tokens
        
        if current_tokens >= min_tokens:
            samples.append(" ".join(current_sample_parts))
            current_sample_parts = []
            current_tokens = 0
            
    # Add any remaining pairs to the last sample if it's too small, or as a new sample
    if current_sample_parts:
        if samples:
            samples[-1] += " " + " ".join(current_sample_parts)
        else:
            samples.append(" ".join(current_sample_parts))
            
    return samples
