#!/usr/bin/env python3
"""
Prompt utilities for Telugu dataset generation.
Separate from shared prompt_utils.py to avoid modifying other languages.
Handles Telugu Unicode range (U+0C00–U+0C7F) for token counting.
"""


def count_tokens_telugu(text: str) -> int:
    """
    Count tokens using LLM-like tokenization with Telugu support.

    Tokenization rules:
    - For Telugu/Devanagari/Kannada: Each Unicode character counts as 1 token
    - For other scripts: Word units (sequences of letters/digits) count as 1 token
    - Symbol units: punctuation, quotes, and other symbols each count as 1 token
    - Whitespace is skipped (not counted)

    Examples:
    - "పుస్తకం" -> 5 tokens (5 Unicode chars)
    - "నీరు" -> 3 tokens (3 Unicode chars: న, ీ, రు)
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

        # Check if character is Devanagari, Telugu, or Kannada
        is_devanagari = "\u0900" <= ch <= "\u097f"
        is_telugu = "\u0c00" <= ch <= "\u0c7f"
        is_kannada = "\u0c80" <= ch <= "\u0cff"

        if is_devanagari or is_telugu or is_kannada:
            count += 1
            i += 1
            continue

        # Word unit: letters/digits (for non-Indic), allowing internal apostrophes
        if ch.isalnum():
            count += 1
            i += 1
            while i < n:
                next_ch = text[i]
                if (
                    "\u0900" <= next_ch <= "\u097f"
                    or "\u0c00" <= next_ch <= "\u0c7f"
                    or "\u0c80" <= next_ch <= "\u0cff"
                ):
                    break
                if next_ch.isalnum():
                    i += 1
                elif next_ch == "'" and i + 1 < n and text[i + 1].isalnum():
                    i += 1
                else:
                    break
            continue

        # Symbol unit: everything else (punctuation, quotes, etc.)
        count += 1
        i += 1

    return count


def ensure_answer_period(answer: str) -> str:
    """Ensure answer ends with a period."""
    answer = answer.strip()
    if not answer.endswith("."):
        return answer + "."
    return answer


def ensure_query_punctuation(query: str) -> str:
    """Ensure query ends with a question mark."""
    query = query.strip()
    if not query:
        return query
    if query.endswith("?"):
        return query
    query = query.rstrip(".!?,;:")
    return query + "?"


def format_qa_pair_telugu(query: str, answer: str) -> str:
    """
    Format a query-answer pair for Telugu TXT output.
    - Ensures query ends with ?
    - Ensures answer ends with period (.)
    - Returns formatted string: "query? answer."

    CRITICAL: Queries MUST end with "?", answers MUST end with "."
    Telugu uses period (.), NOT danda (।).
    """
    query_clean = query.strip()
    query_clean = ensure_query_punctuation(query_clean)
    answer_clean = ensure_answer_period(answer)
    return f"{query_clean} {answer_clean}"


def combine_qa_pairs_to_reach_min_tokens_telugu(
    qa_pairs: list[tuple[str, str]], min_tokens: int = 512
) -> list[str]:
    """
    Combine QA pairs into lines of ≥min_tokens each for Telugu.
    Uses Telugu-aware token counting.
    """
    if not qa_pairs:
        return []

    # Pre-format all pairs and calculate their tokens once
    formatted_pairs = []
    for q, a in qa_pairs:
        fmt = format_qa_pair_telugu(q, a)
        formatted_pairs.append((fmt, count_tokens_telugu(fmt)))

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
        if samples:
            samples[-1] = (
                samples[-1].rstrip("\n") + " " + " ".join(current_sample_parts) + "\n"
            )
        else:
            samples.append(" ".join(current_sample_parts) + "\n")

    return samples
