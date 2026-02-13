#!/usr/bin/env python3
"""
Generate Statement 10: Word Comparison (ਸ਼ਬਦ ਤੁਲਨਾ) questions for Punjabi
Target: 170,000 unique pairs to reach the 200,000 total unique goal.
"""

import os
import random
import sys
from itertools import combinations

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi


def get_punjabi_grapheme_clusters(word: str) -> list[str]:
    return regex.findall(r"\X", word)


TEMPLATES_LONGER = [
    '"{word1}" ਅਤੇ "{word2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ ਲੰਬਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ ਵੱਡਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"{word1}" ਜਾਂ "{word2}", ਇਹਨਾਂ ਵਿੱਚੋਂ ਕਿਸ ਵਿੱਚ ਜ਼ਿਆਦਾ ਅੱਖਰ ਹਨ?',
    'ਵੱਧ ਅੱਖਰਾਂ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    'ਲੰਬਾਈ ਵਿੱਚ ਕਿਹੜਾ ਸ਼ਬਦ ਵੱਡਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
]

TEMPLATES_SHORTER = [
    '"{word1}" ਅਤੇ "{word2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ ਛੋਟਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ ਛੋਟਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"{word1}" ਜਾਂ "{word2}", ਇਹਨਾਂ ਵਿੱਚੋਂ ਕਿਸ ਵਿੱਚ ਘੱਟ ਅੱਖਰ ਹਨ?',
    'ਘੱਟ ਅੱਖਰਾਂ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    'ਲੰਬਾਈ ਵਿੱਚ ਕਿਹੜਾ ਸ਼ਬਦ ਛੋਟਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    # Pre-calculate counts
    word_counts = {w: len(get_punjabi_grapheme_clusters(w)) for w in words}

    # Generate all pairs
    all_pairs = list(combinations(words, 2))
    random.shuffle(all_pairs)

    for w1, w2 in all_pairs:
        c1, c2 = word_counts[w1], word_counts[w2]
        if c1 == c2:
            continue

        longer = w1 if c1 > c2 else w2
        shorter = w2 if c1 > c2 else w1

        # Longer questions
        for template in TEMPLATES_LONGER:
            # Randomize order of options
            o1, o2 = (w1, w2) if random.random() < 0.5 else (w2, w1)
            query = template.format(word1=o1, word2=o2)
            answer = longer
            samples.add((query, answer))
            if len(samples) >= target_count:
                return list(samples)

        # Shorter questions
        for template in TEMPLATES_SHORTER:
            o1, o2 = (w1, w2) if random.random() < 0.5 else (w2, w1)
            query = template.format(word1=o1, word2=o2)
            answer = shorter
            samples.add((query, answer))
            if len(samples) >= target_count:
                return list(samples)

    return list(samples)


if __name__ == "__main__":
    target_count = 170000
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S10 Punjabi Word Comparison: Generated {len(samples)} unique samples")
