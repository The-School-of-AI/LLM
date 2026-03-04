#!/usr/bin/env python3
"""
Generate Statement 4: Character/Akshara Count
Target: 15,000 pairs
Focus: Counting grapheme clusters (visual units).
"""

import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from group1_assamese.assamese_vocabulary_expanded import (
        ALL_WORDS_EXPANDED_UNIQUE as ALL_WORDS_UNIQUE,
    )
except ImportError:
    from group1_assamese.assamese_vocabulary import ALL_WORDS_UNIQUE

from prompt_utils import format_qa_pair_hindi

# Expand word list
WORDS = ALL_WORDS_UNIQUE * 50

TEMPLATES = [
    '"{word}"ত কেইটা আখৰ আছে?',
    '"{word}" শব্দটোত আখৰৰ সংখ্যা কিমান?',
    '"{word}"ত বৰ্ণৰ সংখ্যা গণনা কৰক।',
    '"{word}" শব্দটোত কিমানটা আখৰ আছে?',
    '"{word}"ত কিমানটা আখৰ আছে?',
    '"{word}" শব্দত মুঠ কেইটা বৰ্ণ আছে?',
    '"{word}"ৰ মুঠ আখৰৰ সংখ্যা কিমান?',
    '"{word}"ৰ আখৰবোৰ গণনা কৰক।',
    '"{word}"ত কিমানটা আখৰ পোৱা যায়?',
    '"{word}"ৰ আখৰবোৰ গণনা কৰিলে কিমান হ\'ব?',
    '"{word}" শব্দটোত মুঠতে কেইটা আখৰ থাকে?',
]

ASSAMESE_NUMERAL = {
    "0": "০",
    "1": "১",
    "2": "২",
    "3": "৩",
    "4": "৪",
    "5": "৫",
    "6": "৬",
    "7": "৭",
    "8": "৮",
    "9": "৯",
}


def to_assamese_numeral(n: int) -> str:
    """Convert integer to Bengali-Assamese numeral string."""
    return "".join(ASSAMESE_NUMERAL.get(c, c) for c in str(n))


def get_assamese_grapheme_clusters(word: str) -> list[str]:
    """Get grapheme clusters using \\X pattern."""
    return regex.findall(r"\X", word)


def main():
    samples = []
    target_count = 15000

    unique_combinations = set()
    max_attempts = target_count * 10
    attempts = 0

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        word = random.choice(WORDS)
        clusters = get_assamese_grapheme_clusters(word)
        count = len(clusters)

        template = random.choice(TEMPLATES)
        query = template.format(word=word)
        # Answer format: Assamese numeral
        answer = to_assamese_numeral(count)

        if (query, answer) not in unique_combinations:
            unique_combinations.add((query, answer))
            samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S4 Count: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
