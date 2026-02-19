#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (ਆਖਰੀ ਅੱਖਰ) questions for Punjabi
Target: 17,200 pairs
"""

import os
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi


def get_punjabi_grapheme_clusters(word: str) -> list[str]:
    # Return clusters, skipping internal spaces
    return [c for c in regex.findall(r"\X", word) if not c.isspace()]


TEMPLATES = [
    '"{word}" ਦਾ ਆਖਰੀ ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ ਅੰਤਿਮ ਅੱਖਰ ਕਿਹੜਾ ਹੈ?',
    '"{word}" ਸ਼ਬਦ ਕਿਸ ਅੱਖਰ ਨਾਲ ਖਤਮ ਹੁੰਦਾ ਹੈ?',
    '"{word}" ਦਾ ਅੰਤਿਮ ਅੱਖਰ ਦੱਸੋ?',
    '"{word}" ਵਿੱਚ ਸਭ ਤੋਂ ਪਿੱਛੇ ਕਿਹੜਾ ਅੱਖਰ ਹੈ?',
    '"{word}" ਕਿਸ ਅੱਖਰ ਤੇ ਸਮਾਪਤ ਹੁੰਦਾ ਹੈ?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    for word in words:
        clusters = get_punjabi_grapheme_clusters(word)
        if not clusters:
            continue
        last_char = clusters[-1]
        for template in TEMPLATES:
            query = template.format(word=word)
            answer = last_char
            samples.add((query, answer))
            if len(samples) >= target_count:
                return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 17200
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S9 Punjabi Last Letter: Generated {len(samples)} unique samples")
