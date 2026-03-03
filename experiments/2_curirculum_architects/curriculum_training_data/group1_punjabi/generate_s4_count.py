#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (ਅੱਖਰ ਗਿਣਤੀ) questions for Punjabi
Target: 25,800 pairs
"""

import os
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi


def get_punjabi_grapheme_clusters(word: str) -> list[str]:
    # Skip spaces in count
    return [c for c in regex.findall(r"\X", word) if not c.isspace()]


TEMPLATES = [
    '"{word}" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?',
    '"{word}" ਸ਼ਬਦ ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹੁੰਦੇ ਹਨ?',
    '"{word}" ਵਿੱਚ ਅੱਖਰਾਂ ਦੀ ਗਿਣਤੀ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ ਕੁੱਲ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?',
    '"{word}" ਵਿੱਚ ਅੱਖਰਾਂ ਦੀ ਸੰਖਿਆ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਮੌਜੂਦ ਹਨ?',
    '"{word}" ਸ਼ਬਦ ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਮੰਨੇ ਜਾਂਦੇ ਹਨ?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    for word in words:
        count = len(get_punjabi_grapheme_clusters(word))
        for template in TEMPLATES:
            query = template.format(word=word)
            answer = str(count)
            samples.add((query, answer))
            if len(samples) >= target_count:
                return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 25800
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S4 Punjabi Letter Count: Generated {len(samples)} unique samples")
