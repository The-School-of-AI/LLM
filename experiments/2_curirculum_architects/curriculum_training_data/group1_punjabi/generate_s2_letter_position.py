#!/usr/bin/env python3
"""
Generate Statement 2: Letter Position (ਅੱਖਰ ਸਥਿਤੀ) questions for Punjabi
Target: 25,800 pairs
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


POSITIONS = [
    "ਪਹਿਲਾ",
    "ਦੂਜਾ",
    "ਤੀਜਾ",
    "ਚੌਥਾ",
    "ਪੰਜਵਾਂ",
    "ਛੇਵਾਂ",
    "ਸੱਤਵਾਂ",
    "ਅੱਠਵਾਂ",
    "ਨੌਵਾਂ",
    "ਦਸਵਾਂ",
]

TEMPLATES = [
    '"{word}" ਦਾ {pos} ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ {pos} ਅੱਖਰ ਕਿਹੜਾ ਹੈ?',
    '"{word}" ਸ਼ਬਦ ਦਾ {pos} ਅੱਖਰ ਦੱਸੋ?',
    '"{word}" ਵਿੱਚ {pos} ਸਥਾਨ ਤੇ ਕਿਹੜਾ ਅੱਖਰ ਹੈ?',
    '"{word}" ਦਾ {pos_num} ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ {pos_num} ਸਥਾਨ ਤੇ ਕੀ ਹੈ?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    for word in words:
        clusters = get_punjabi_grapheme_clusters(word)
        for i, char in enumerate(clusters):
            if i >= len(POSITIONS):
                break
            pos_word = POSITIONS[i]
            pos_num = str(i + 1)
            for template in TEMPLATES:
                query = template.format(word=word, pos=pos_word, pos_num=pos_num)
                answer = char
                samples.add((query, answer))
                if len(samples) >= target_count:
                    return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 25800
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S2 Punjabi Letter Position: Generated {len(samples)} unique samples")
