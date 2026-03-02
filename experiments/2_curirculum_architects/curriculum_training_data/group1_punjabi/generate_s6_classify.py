#!/usr/bin/env python3
"""
Generate Statement 6: Classification (ਸ਼੍ਰੇਣੀਬੱਧਤਾ) questions for Punjabi
Target: 20,000 pairs
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import CLASSIFICATION_CATEGORIES
from prompt_utils import format_qa_pair_hindi

TEMPLATES = [
    '"{word}" ਕਿਸ ਪ੍ਰਕਾਰ ਦੀ ਚੀਜ਼ ਹੈ, {options}?',
    '"{word}" ਨੂੰ ਤੁਸੀਂ ਕਿਸ ਸ਼੍ਰੇਣੀ ਵਿੱਚ ਰੱਖੋਗੇ, {options}?',
    '"{word}" ਕੀ ਹੈ, {options}?',
    '"{word}" ਦਾ ਸੰਬੰਧ ਕਿਸ ਨਾਲ ਹੈ, {options}?',
    '"{word}" ਇਹਨਾਂ ਵਿੱਚੋਂ ਕੀ ਹੈ: {options}?',
    '"{word}" ਕਿਸ ਵਰਗ ਵਿੱਚ ਆਉਂਦਾ ਹੈ, {options}?',
]


def generate_samples(target_count):
    samples = set()
    categories = list(CLASSIFICATION_CATEGORIES.keys())
    options_str = ", ".join(categories[:-1]) + " ਜਾਂ " + categories[-1]

    for cat, words in CLASSIFICATION_CATEGORIES.items():
        for word in words:
            for template in TEMPLATES:
                query = template.format(word=word, options=options_str)
                answer = cat
                samples.add((query, answer))
                if len(samples) >= target_count:
                    return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 20000
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S6 Punjabi Classification: Generated {len(samples)} unique samples")
