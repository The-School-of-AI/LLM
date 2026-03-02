#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (ਸੰਖਿਆ ਵਰਤਨੀ) questions for Punjabi
Target: 10,000 pairs
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import NUMBERS_1_100
from prompt_utils import format_qa_pair_hindi

TEMPLATES = [
    '"{num}" ਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਕਿਵੇਂ ਲਿਖਦੇ ਹਨ?',
    'ਸੰਖਿਆ "{num}" ਦਾ ਪੰਜਾਬੀ ਵਿੱਚ ਨਾਮ ਕੀ ਹੈ?',
    '"{num}" ਦੀ ਪੰਜਾਬੀ ਵਰਤਨੀ ਕੀ ਹੈ?',
    '"{num}" ਨੂੰ ਸ਼ਬਦਾਂ ਵਿੱਚ ਕਿਵੇਂ ਲਿਖਿਆ ਜਾਂਦਾ ਹੈ?',
    'ਪੰਜਾਬੀ ਵਿੱਚ "{num}" ਨੂੰ ਕੀ ਕਹਿੰਦੇ ਹਨ?',
    '"{num}" ਦਾ ਸ਼ਬਦ ਰੂਪ ਦੱਸੋ?',
]


def generate_samples(target_count):
    samples = set()
    num_list = list(NUMBERS_1_100.items())

    for num, name in num_list:
        for template in TEMPLATES:
            query = template.format(num=num)
            answer = name
            samples.add((query, answer))
            if len(samples) >= target_count:
                return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 10000
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S8 Punjabi Number Spelling: Generated {len(samples)} unique samples")
