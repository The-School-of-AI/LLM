#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (ਅੱਖਰ ਦਾ ਸਥਾਨ) questions for Punjabi
Target: 17,200 pairs
"""

import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi


def get_punjabi_grapheme_clusters(word: str) -> list[str]:
    # Return clusters, skipping internal spaces
    return [c for c in regex.findall(r"\X", word) if not c.isspace()]


POSITIONS_ANS = [
    "ਪਹਿਲੇ",
    "ਦੂਜੇ",
    "ਤੀਜੇ",
    "ਚੌਥੇ",
    "ਪੰਚਵੇਂ",
    "ਛੇਵੇਂ",
    "ਸੱਤਵੇਂ",
    "ਅੱਠਵੇਂ",
    "ਨੌਵੇਂ",
    "ਦਸਵੇਂ",
]

TEMPLATES = [
    '"{word}" ਵਿੱਚ "{char}" ਕਿਹੜੇ ਸਥਾਨ ਤੇ ਹੈ?',
    '"{word}" ਵਿੱਚ "{char}" ਅੱਖਰ ਦਾ ਸਥਾਨ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ "{char}" ਕਿੰਨਵੇਂ ਸਥਾਨ ਤੇ ਆਉਂਦਾ ਹੈ?',
    '"{word}" ਵਿੱਚ "{char}" ਕਿੱਥੇ ਹੈ, {options}?',
    '"{char}" ਅੱਖਰ "{word}" ਵਿੱਚ ਕਿਹੜੇ ਸਥਾਨ ਤੇ ਹੈ?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    for word in words:
        clusters = get_punjabi_grapheme_clusters(word)
        for i, char in enumerate(clusters):
            if i >= len(POSITIONS_ANS):
                break
            pos_ans = POSITIONS_ANS[i]

            # For the option template
            others = [p for p in POSITIONS_ANS[: len(clusters)] if p != pos_ans]
            if others:
                opt1 = pos_ans
                opt2 = random.choice(others)
                opts = (
                    f"{opt1} ਜਾਂ {opt2}"
                    if random.random() < 0.5
                    else f"{opt2} ਜਾਂ {opt1}"
                )
            else:
                opts = pos_ans

            for template in TEMPLATES:
                query = template.format(word=word, char=char, options=opts)
                answer = pos_ans
                samples.add((query, answer))
                if len(samples) >= target_count:
                    return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 17200
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S7 Punjabi Position of Letter: Generated {len(samples)} unique samples")
