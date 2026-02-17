#!/usr/bin/env python3
"""
Generate Statement 8: Word Boundaries
Target: 15,000 pairs
Focus: First letter, Last letter, Prefix identification.
"""

import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi

WORDS = ALL_WORDS_UNIQUE * 50

TEMPLATES_FIRST = [
    '"{word}"ৰ প্ৰথম আখৰটো কি?',
    '"{word}" শব্দটো কি আখৰেৰে আৰম্ভ হৈছে?',
    '"{word}"ৰ আৰম্ভণিৰ বৰ্ণটো কি?',
]

TEMPLATES_LAST = [
    '"{word}"ৰ শেষৰ আখৰটো কি?',
    '"{word}" শব্দটো কি আখৰেৰে শেষ হৈছে?',
    '"{word}"ৰ অন্তিম বৰ্ণটো কি?',
]


def get_assamese_grapheme_clusters(word: str) -> list[str]:
    return regex.findall(r"\X", word)


def main():
    samples = []
    target_count = 15000

    while len(samples) < target_count:
        word = random.choice(WORDS)
        clusters = get_assamese_grapheme_clusters(word)
        if not clusters:
            continue

        if random.random() < 0.5:
            # First letter
            template = random.choice(TEMPLATES_FIRST)
            query = template.format(word=word)
            answer = clusters[0]
        else:
            # Last letter
            template = random.choice(TEMPLATES_LAST)
            query = template.format(word=word)
            answer = clusters[-1]

        samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S8 Boundaries: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
