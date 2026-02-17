#!/usr/bin/env python3
"""
Generate Statement 6: Semantic Classification
Target: 20,000 pairs
Focus: Living/Non-living, Action vs Object, Categories.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import CLASSIFICATION_CATEGORIES
from prompt_utils import format_qa_pair_hindi

TEMPLATES = [
    '"{word}" কি শ্ৰেণীৰ অন্তৰ্গত: {cat1} নে {cat2}?',
    '"{word}" এটা {cat1} নে {cat2}?',
    '"{word}"ক কি বুলি শ্ৰেণীবিভাজন কৰিব পাৰি: {cat1} নে {cat2}?',
    '"{word}"ৰ শ্ৰেণী কি? {cat1} নে {cat2}?',
]


def main():
    samples = []
    target_count = 20000

    categories = list(CLASSIFICATION_CATEGORIES.keys())

    while len(samples) < target_count:
        # Pick a target category and word
        target_cat = random.choice(categories)
        words = CLASSIFICATION_CATEGORIES[target_cat]
        if not words:
            continue
        word = random.choice(words)

        # Pick a distractor category
        distractor_cat = random.choice([c for c in categories if c != target_cat])

        # Format categories for display (remove English translation in parens if desired,
        # but the keys like "জীৱ (Living)" are helpful context.
        # Maybe strip English for the prompt to keep it pure Assamese?
        # Let's keep the key as is for now, or strip.
        # "জীৱ (Living)" -> "জীৱ" might be more natural.

        cat1_display = target_cat.split(" (")[0]
        cat2_display = distractor_cat.split(" (")[0]

        # Randomize order in prompt
        opts = [(cat1_display, "correct"), (cat2_display, "wrong")]
        random.shuffle(opts)

        template = random.choice(TEMPLATES)
        query = template.format(word=word, cat1=opts[0][0], cat2=opts[1][0])
        answer = cat1_display

        samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S6 Classification: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
