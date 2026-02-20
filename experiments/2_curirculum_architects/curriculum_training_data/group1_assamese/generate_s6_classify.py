#!/usr/bin/env python3
"""
Generate Statement 6: Semantic Classification
Target: 10,000 pairs (no exact duplicates)
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
    seen = set()  # Track (query, answer) to avoid exact duplicates
    target_count = 10000

    categories = list(CLASSIFICATION_CATEGORIES.keys())
    max_attempts = target_count * 15
    attempts = 0

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        target_cat = random.choice(categories)
        words = CLASSIFICATION_CATEGORIES[target_cat]
        if not words:
            continue
        word = random.choice(words)

        distractor_cat = random.choice([c for c in categories if c != target_cat])
        cat1_display = target_cat.split(" (")[0]
        cat2_display = distractor_cat.split(" (")[0]

        opts = [(cat1_display, "correct"), (cat2_display, "wrong")]
        random.shuffle(opts)

        template = random.choice(TEMPLATES)
        query = template.format(word=word, cat1=opts[0][0], cat2=opts[1][0])
        answer = cat1_display
        key = (query, answer)
        if key not in seen:
            seen.add(key)
            samples.append((query, answer))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S6 Classification: Generated {len(samples)} samples (no duplicates)")

if __name__ == "__main__":
    main()
