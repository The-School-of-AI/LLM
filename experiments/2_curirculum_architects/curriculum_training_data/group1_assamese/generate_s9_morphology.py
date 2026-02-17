#!/usr/bin/env python3
"""
Generate Statement 9: Morphology (Roots + Suffixes)
Target: 30,000 pairs
Focus: Root word extraction and suffix identification.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    EASY_OBJECTS,
    EASY_PEOPLE,
    SUFFIXES,
    VERBS,
)
from prompt_utils import format_qa_pair_hindi

# Define specific suffix categories based on SUFFIXES dict
HUMAN_SUFFIXES = ["সকল", "হঁত", "জন", "জনী"]
# Note: "বোৰ", "বিলাক", "মখা" are in 'plural' but mostly for objects/general.
# "টো", "টি" can be both but safe for objects.
# "খন", "ডাল", etc are classifiers for objects.
OBJECT_SUFFIXES = [
    "বোৰ",
    "বিলাক",
    "মখা",
    "টো",
    "টি",
    "খন",
    "খনি",
    "ডাল",
    "চটা",
    "গছ",
    "পাত",
    "জোপা",
]
CASE_SUFFIXES = ["ৰ", "লৈ", "ত", "ৰে", "ৰপৰা", "লৈকে", "ক"]

# Define allowed combinations
# (Word List, List of allowed suffix lists)
COMBINATIONS = [
    (
        EASY_PEOPLE,
        [HUMAN_SUFFIXES, CASE_SUFFIXES, ["বোৰ", "বিলাক"]],
    ),  # People can take plural/case
    (EASY_OBJECTS, [OBJECT_SUFFIXES, CASE_SUFFIXES]),  # Objects avoid human suffixes
    (
        VERBS,
        [HUMAN_SUFFIXES, OBJECT_SUFFIXES, CASE_SUFFIXES],
    ),  # Nominalized verbs can be flexible
]

TEMPLATES_ROOT = [
    '"{inflected}" শব্দটোৰ মূল শব্দ কি?',
    '"{inflected}"ৰ মূল কি?',
    '"{inflected}" - ইয়াৰ মূল শব্দটো বাছনি কৰক।',
]

TEMPLATES_SUFFIX = [
    '"{inflected}" শব্দটোত কি বিভক্তি/প্ৰত্যয় যোগ হৈছে?',
    '"{inflected}"ৰ শেষত কি যোগ হৈছে?',
    '"{root}"ৰ লগত "{suffix}" যোগ কৰিলে কি হ\'ব?',  # Synthesis
]


def main():
    samples = []
    target_count = 30000

    # Flatten suffixes
    all_suffixes = []
    for type_list in SUFFIXES.values():
        all_suffixes.extend(type_list)

    while len(samples) < target_count:
        # Select a category of words and compatible suffixes
        word_list, allowed_suffix_groups = random.choice(COMBINATIONS)
        root = random.choice(word_list)

        # Pick a suffix group then a suffix
        suffix_group = random.choice(allowed_suffix_groups)
        suffix = random.choice(suffix_group)

        # Simple agglutination (naive concatenation)
        # Note: Assamese sandhi rules are complex.
        # For this dataset, we use simple concatenation which works for many cases
        # or accepted colloquial forms.
        # Ideally we'd have a conjugator, but naive approx is standard for this level.
        inflected = root + suffix

        task_type = random.random()

        if task_type < 0.4:
            # Identify Root
            template = random.choice(TEMPLATES_ROOT)
            query = template.format(inflected=inflected)
            answer = root
            samples.append((query, answer))

        elif task_type < 0.7:
            # Identify Suffix
            template = random.choice(TEMPLATES_SUFFIX[:2])
            query = template.format(inflected=inflected)
            answer = suffix
            samples.append((query, answer))

        else:
            # Synthesis (Root + Suffix -> Inflected)
            template = TEMPLATES_SUFFIX[2]
            query = template.format(root=root, suffix=suffix)
            answer = inflected
            samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S9 Morphology: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
