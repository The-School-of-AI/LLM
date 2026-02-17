#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (Verb endings)
Target: 15,000 pairs
Focus: Matching words with same ending sounds/suffixes.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    ALL_WORDS_UNIQUE,
    RHYMING_GROUPS,
    RHYMING_PAIRS,
)
from prompt_utils import format_qa_pair_hindi

TEMPLATES_PAIR = [
    '"{word}"ৰ লগত ছন্দ মিল থকা শব্দ এটা কোৱা?',
    '"{word}"ৰ এটা ছন্দোবদ্ধ শব্দ কি?',
    '"{word}"ৰ লগত মিল থকা শব্দ কি?',
]

TEMPLATES_CHOICE = [
    '"{word}"ৰ লগত কোনটো শব্দৰ ছন্দ মিলে: "{option1}" নে "{option2}"?',
    '"{word}" আৰু "{option1}"ৰ মাজত ছন্দৰ মিল আছেনে?',  # This is yes/no, maybe stick to choice
    '"{word}"ৰ লগত ছন্দ মিল থকা শব্দটো বাছনি কৰক: "{option1}", "{option2}"।',
]


def main():
    samples = []
    target_count = 15000

    # 1. Generate from RHYMING_PAIRS (Direct mapping)
    pairs = list(RHYMING_PAIRS.items())

    # 2. Generate from RHYMING_GROUPS (Combinations within group)
    group_pairs = []
    for group in RHYMING_GROUPS.values():
        if len(group) >= 2:
            # Generate all pairs in group
            for i in range(len(group)):
                for j in range(len(group)):
                    if i != j:
                        group_pairs.append((group[i], group[j]))

    all_pairs = pairs + group_pairs

    while len(samples) < target_count:
        # Pick a valid rhyming pair
        word, rhyme = random.choice(all_pairs)

        # Strategy 1: Direct question (Open ended -> formatted as single answer)
        if random.random() < 0.4:
            template = random.choice(TEMPLATES_PAIR)
            query = template.format(word=word)
            answer = rhyme
            samples.append((query, answer))

        # Strategy 2: Multiple Choice
        else:
            # Pick a distractor (non-rhyming)
            # Ideally ending shouldn't match.
            # Simple heuristic: word not in the same rhyming group/pair
            distractor = random.choice(ALL_WORDS_UNIQUE)
            if distractor == rhyme or distractor == word:
                continue

            template = random.choice(TEMPLATES_CHOICE)

            # Randomize options
            opts = [(rhyme, "correct"), (distractor, "wrong")]
            random.shuffle(opts)
            opt1, type1 = opts[0]
            opt2, type2 = opts[1]

            query = template.format(word=word, option1=opt1, option2=opt2)
            answer = rhyme
            samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S5 Rhyming: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
