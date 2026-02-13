#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (ਅੱਖਰ ਗਿਣਤੀ) questions for Punjabi
Target: 25,800 pairs (12.9% of 200,000)
"""
import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_hindi  # noqa: E402

def get_punjabi_grapheme_clusters(word: str) -> list[str]:
    """Get grapheme clusters for Punjabi word (for counting/length/position)."""
    return regex.findall(r"\X", word)

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Question templates
TEMPLATES = [
    '"{word}" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?',
    '"{word}" ਸ਼ਬਦ ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?',
    '"{word}" ਵਿੱਚ ਅੱਖਰਾਂ ਦੀ ਸੰਖਿਆ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ ਕੁੱਲ ਕਿੰਨੇ ਅੱਖਰ ਹਨ?',
    '"{word}" ਸ਼ਬਦ ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਹੁੰਦੇ ਹਨ?',
    '"{word}" ਵਿੱਚ ਅੱਖਰਾਂ ਦੀ ਗਿਣਤੀ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ ਕਿੰਨੇ ਅੱਖਰ ਮੌਜੂਦ ਹਨ?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_punjabi_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = str(cluster_count)
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement
samples = list(unique_combinations.values())
if len(samples) > target_count:
    samples = samples[:target_count]
else:
    unique_word_list = list(set(all_words))
    while len(samples) < target_count:
        word = random.choice(unique_word_list)
        clusters = get_punjabi_grapheme_clusters(word)
        cluster_count = len(clusters)
        if cluster_count == 0:
            continue
        template = random.choice(TEMPLATES)
        query = template.format(word=word)
        answer = str(cluster_count)
        samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S4 Punjabi Letter Count: Generated {len(samples)} samples")
