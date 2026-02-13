#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (ਅੱਖਰ ਦੀ ਸਥਿਤੀ) questions for Punjabi
Target: 17,200 pairs (8.6% of 200,000)
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

# Position names in Punjabi
POSITIONS = [
    ("ਪਹਿਲਾ", "1"),
    ("ਦੂਜਾ", "2"),
    ("ਤੀਜਾ", "3"),
    ("ਚੌਥਾ", "4"),
    ("ਪੰਜਵਾਂ", "5"),
    ("ਛੇਵਾਂ", "6"),
    ("ਸੱਤਵਾਂ", "7"),
    ("ਅੱਠਵਾਂ", "8"),
    ("ਨੌਂਵਾਂ", "9"),
    ("ਦਸਵਾਂ", "10"),
]

# Question templates
TEMPLATES = [
    '"{word}" ਵਿੱਚ "{char}" ਅੱਖਰ ਕਿਸ ਸਥਾਨ ਤੇ ਹੈ?',
    '"{word}" ਵਿੱਚ "{char}" ਅੱਖਰ ਕਿੱਥੇ ਹੈ?',
    '"{word}" ਸ਼ਬਦ ਵਿੱਚ "{char}" ਅੱਖਰ ਕਿਸ ਸਥਾਨ ਤੇ ਹੈ?',
    '"{word}" ਵਿੱਚ "{char}" ਕਿਸ ਸਥਾਨ ਤੇ ਮਿਲਦਾ ਹੈ?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 17200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_punjabi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    for cluster in clusters:
        cluster_positions = [i + 1 for i, c in enumerate(clusters) if c == cluster]
        if not cluster_positions:
            continue

        # Use first occurrence
        pos_num = cluster_positions[0]
        if pos_num <= len(POSITIONS):
            pos_name, pos_str = POSITIONS[pos_num - 1]
        else:
            pos_name = f"{pos_num}ਵਾਂ"
            pos_str = str(pos_num)

        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word, char=cluster)
            answer = pos_name if random.random() < 0.5 else pos_str
            key = (word, cluster, template_idx)
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
        if len(clusters) == 0:
            continue
        cluster = random.choice(clusters)
        pos_num = clusters.index(cluster) + 1
        if pos_num <= len(POSITIONS):
            pos_name, pos_str = POSITIONS[pos_num - 1]
        else:
            pos_name = f"{pos_num}ਵਾਂ"
            pos_str = str(pos_num)

        template = random.choice(TEMPLATES)
        query = template.format(word=word, char=cluster)
        answer = pos_name if random.random() < 0.5 else pos_str
        samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S7 Punjabi Position of Letter: Generated {len(samples)} samples")
