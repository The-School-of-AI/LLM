#!/usr/bin/env python3
"""
Generate Statement 2: Letter at Position (ਅੱਖਰ ਸਥਿਤੀ) questions for Punjabi
Target: 25,800 pairs (12.9% of 200,000)
"""
import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.generate_s1_spelling import get_punjabi_characters  # noqa: E402
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
    ("ਪਹਿਲਾ", 1),
    ("ਦੂਜਾ", 2),
    ("ਤੀਜਾ", 3),
    ("ਚੌਥਾ", 4),
    ("ਪੰਜਵਾਂ", 5),
    ("ਛੇਵਾਂ", 6),
    ("ਸੱਤਵਾਂ", 7),
    ("ਅੱਠਵਾਂ", 8),
    ("ਨੌਂਵਾਂ", 9),
    ("ਦਸਵਾਂ", 10),
]

# Question templates
TEMPLATES = [
    '"{word}" ਦਾ {position} ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ {position} ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਸ਼ਬਦ ਦਾ {position} ਅੱਖਰ ਕੀ ਹੈ?',
    '"{word}" ਵਿੱਚ {position} ਸਥਾਨ ਤੇ ਕਿਹੜਾ ਅੱਖਰ ਹੈ?',
    '"{word}" ਦਾ {position} ਅੱਖਰ ਦੱਸੋ?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_punjabi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        for pos_name, pos_num in POSITIONS:
            if pos_num <= len(clusters):
                query = template.format(word=word, position=pos_name)
                answer = clusters[pos_num - 1]  # 0-indexed
                key = (word, template_idx, pos_num)
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
        template = random.choice(TEMPLATES)
        pos_name, pos_num = random.choice(POSITIONS)
        if pos_num <= len(clusters):
            query = template.format(word=word, position=pos_name)
            answer = clusters[pos_num - 1]
            samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S2 Punjabi Letter Position: Generated {len(samples)} samples")
