#!/usr/bin/env python3
"""
Generate Statement 2: Letter at Position (अक्षर स्थिती) questions
Target: 25,800 pairs (12.9% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import (  # noqa: E402
    format_qa_pair_marathi,
    get_marathi_grapheme_clusters,
)

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Position names in Marathi
POSITIONS = [
    ("पहिले", 1),
    ("दुसरे", 2),
    ("तिसरे", 3),
    ("चौथे", 4),
    ("पाचवे", 5),
    ("सहावे", 6),
    ("सातवे", 7),
    ("आठवे", 8),
    ("नववे", 9),
    ("दहावे", 10),
]

# Question templates
TEMPLATES = [
    '"{word}" चे {position} अक्षर काय आहे?',
    '"{word}" मध्ये {position} अक्षर काय आहे?',
    '"{word}" शब्दाचे {position} अक्षर काय आहे?',
    '"{word}" मध्ये {position} स्थानावर कोणते अक्षर आहे?',
    '"{word}" चे {position} अक्षर सांगा?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_marathi_grapheme_clusters(word)
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

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_marathi_grapheme_clusters(word)
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
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S2 Letter Position: Generated {len(samples)} samples")
