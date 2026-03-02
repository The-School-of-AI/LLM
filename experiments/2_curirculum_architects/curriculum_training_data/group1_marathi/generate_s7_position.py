#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (अक्षराची स्थिती) questions
Target: 17,200 pairs (8.6% of 200,000)
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
    ("पहिले", "1"),
    ("दुसरे", "2"),
    ("तिसरे", "3"),
    ("चौथे", "4"),
    ("पाचवे", "5"),
    ("सहावे", "6"),
    ("सातवे", "7"),
    ("आठवे", "8"),
    ("नववे", "9"),
    ("दहावे", "10"),
]

# Question templates
TEMPLATES = [
    '"{word}" मध्ये "{char}" अक्षर कोणत्या स्थानावर आहे?',
    '"{word}" मध्ये "{char}" अक्षर कुठे आहे?',
    '"{word}" शब्दात "{char}" अक्षर कोणत्या स्थानावर आहे?',
    '"{word}" मध्ये "{char}" कोणत्या स्थानावर मिळते?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 17200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_marathi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    # Iterate through grapheme clusters (not Unicode characters)
    for cluster in clusters:
        cluster_positions = [i + 1 for i, c in enumerate(clusters) if c == cluster]
        if not cluster_positions:
            continue

        # Use first occurrence
        pos_num = cluster_positions[0]
        if pos_num <= len(POSITIONS):
            pos_name, pos_str = POSITIONS[pos_num - 1]
        else:
            pos_name = f"{pos_num}वे"
            pos_str = str(pos_num)

        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word, char=cluster)
            # Use word form or numeric form randomly
            answer = pos_name if random.random() < 0.5 else pos_str
            key = (word, cluster, template_idx)
            if key not in unique_combinations:
                unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_marathi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    cluster = random.choice(clusters)
    cluster_positions = [i + 1 for i, c in enumerate(clusters) if c == cluster]
    if not cluster_positions:
        continue

    pos_num = cluster_positions[0]
    if pos_num <= len(POSITIONS):
        pos_name, pos_str = POSITIONS[pos_num - 1]
    else:
        pos_name = f"{pos_num}वे"
        pos_str = str(pos_num)

    template = random.choice(TEMPLATES)
    query = template.format(word=word, char=cluster)
    answer = pos_name if random.random() < 0.5 else pos_str
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S7 Position of Letter: Generated {len(samples)} samples")
