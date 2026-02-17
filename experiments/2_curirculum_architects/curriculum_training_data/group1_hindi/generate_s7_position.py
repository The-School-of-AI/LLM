#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (अक्षर की स्थिति) questions
Target: 17,200 pairs (8.6% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.generate_s1_spelling import get_hindi_grapheme_clusters  # noqa: E402
from group1_hindi.hindi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Position names in Hindi
POSITIONS = [
    ("पहला", "1"),
    ("दूसरा", "2"),
    ("तीसरा", "3"),
    ("चौथा", "4"),
    ("पांचवां", "5"),
    ("छठा", "6"),
    ("सातवां", "7"),
    ("आठवां", "8"),
    ("नौवां", "9"),
    ("दसवां", "10"),
]

# Question templates
TEMPLATES = [
    '"{word}" में "{char}" अक्षर किस स्थान पर है?',
    '"{word}" में "{char}" अक्षर कहाँ है?',
    '"{word}" शब्द में "{char}" अक्षर किस स्थान पर है?',
    '"{word}" में "{char}" किस स्थान पर मिलता है?',
    'बताइए "{word}" में "{char}" किस स्थान पर है?',
    '"{word}" में "{char}" का स्थान क्या है?',
    '"{word}" शब्द में "{char}" का स्थान बताइए?',
    '"{word}" में "{char}" किस जगह पर है?',
    '"{word}" में "{char}" अक्षर का स्थान क्या है?',
    '"{word}" में "{char}" किस नंबर पर है?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 17200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_hindi_grapheme_clusters(word)
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
            pos_name = f"{pos_num}वां"
            pos_str = str(pos_num)

        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word, char=cluster)
            # Use word form or numeric form randomly
            answer = pos_name if random.random() < 0.5 else pos_str
            key = (word, cluster, template_idx)
            if key not in unique_combinations:
                unique_combinations[key] = (query, answer)

# Only use unique combinations - NO sampling with replacement
samples = list(unique_combinations.values())
unique_count = len(samples)

if unique_count < target_count:
    print(f"Warning: Only {unique_count} unique combinations (target: {target_count})")
else:
    samples = samples[:target_count]

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(
    f"S7 Position of Letter: Generated {len(samples)} unique samples (target: {target_count})"
)
