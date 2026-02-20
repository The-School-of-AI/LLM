#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (शेवटचे अक्षर) questions
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

# Question templates
TEMPLATES = [
    '"{word}" चे शेवटचे अक्षर काय आहे?',
    '"{word}" कोणत्या अक्षराने संपते?',
    '"{word}" शब्दाचे शेवटचे अक्षर काय आहे?',
    '"{word}" चे आखेरचे अक्षर काय आहे?',
    '"{word}" कोणत्या अक्षरावर संपते?',
    '"{word}" च्या शेवटी कोणते अक्षर आहे?',
    '"{word}" या शब्दाचा शेवट कोणत्या अक्षराने होतो?',
    '"{word}" मधील अंतिम अक्षर सांगा?',
    '"{word}" चा अंत कोणत्या अक्षराने होतो?',
    '"{word}" शब्दाचा शेवटचा वर्ण कोणता आहे?',
]

all_words_set = set(EASY_WORDS_UNIQUE + MEDIUM_WORDS_UNIQUE + HARD_WORDS_UNIQUE)
samples = []
target_count = 17200

# Generate all possible unique combinations
unique_combinations = []
all_words_list = list(all_words_set)
random.shuffle(all_words_list)

for word in all_words_list:
    clusters = get_marathi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue
    last_cluster = clusters[-1]

    # Shuffle templates for each word
    current_templates = list(enumerate(TEMPLATES))
    random.shuffle(current_templates)

    for template_idx, template in current_templates:
        query = template.format(word=word)
        answer = last_cluster
        unique_combinations.append((query, answer))
        if len(unique_combinations) >= target_count:
            break
    if len(unique_combinations) >= target_count:
        break

samples = unique_combinations

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S9 Last Letter: Generated {len(samples)} samples")
