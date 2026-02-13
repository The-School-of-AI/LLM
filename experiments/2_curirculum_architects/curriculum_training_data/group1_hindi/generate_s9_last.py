#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (अंतिम अक्षर) questions
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

# Question templates
TEMPLATES = [
    '"{word}" का अंतिम अक्षर क्या है?',
    '"{word}" किस अक्षर से समाप्त होता है?',
    '"{word}" शब्द का अंतिम अक्षर क्या है?',
    '"{word}" का आखिरी अक्षर क्या है?',
    '"{word}" किस अक्षर पर खत्म होता है?',
    '"{word}" का अंत में कौन सा अक्षर है?',
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
    
    last_cluster = clusters[-1]
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = last_cluster
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_hindi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue
    
    last_cluster = clusters[-1]
    template = random.choice(TEMPLATES)
    query = template.format(word=word)
    answer = last_cluster
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S9 Last Letter: Generated {len(samples)} samples")
