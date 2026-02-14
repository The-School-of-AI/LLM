#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (చివరి అక్షరం) questions - Telugu
Target: 18,000 pairs (9% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import get_telugu_grapheme_clusters  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Templates — Telugu uses invariant postpositions (no genitive suffix needed)
TEMPLATES = [
    '"{word}" లోని చివరి అక్షరం ఏమిటి?',
    '"{word}" యొక్క చివరి అక్షరం ఏమిటి?',
    '"{word}" పదం ఏ అక్షరంతో అంతమవుతుంది?',
    '"{word}" పదం యొక్క ఆఖరి అక్షరం చెప్పండి?',
    '"{word}" లో చివరి అక్షరం ఏది?',
    '"{word}" పదం చివరన ఏ అక్షరం ఉంది?',
    '"{word}" పదం యొక్క అంతిమ అక్షరం ఏమిటి?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 18000

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_telugu_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    last_cluster = clusters[-1]
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = last_cluster
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_telugu_grapheme_clusters(word)
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
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S9 Last Letter (Telugu): Generated {len(samples)} samples")
