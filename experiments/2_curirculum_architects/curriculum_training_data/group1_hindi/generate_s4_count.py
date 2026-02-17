#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (अक्षर गिनती) questions
Target: 25,800 pairs (12.9% of 200,000)
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
    '"{word}" में कितने अक्षर हैं?',
    '"{word}" शब्द में कितने अक्षर हैं?',
    '"{word}" में अक्षरों की संख्या क्या है?',
    '"{word}" में कुल कितने अक्षर हैं?',
    '"{word}" शब्द में कितने अक्षर होते हैं?',
    '"{word}" में अक्षरों की गिनती क्या है?',
    '"{word}" में कितने अक्षर मौजूद हैं?',
    'बताइए "{word}" में कितने अक्षर हैं?',
    '"{word}" शब्द में अक्षरों की संख्या बताइए?',
    '"{word}" में अक्षरों की गणना करें?',
    # Additional 10 templates
    '"{word}" के कितने अक्षर हैं?',
    '"{word}" में अक्षर कितने हैं?',
    'बताओ "{word}" में कितने अक्षर हैं?',
    '"{word}" शब्द के अक्षरों की संख्या क्या है?',
    '"{word}" में अक्षरों की संख्या बताओ?',
    '"{word}" के कुल अक्षर कितने हैं?',
    '"{word}" शब्द में कितने अक्षर गिने जाते हैं?',
    '"{word}" का अक्षर गणना क्या है?',
    '"{word}" में कितने वर्ण हैं?',
    '"{word}" शब्द के वर्णों की संख्या क्या है?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 30000  # Increased from 25800 for 200K push

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_hindi_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = str(cluster_count)
        key = (word, template_idx)
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

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(
    f"S4 Letter Count: Generated {len(samples)} unique samples (target: {target_count})"
)
