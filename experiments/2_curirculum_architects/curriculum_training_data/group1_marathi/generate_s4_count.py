#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (अक्षर गणना) questions
Target: 25,800 pairs (12.9% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.generate_s1_spelling import (  # noqa: E402
    get_marathi_grapheme_clusters,
)
from group1_marathi.marathi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_marathi  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Question templates
TEMPLATES = [
    '"{word}" मध्ये किती अक्षरे आहेत?',
    '"{word}" शब्दात किती अक्षरे आहेत?',
    '"{word}" मध्ये अक्षरांची संख्या काय आहे?',
    '"{word}" मध्ये एकूण किती अक्षरे आहेत?',
    '"{word}" शब्दात किती अक्षरे असतात?',
    '"{word}" मध्ये अक्षरांची गणना काय आहे?',
    '"{word}" मध्ये किती अक्षरे उपस्थित आहेत?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_marathi_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = str(cluster_count)
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_marathi_grapheme_clusters(word)
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
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S4 Letter Count: Generated {len(samples)} samples")
