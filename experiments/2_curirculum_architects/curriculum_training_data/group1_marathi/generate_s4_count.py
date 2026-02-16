#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (अक्षर गणना) questions
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

# Question templates
TEMPLATES = [
    '"{word}" मध्ये किती अक्षरे आहेत?',
    '"{word}" शब्दात किती अक्षरे आहेत?',
    '"{word}" मध्ये अक्षरांची संख्या काय आहे?',
    '"{word}" मध्ये एकूण किती अक्षरे आहेत?',
    '"{word}" शब्दात किती अक्षरे असतात?',
    '"{word}" मध्ये अक्षरांची गणना काय आहे?',
    '"{word}" मध्ये किती अक्षरे उपस्थित आहेत?',
    '"{word}" या शब्दात किती अक्षरे भरली आहेत?',
    '"{word}" मध्ये अक्षरांची संख्या सांगा?',
    '"{word}" मध्ये एकूण अक्षरे किती?',
    '"{word}" या शब्दातील अक्षरांची गणना सांगा?',
    '"{word}" शब्दाची लांबी (अक्षरांत) किती आहे?',
]

all_words_set = set(EASY_WORDS_UNIQUE + MEDIUM_WORDS_UNIQUE + HARD_WORDS_UNIQUE)
samples = []
target_count = 25800

# Generate all possible unique combinations
unique_combinations = []
all_words_list = list(all_words_set)
random.shuffle(all_words_list)

for word in all_words_list:
    clusters = get_marathi_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    # Shuffle templates for each word
    current_templates = list(enumerate(TEMPLATES))
    random.shuffle(current_templates)

    for template_idx, template in current_templates:
        query = template.format(word=word)
        answer = str(cluster_count)
        unique_combinations.append((query, answer))
        if len(unique_combinations) >= target_count:
            break
    if len(unique_combinations) >= target_count:
        break

samples = unique_combinations

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S4 Letter Count: Generated {len(samples)} samples")
