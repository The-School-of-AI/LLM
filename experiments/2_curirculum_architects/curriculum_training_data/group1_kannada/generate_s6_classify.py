#!/usr/bin/env python3
"""
Generate Statement 6: Classification (ವರ್ಗೀಕರಣ) questions - Kannada
Target: 20,000 pairs (10% of 200,000)
Uses S6_CLASSIFICATION_VOCABULARY: only words with clear classifications,
with category-appropriate option pairs (e.g. ಆಹಾರ vs ಹಣ್ಣು for food, not ವಾಹನ vs ವಸ್ತು).
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_vocabulary import S6_CLASSIFICATION_VOCABULARY  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Question templates - use {word}, {options} for format()
TEMPLATES = [
    '"{word}" ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ? {options}?',
    '"{word}" ಏನು? {options}?',
    '"{word}" ಪದ ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ, {options}?',
    '"{word}" ಅನ್ನು ಯಾವ ವರ್ಗದಲ್ಲಿ ಇಡಬಹುದು? {options}?',
    '{options} — "{word}" ಯಾವ ವರ್ಗ?',
    '"{word}" ಎಂಬ ಪದದ ವರ್ಗೀಕರಣ ಏನು? {options}?',
]

# Build (word, category, options) from S6 vocabulary
classification_items = []
for category, (word_list, option_pairs) in S6_CLASSIFICATION_VOCABULARY.items():
    for word in word_list:
        if not word:
            continue
        opt1, opt2 = random.choice(option_pairs)
        options_str = f"{opt1} ಅಥವಾ {opt2}"
        classification_items.append((word, category, options_str))

samples = []
target_count = 10000
unique_combinations = {}

for word, category, options_str in classification_items:
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word, options=options_str)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
seen_qa = set((q, a) for q, a in samples)
all_items = classification_items * 20
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    word, category, options_str = random.choice(all_items)
    template = random.choice(TEMPLATES)
    query = template.format(word=word, options=options_str)
    answer = category
    if (query, answer) not in seen_qa:
        seen_qa.add((query, answer))
        samples.append((query, answer))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S6 Classification (Kannada): Generated {len(samples)} samples")
