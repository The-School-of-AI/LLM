#!/usr/bin/env python3
"""
Generate Statement 6: Classification (ವರ್ಗೀಕರಣ) questions - Kannada
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Related category pairs for 2-option questions (each pair is semantically related)
RELATED_PAIRS = [
    ("ಹಣ್ಣು", "ತರಕಾರಿ"),
    ("ಹೂವು", "ಪ್ರಕೃತಿ"),
    ("ಪ್ರಾಣಿ", "ಪಕ್ಷಿ"),
    ("ವಾಹನ", "ವಸ್ತು"),
    ("ಸ್ಥಳ", "ಪ್ರಕೃತಿ"),
    ("ವ್ಯಕ್ತಿ", "ವಸ್ತು"),
    ("ಹಣ್ಣು", "ಹೂವು"),
    ("ತರಕಾರಿ", "ಹಣ್ಣು"),
    ("ಪ್ರಾಣಿ", "ವಾಹನ"),
    ("ಸ್ಥಳ", "ವಸ್ತು"),
]


def get_two_options(category: str) -> str:
    """Pick 2 related categories including the word's category."""
    for c1, c2 in RELATED_PAIRS:
        if category == c1:
            return f"{c1} ಅಥವಾ {c2}"
        if category == c2:
            return f"{c1} ಅಥವಾ {c2}"
    return f"{category} ಅಥವಾ ವಸ್ತು"


# Question templates - use 2 related categories per question. Use {word}, {options} for format().
TEMPLATES = [
    '"{word}" ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ? {options}?',
    '"{word}" ಏನು? {options}?',
    '"{word}" ಪದ ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ, {options}?',
    '"{word}" ಅನ್ನು ಯಾವ ವರ್ಗದಲ್ಲಿ ಇಡಬಹುದು? {options}?',
    '{options} — "{word}" ಯಾವ ವರ್ಗ?',
    '"{word}" ಎಂಬ ಪದದ ವರ್ಗೀಕರಣ ಏನು? {options}?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    return "ವಸ್ತು"


samples = []
target_count = 10000
all_words = []
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words.extend(word_list)

all_words = all_words * 20
unique_combinations = {}

for word in set(all_words):
    category = classify_word(word)
    options = get_two_options(category)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word, options=options)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    word = random.choice(list(set(all_words)))
    category = classify_word(word)
    options = get_two_options(category)
    template = random.choice(TEMPLATES)
    query = template.format(word=word, options=options)
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
