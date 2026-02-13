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

# Question templates (ವ್ಯಕ್ತಿ=person, ಪ್ರಾಣಿ=animal, ವಸ್ತು=object). Varied phrasings.
TEMPLATES = [
    '"{word}" ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು?',
    '"{word}" ಏನು, ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು?',
    '"{word}" ಪದ ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ, ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು?',
    '"{word}" ಅನ್ನು ಯಾವ ವರ್ಗದಲ್ಲಿ ಇಡಬಹುದು, ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು?',
    '"{word}" ಯಾವ ರೀತಿಯ ವಸ್ತು, ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ ಅಥವಾ ವಸ್ತು?',
    'ವ್ಯಕ್ತಿ, ಪ್ರಾಣಿ, ವಸ್ತು — "{word}" ಯಾವ ವರ್ಗ?',
    '"{word}" ಎಂಬ ಪದದ ವರ್ಗೀಕರಣ ಏನು?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    return "ವಸ್ತು"


samples = []
target_count = 20000
all_words = []
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words.extend(word_list)

all_words = all_words * 20
unique_combinations = {}

for word in set(all_words):
    category = classify_word(word)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    category = classify_word(word)
    template = random.choice(TEMPLATES)
    query = template.format(word=word)
    answer = category
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S6 Classification (Kannada): Generated {len(samples)} samples")
