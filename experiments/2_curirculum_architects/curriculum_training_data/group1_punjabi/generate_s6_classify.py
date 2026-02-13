#!/usr/bin/env python3
"""
Generate Statement 6: Classification (ਸ਼੍ਰੇਣੀਬੱਧਤਾ) questions for Punjabi
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Question templates
TEMPLATES = [
    '"{word}" ਇੱਕ ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ ਹੈ?',
    '"{word}" ਕੀ ਹੈ, ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ?',
    '"{word}" ਸ਼ਬਦ ਕਿਸ ਸ਼੍ਰੇਣੀ ਵਿੱਚ ਆਉਂਦਾ ਹੈ, ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ?',
    '"{word}" ਨੂੰ ਕਿਸ ਸ਼੍ਰੇਣੀ ਵਿੱਚ ਰੱਖਿਆ ਜਾ ਸਕਦਾ ਹੈ, ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ?',
    '"{word}" ਕਿਸ ਪ੍ਰਕਾਰ ਦੀ ਚੀਜ਼ ਹੈ, ਵਿਅਕਤੀ, ਜਾਨਵਰ ਜਾਂ ਵਸਤੂ?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    # Default to ਵਸਤੂ if not found
    return "ਵਸਤੂ"


samples = []
target_count = 20000
all_words = []
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words.extend(word_list)

# Expand word list
all_words = all_words * 20
unique_combinations = {}

# Generate samples
for word in set(all_words):
    category = classify_word(word)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
if len(samples) > target_count:
    samples = samples[:target_count]
else:
    unique_word_list = list(set(all_words))
    while len(samples) < target_count:
        word = random.choice(unique_word_list)
        category = classify_word(word)
        template = random.choice(TEMPLATES)
        query = template.format(word=word)
        answer = category
        samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S6 Punjabi Classification: Generated {len(samples)} samples")
