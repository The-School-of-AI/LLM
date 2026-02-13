#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (ਸੰਖਿਆ ਵਰਤਨੀ) questions for Punjabi
Target: 10,000 pairs (5% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import NUMBERS  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Question templates - two types: number to name, and name to spelling
TEMPLATES_NAME = [
    "{num} ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?",
    "{num} ਦਾ ਨਾਮ ਕੀ ਹੈ?",
    "{num} ਨੂੰ ਪੰਜਾਬੀ ਵਿੱਚ ਕੀ ਕਹਿੰਦੇ ਹਨ?",
    "{num} ਦਾ ਪੰਜਾਬੀ ਨਾਮ ਕੀ ਹੈ?",
    "{num} ਦੀ ਸੰਖਿਆ ਦਾ ਨਾਮ ਕੀ ਹੈ?",
]

TEMPLATES_SPELLING = [
    '"{word}" ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?',
    '"{word}" ਨੂੰ ਕਿਵੇਂ ਲਿਖਦੇ ਹਨ?',
    '"{word}" ਦੇ ਅੱਖਰ ਕੀ ਹਨ?',
    '"{word}" ਦੀ ਵਰਤਨੀ ਦੱਸੋ?',
    '"{word}" ਸ਼ਬਦ ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?',
]


def get_punjabi_characters(word: str) -> list[str]:
    """Break down a Punjabi word into its constituent Unicode characters."""
    return list(word)


samples = []
target_count = 10000
unique_combinations = {}

# Generate samples for number to name
for num in range(1, 101):  # 1 to 100
    if num <= len(NUMBERS):
        word = NUMBERS[num - 1]
    else:
        continue

    for template_idx, template in enumerate(TEMPLATES_NAME):
        query = template.format(num=num)
        answer = word
        key = (num, template_idx, "name")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Generate samples for name to spelling
for word in NUMBERS:
    chars = get_punjabi_characters(word)
    if len(chars) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(chars)
        key = (word, template_idx, "spelling")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement
samples = list(unique_combinations.values())
if len(samples) > target_count:
    samples = samples[:target_count]
else:
    while len(samples) < target_count:
        if random.random() < 0.5:
            num = random.randint(1, 100)
            if num <= len(NUMBERS):
                word = NUMBERS[num - 1]
                template = random.choice(TEMPLATES_NAME)
                query = template.format(num=num)
                answer = word
                samples.append((query, answer))
        else:
            word = random.choice(NUMBERS)
            chars = get_punjabi_characters(word)
            if len(chars) > 0:
                template = random.choice(TEMPLATES_SPELLING)
                query = template.format(word=word)
                answer = ", ".join(chars)
                samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S8 Punjabi Number Spelling: Generated {len(samples)} samples")
