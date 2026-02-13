#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (संख्या स्पेलिंग) questions
Target: 10,000 pairs (5% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.generate_s1_spelling import (  # noqa: E402
    get_marathi_grapheme_clusters,
)
from group1_marathi.marathi_vocabulary import NUMBERS  # noqa: E402
from prompt_utils import format_qa_pair_marathi  # noqa: E402

# Question templates - two types: number to name, and name to spelling
TEMPLATES_NAME = [
    "{num} ची वर्तनी काय आहे?",
    "{num} चे नाव काय आहे?",
    "{num} ला मराठीत काय म्हणतात?",
    "{num} चे मराठी नाव काय आहे?",
    "{num} या संख्येचे नाव काय आहे?",
]

TEMPLATES_SPELLING = [
    '"{word}" ची वर्तनी काय आहे?',
    '"{word}" कसे लिहायचे?',
    '"{word}" ची अक्षरे काय आहेत?',
    '"{word}" ची वर्तनी सांगा?',
    '"{word}" शब्दाची वर्तनी काय आहे?',
]

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
    chars = get_marathi_grapheme_clusters(word)
    if len(chars) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(chars)
        key = (word, template_idx, "spelling")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    if random.random() < 0.5:
        # Number to name
        num = random.randint(1, 100)
        if num <= len(NUMBERS):
            word = NUMBERS[num - 1]
            template = random.choice(TEMPLATES_NAME)
            query = template.format(num=num)
            answer = word
            samples.append((query, answer))
    else:
        # Name to spelling
        word = random.choice(NUMBERS)
        chars = get_marathi_grapheme_clusters(word)
        if len(chars) > 0:
            template = random.choice(TEMPLATES_SPELLING)
            query = template.format(word=word)
            answer = ", ".join(chars)
            samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S8 Number Spelling: Generated {len(samples)} samples")
