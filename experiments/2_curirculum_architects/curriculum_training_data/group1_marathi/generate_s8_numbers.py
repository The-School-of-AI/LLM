#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (संख्या स्पेलिंग) questions
Target: 10,000 pairs (5% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
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


def get_marathi_number_name(n: int) -> str:
    """Algorithmic Marathi number naming up to 1000"""
    if n <= 100:
        return NUMBERS[n - 1]
    if n == 1000:
        return "एक हजार"

    hundreds = n // 100
    rem = n % 100

    hundreds_names = ["", "एक", "दोन", "तीन", "चार", "पाच", "सहा", "सात", "आठ", "नऊ"]

    if rem == 0:
        if hundreds == 1:
            return "शंभर"
        return hundreds_names[hundreds] + "शे"
    else:
        prefix = "एकशे" if hundreds == 1 else hundreds_names[hundreds] + "शे"
        return prefix + " " + NUMBERS[rem - 1]


samples = []
target_count = 10000
unique_combinations = []

# Generate all numbers from 1 to 1000
all_numbers = list(range(1, 1001))
random.shuffle(all_numbers)

for num in all_numbers:
    word = get_marathi_number_name(num)

    # 1. Number to Name (5 templates)
    # 2. Name to Spelling (5 templates)

    # Mix of templates to reach target 10000 with 1000 numbers
    # Each number gets 10 possible combinations (5 name + 5 spelling)
    # Total 10,000 unique combinations available

    # Name queries
    for template_idx, template in enumerate(TEMPLATES_NAME):
        query = template.format(num=num)
        answer = word
        unique_combinations.append((query, answer))

    # Spelling queries
    chars = list(word.replace(" ", ""))  # Remove spaces for spelling breakdown
    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(chars)
        unique_combinations.append((query, answer))

    if len(unique_combinations) >= target_count:
        break

samples = unique_combinations[:target_count]

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S8 Number Spelling: Generated {len(samples)} samples")
