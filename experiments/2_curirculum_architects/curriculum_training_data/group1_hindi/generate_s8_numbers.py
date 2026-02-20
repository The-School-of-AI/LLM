#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (संख्या वर्तनी) questions
Target: 10,000 pairs (5% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.generate_s1_spelling import get_hindi_characters  # noqa: E402
from group1_hindi.hindi_vocabulary import NUMBERS  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Question templates - two types: number to name, and name to spelling
TEMPLATES_NAME = [
    "{num} की वर्तनी क्या है?",
    "{num} का नाम क्या है?",
    "{num} को हिंदी में क्या कहते हैं?",
    "{num} का हिंदी नाम क्या है?",
    "{num} की संख्या का नाम क्या है?",
]

TEMPLATES_SPELLING = [
    '"{word}" की वर्तनी क्या है?',
    '"{word}" को कैसे लिखते हैं?',
    '"{word}" के अक्षर क्या हैं?',
    '"{word}" का वर्तनी बताइए?',
    '"{word}" शब्द की वर्तनी क्या है?',
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
    chars = get_hindi_characters(word)
    if len(chars) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(chars)
        key = (word, template_idx, "spelling")
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

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(
    f"S8 Number Spelling: Generated {len(samples)} unique samples (target: {target_count})"
)
