#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (సంఖ్య అక్షరక్రమం) questions - Telugu
Target: 12,000 pairs (6% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.telugu_grammar import get_telugu_aksharas  # noqa: E402
from group1_telugu.telugu_vocabulary import NUMBERS  # noqa: E402
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Templates — Telugu uses invariant యొక్క (no genitive suffix needed)
TEMPLATES_NAME = [
    "{num} యొక్క పేరు ఏమిటి?",
    "{num} ను తెలుగులో ఏమంటారు?",
    "{num} సంఖ్య యొక్క పేరు ఏమిటి?",
    "{num} అంకె యొక్క తెలుగు పేరు చెప్పండి?",
    "{num} ను తెలుగులో ఎలా చెబుతారు?",
]

TEMPLATES_SPELLING = [
    '"{word}" పదం యొక్క అక్షరక్రమం ఏమిటి?',
    '"{word}" పదాన్ని ఎలా వ్రాయాలి?',
    '"{word}" పదంలోని అక్షరాలు ఏమిటి?',
    '"{word}" పదం యొక్క అక్షరక్రమం చెప్పండి?',
    '"{word}" అనే సంఖ్య పదాన్ని అక్షరాల వారీగా వ్రాయండి?',
]

samples = []
target_count = 12000
unique_combinations = {}

# Number to name
for num in range(1, 101):
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

# Name to spelling (akshara-level)
for word in NUMBERS:
    aksharas = get_telugu_aksharas(word)
    if len(aksharas) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(aksharas)
        key = (word, template_idx, "spelling")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
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
        aksharas = get_telugu_aksharas(word)
        if len(aksharas) > 0:
            template = random.choice(TEMPLATES_SPELLING)
            query = template.format(word=word)
            answer = ", ".join(aksharas)
            samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S8 Number Spelling (Telugu): Generated {len(samples)} samples")
