#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (సంఖ్య అక్షరక్రమం) questions - Telugu
Target: 12,000 pairs (6% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402
from group1_telugu.telugu_grammar import get_telugu_aksharas_with_roots  # noqa: E402
from group1_telugu.telugu_vocabulary import NUMBERS_EXTENDED  # noqa: E402

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
for num in range(1, len(NUMBERS_EXTENDED) + 1):
    word = NUMBERS_EXTENDED[num - 1]
    for template_idx, template in enumerate(TEMPLATES_NAME):
        query = template.format(num=num)
        answer = word
        key = (num, template_idx, "name")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Name to spelling (root-level)
for word in NUMBERS_EXTENDED:
    roots = get_telugu_aksharas_with_roots(word)
    if len(roots) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ",".join(roots)
        key = (word, template_idx, "spelling")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())

# Track seen lines for dedup
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

max_attempts = target_count * 10
attempts = 0
while len(samples) < target_count and attempts < max_attempts:
    attempts += 1
    q, a = None, None
    if random.random() < 0.5:
        num = random.randint(1, len(NUMBERS_EXTENDED))
        word = NUMBERS_EXTENDED[num - 1]
        template = random.choice(TEMPLATES_NAME)
        q = template.format(num=num)
        a = word
    else:
        word = random.choice(NUMBERS_EXTENDED)
        roots = get_telugu_aksharas_with_roots(word)
        if len(roots) > 0:
            template = random.choice(TEMPLATES_SPELLING)
            q = template.format(word=word)
            a = ",".join(roots)

    if q and a and (q, a) not in seen_lines:
        seen_lines.add((q, a))
        samples.append((q, a))

# Final dedup
unique_samples = []
final_seen = set()
for q, a in samples:
    if (q, a) not in final_seen:
        final_seen.add((q, a))
        unique_samples.append((q, a))
samples = unique_samples

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S8 Number Spelling (Telugu): Generated {len(samples)} samples")
