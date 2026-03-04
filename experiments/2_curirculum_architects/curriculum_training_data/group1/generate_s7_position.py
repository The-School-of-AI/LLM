#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

WORDS = ["cat", "dog", "tiger", "banana", "elephant", "table", "computer"] * 50

templates = [
    "At what location does letter '{letter}' come in '{word}'?",
    "Where is the letter '{letter}' in '{word}'?",
    "What position is '{letter}' in '{word}'?",
    "Find the position of letter '{letter}' in '{word}'",
    "What is the first position of '{letter}' in '{word}'?",
]

samples = {}
attempts = 0

while len(samples) < 6000 and attempts < 100000:
    attempts += 1
    word = random.choice(WORDS)
    letter = random.choice(list(word.lower()))
    positions = [i + 1 for i, c in enumerate(word.lower()) if c == letter]
    if not positions:
        continue
    template = random.choice(templates)
    query = template.format(letter=letter, word=word)
    if query not in samples:
        samples[query] = str(positions[0])

with open("group1_s7.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S7 Position of Letter: Generated {len(samples)} samples")
