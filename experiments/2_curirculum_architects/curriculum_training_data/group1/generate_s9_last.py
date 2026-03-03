#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

WORDS = [
    "cat",
    "dog",
    "rat",
    "bat",
    "tiger",
    "horse",
    "mouse",
    "table",
    "chair",
    "phone",
    "apple",
    "elephant",
    "computer",
    "keyboard",
] * 30

templates = [
    "What letter does '{word}' end with?",
    "What is the last letter of '{word}'?",
    "Which letter does '{word}' end with?",
    "Tell me the ending letter of '{word}'",
    "Find the final letter in '{word}'",
    "What's the last letter of '{word}'?",
]

samples = {}
attempts = 0

while len(samples) < 6000 and attempts < 100000:
    attempts += 1
    word = random.choice(WORDS)
    template = random.choice(templates)
    query = template.format(word=word)
    if query not in samples:
        answer = word[-1].lower()
        samples[query] = answer

with open("group1_s9.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S9 Last Letter: Generated {len(samples)} samples")
