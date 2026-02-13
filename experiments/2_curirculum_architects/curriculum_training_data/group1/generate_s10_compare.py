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
    "sheep",
    "table",
    "chair",
    "phone",
    "apple",
    "bread",
    "elephant",
    "giraffe",
    "computer",
    "keyboard",
] * 20

templates = [
    "Which word is longer, '{w1}' or '{w2}'?",
    "Is '{w1}' longer than '{w2}'?",
    "Compare the length of '{w1}' and '{w2}'",
    "Which is the longer word: '{w1}' or '{w2}'?",
    "Tell me which word has more letters, '{w1}' or '{w2}'",
]

samples = {}
attempts = 0

while len(samples) < 5500 and attempts < 100000:
    attempts += 1
    w1 = random.choice(WORDS)
    w2 = random.choice(WORDS)
    if w1 == w2:
        continue
    template = random.choice(templates)
    query = template.format(w1=w1, w2=w2)
    if query not in samples:
        if len(w1) > len(w2):
            answer = w1
        elif len(w2) > len(w1):
            answer = w2
        else:
            answer = random.choice(["equal", "both are equal"])
        samples[query] = answer

with open("group1_s10.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S10 Word Comparison: Generated {len(samples)} samples")
