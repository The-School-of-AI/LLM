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
    "hospital",
] * 30

templates = [
    'How many alphabets are there in "{word}"?',
    'Count the letters in "{word}"',
    'Letter count of "{word}"',
    'How many letters are in "{word}"?',
    'What is the length of word "{word}"?',
    'How many letters does "{word}" have?',
]

samples = {}
attempts = 0

while len(samples) < 9000 and attempts < 100000:
    attempts += 1
    word = random.choice(WORDS)
    template = random.choice(templates)
    query = template.format(word=word)
    if query not in samples:
        answer = str(len(word))
        samples[query] = answer

with open("group1_s4.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S4 Letter Count: Generated {len(samples)} samples")
