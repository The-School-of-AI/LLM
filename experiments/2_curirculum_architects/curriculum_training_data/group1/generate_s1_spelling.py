#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

EASY_WORDS = [
    "cat",
    "dog",
    "rat",
    "bat",
    "bee",
    "cow",
    "pig",
    "fox",
    "owl",
    "ant",
    "hat",
    "mat",
    "sat",
    "pat",
    "bed",
    "red",
    "pen",
    "sun",
    "run",
    "cup",
    "box",
    "key",
    "toy",
    "car",
    "bag",
    "egg",
] * 20
MEDIUM_WORDS = [
    "tiger",
    "horse",
    "mouse",
    "sheep",
    "whale",
    "table",
    "chair",
    "phone",
    "apple",
    "bread",
    "water",
    "cloud",
    "house",
    "train",
] * 30
HARD_WORDS = [
    "elephant",
    "giraffe",
    "penguin",
    "computer",
    "keyboard",
    "hospital",
    "chocolate",
    "sandwich",
    "butterfly",
] * 40

templates = [
    'What is the spelling of "{word}"?',
    'How do you spell "{word}"?',
    'Can you spell "{word}"?',
    'Write the spelling of "{word}".',
    'Spell "{word}".',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = {}
attempts = 0

while len(samples) < 10000 and attempts < 100000:
    attempts += 1
    word = random.choice(all_words)
    template = random.choice(templates)
    query = template.format(word=word)
    if query not in samples:
        answer = ", ".join(word.lower())
        samples[query] = answer

with open("group1_s1.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S1 Spelling: Generated {len(samples)} samples")
