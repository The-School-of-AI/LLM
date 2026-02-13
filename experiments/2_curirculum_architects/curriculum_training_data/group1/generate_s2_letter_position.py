#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

WORDS = [
    "cat",
    "dog",
    "tiger",
    "horse",
    "elephant",
    "table",
    "chair",
    "apple",
    "orange",
    "computer",
    "keyboard",
] * 50

templates = [
    'Can you say the {pos} letter in "{word}"?',
    'Tell me the {pos} letter of "{word}"',
    'Give me the {pos} letter of "{word}"',
    'What is the {pos} letter in "{word}"?',
]

positions = {
    1: ["first", "1st"],
    2: ["second", "2nd"],
    3: ["third", "3rd"],
    4: ["fourth", "4th"],
    5: ["fifth", "5th"],
    6: ["sixth", "6th"],
    7: ["seventh", "7th"],
    8: ["eighth", "8th"],
    9: ["ninth", "9th"],
}

samples = {}
attempts = 0

while len(samples) < 9000 and attempts < 100000:
    attempts += 1
    word = random.choice(WORDS)
    pos_num = random.randint(1, min(len(word), 9))
    pos_word = random.choice(positions.get(pos_num, [str(pos_num)]))
    template = random.choice(templates)
    query = template.format(pos=pos_word, word=word)
    if query not in samples:
        answer = word[pos_num - 1].lower()
        samples[query] = answer

with open("group1_s2.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S2 Letter Position: Generated {len(samples)} samples")
