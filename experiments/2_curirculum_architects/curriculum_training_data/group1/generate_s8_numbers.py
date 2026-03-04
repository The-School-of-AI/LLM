#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

num_words = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty-one",
    25: "twenty-five",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
    100: "one hundred",
}

templates = [
    "What is the spelling of {num}?",
    "Spell {num}",
    "How do you spell {num}?",
    "Write the spelling of {num}",
    "What's the spelling of {num}?",
    "Can you spell {num}?",
]

samples = {}
attempts = 0
numbers = list(num_words.keys()) * 50

while len(samples) < 3500 and attempts < 100000:
    attempts += 1
    num = random.choice(numbers)
    num_str = str(num) if random.choice([True, False]) else num_words[num]
    template = random.choice(templates)
    query = template.format(num=num_str)
    if query not in samples:
        answer = ", ".join(num_words[num])
        samples[query] = answer

with open("group1_s8.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S8 Number Spelling: Generated {len(samples)} samples")
