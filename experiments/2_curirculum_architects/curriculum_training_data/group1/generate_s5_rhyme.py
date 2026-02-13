#!/usr/bin/env python3
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair  # noqa: E402

rhymes = [
    ("cat", "bat"),
    ("cat", "hat"),
    ("cat", "mat"),
    ("cat", "rat"),
    ("dog", "log"),
    ("dog", "fog"),
    ("sun", "run"),
    ("sun", "fun"),
    ("bed", "red"),
    ("bed", "fed"),
    ("pen", "ten"),
    ("pen", "den"),
    ("box", "fox"),
    ("tree", "bee"),
    ("tree", "see"),
    ("moon", "soon"),
    ("moon", "noon"),
    ("book", "look"),
    ("book", "cook"),
    ("cake", "lake"),
    ("cake", "make"),
    ("cake", "take"),
    ("night", "light"),
    ("night", "sight"),
    ("night", "bright"),
    ("train", "rain"),
    ("train", "brain"),
    ("house", "mouse"),
    ("ball", "call"),
    ("ball", "fall"),
    ("bell", "fell"),
    ("bell", "tell"),
    ("hill", "fill"),
    ("hill", "will"),
    ("block", "clock"),
    ("block", "rock"),
] * 100

templates = [
    "What word rhymes with '{word}'?",
    "Tell me a word that rhymes with '{word}'",
    "Give me a rhyming word for '{word}'",
    "Find a rhyme for '{word}'",
    "What rhymes with '{word}'?",
    "Can you give me a word that rhymes with '{word}'?",
]

samples = {}
attempts = 0

while len(samples) < 7000 and attempts < 100000:
    attempts += 1
    word, rhyme = random.choice(rhymes)
    template = random.choice(templates)
    query = template.format(word=word)
    if query not in samples:
        samples[query] = rhyme

with open("group1_s5.txt", "w", encoding="utf-8") as f:
    for query, answer in samples.items():
        f.write(format_qa_pair(query, answer) + "\n")
print(f"S5 Rhyming: Generated {len(samples)} samples")
