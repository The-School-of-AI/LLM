#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (स्पेलिंग) questions
Target: 28,600 pairs (14.3% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_marathi

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Marathi question templates for spelling
TEMPLATES = [
    '"{word}" ची वर्तनी काय आहे?',
    '"{word}" कसे लिहायचे?',
    '"{word}" ची अक्षरे काय आहेत?',
    '"{word}" ची वर्तनी सांगा?',
    '"{word}" शब्दाची वर्तनी काय आहे?',
    '"{word}" ची योग्य वर्तनी काय आहे?',
    '"{word}" शब्द कसा लिहिला जातो?',
    '"{word}" ची वर्तनी लिहा?',
    '"{word}" ची वर्तनी काय होते?',
    '"{word}" शब्दाची वर्तनी काय आहे?',
    '"{word}" कशी वर्तनी करायची?',
    '"{word}" ची योग्य वर्तनी सांगा?',
    '"{word}" ची वर्तनी काय आहे?',
    '"{word}" शब्दाची वर्तनी सांगा?',
    '"{word}" कसे लिहिले जाते?',
]


def get_marathi_characters(word: str) -> list[str]:
    """
    Break down a Marathi word into its constituent Unicode characters.
    Each Unicode character (consonant, vowel, matra, anusvara) is separate.
    This matches the spelling format where each character is shown separately.

    Example: "पाणी" → ['प', 'ा', 'ण', 'ी'] (4 Unicode chars)
    Example: "कोंबडी" → ['क', 'ो', 'ं', 'ब', 'ड', 'ी'] (6 Unicode chars)
    Example: "कमळ" → ['क', 'म', 'ळ'] (3 Unicode chars)

    Used for: Spelling questions (S1, S8)
    """
    # Simply return each Unicode character separately
    # This matches the spelling format and token counting logic
    return list(word)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as comma-separated Unicode characters"""
    # Use Unicode characters (including matras as separate chars) as requested
    chars = list(word)
    return ", ".join(chars)


all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 28600

# Generate all unique combinations first
unique_combinations = {}
for word in set(all_words):  # Use unique words
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = generate_spelling_answer(word)
        unique_combinations[(word, template_idx)] = (query, answer)

# If we have enough unique combinations, use them
if len(unique_combinations) >= target_count:
    samples = list(unique_combinations.values())[:target_count]
else:
    # Use all unique combinations, then randomly sample with replacement to reach target
    samples = list(unique_combinations.values())
    while len(samples) < target_count:
        word = random.choice(list(set(all_words)))
        template_idx = random.randint(0, len(TEMPLATES) - 1)
        template = TEMPLATES[template_idx]
        query = template.format(word=word)
        answer = generate_spelling_answer(word)
        samples.append((query, answer))

# Shuffle for randomness
random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S1 Spelling: Generated {len(samples)} samples")
