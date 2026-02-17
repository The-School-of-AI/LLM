#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (বানান) questions for Assamese
Target: 28,600 pairs (14.3% of 200,000)
"""

import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Assamese question templates for spelling (বানান)
TEMPLATES = [
    '"{word}"ৰ বানানটো ক\'ব পাৰিব নেকি?',
    '"{word}"ৰ বানানটো ক\'ব পাৰিবানে?',
    '"{word}"ৰ বানানটো ক\'ব পাৰিবা নেকি?',
    '"{word}"ৰ বানানটো কি?',
    '"{word}"ৰ বানানটো কোৱা?',
    '"{word}"ৰ বানানটো পাৰিবনে?',
    '"{word}"ৰ বানানটো কি হ\'ব বাৰু?',
    '"{word}"ৰ বানানটো কেনেকৈ লিখে?',
    '"{word}"ৰ বানানটো জানেনে?',
    '"{word}" শব্দটোৰ বানানটো কওক?',
    '"{word}" শব্দটোৰ বানানটো কোৱা?',
    '"{word}" শব্দটোৰ বানানটো কি?',
    '"{word}" শব্দটোৰ বানানটো পাৰিবনে?',
    '"{word}" শব্দটোৰ বানানটো কি হ\'ব বাৰু?',
    '"{word}"টো লিখি দেখুৱাব পাৰিব নেকি?',
    '"{word}" শব্দটো কেনেকৈ লিখিব?',
    '"{word}"ৰ শুদ্ধ বানানটো কি?',
]


def get_assamese_characters(word: str) -> list[str]:
    """
    Break down an Assamese word into its constituent Unicode characters.
    Each Unicode character (consonant, vowel, matra) is separate.
    Used for: Spelling questions (S1, S8)
    """
    return list(word)


def get_assamese_grapheme_clusters(word: str) -> list[str]:
    """
    Get grapheme clusters for Assamese word (for counting/length/position).
    Uses regex library's \\X pattern (Unicode UAX#29 compliant).
    Used for: Counting, length, and position questions (S2, S4, S7, S9, S10)
    """
    return regex.findall(r"\X", word)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as comma-separated characters"""
    chars = get_assamese_characters(word)
    return ", ".join(chars)


all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25000

# Generate all unique combinations first
unique_combinations = {}
for word in set(all_words):
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = generate_spelling_answer(word)
        unique_combinations[(word, template_idx)] = (query, answer)

# If we have enough unique combinations, use them
if len(unique_combinations) >= target_count:
    samples = list(unique_combinations.values())[:target_count]
else:
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
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S1 Spelling: Generated {len(samples)} samples")
