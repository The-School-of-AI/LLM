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
    '"{word}" शब्दाचा वर्णविच्छेद काय आहे?',
    '"{word}" कशी वर्तनी करायची?',
    '"{word}" ची अचूक वर्तनी सांगा?',
    '"{word}" शब्दाचे स्पेलिंग काय आहे?',
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


all_words_set = set(EASY_WORDS_UNIQUE + MEDIUM_WORDS_UNIQUE + HARD_WORDS_UNIQUE)
samples = []
target_count = 28600

# Generate all possible unique combinations
unique_combinations = []
all_words_list = list(all_words_set)
random.shuffle(all_words_list)

for word in all_words_list:
    # Shuffle templates for each word
    current_templates = list(enumerate(TEMPLATES))
    random.shuffle(current_templates)

    for template_idx, template in current_templates:
        query = template.format(word=word)
        answer = generate_spelling_answer(word)
        unique_combinations.append((query, answer))
        if len(unique_combinations) >= target_count:
            break
    if len(unique_combinations) >= target_count:
        break

samples = unique_combinations

# Shuffle for randomness
random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S1 Spelling: Generated {len(samples)} samples")
