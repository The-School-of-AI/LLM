#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (ਵਰਤਨੀ) questions for Punjabi
Target: 28,600 pairs (14.3% of 200,000)
"""
import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Punjabi question templates for spelling
TEMPLATES = [
    '"{word}" ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?',
    '"{word}" ਨੂੰ ਕਿਵੇਂ ਲਿਖਦੇ ਹਨ?',
    '"{word}" ਦੇ ਅੱਖਰ ਕੀ ਹਨ?',
    '"{word}" ਦੀ ਵਰਤਨੀ ਦੱਸੋ?',
    '"{word}" ਸ਼ਬਦ ਦੀ ਵਰਤਨੀ ਕੀ ਹੈ?',
    '"{word}" ਦਾ ਸਹੀ ਸ਼ਬਦ ਜੋੜ ਕੀ ਹੈ?',
    '"{word}" ਨੂੰ ਕਿਵੇਂ ਲਿਖਿਆ ਜਾਂਦਾ ਹੈ?',
    '"{word}" ਦੀ ਵਰਤਨੀ ਲਿਖੋ?',
    '"{word}" ਦਾ ਸ਼ਬਦ ਜੋੜ ਕੀ ਹੁੰਦਾ ਹੈ?',
    '"{word}" ਸ਼ਬਦ ਦਾ ਸਹੀ ਰੂਪ ਕੀ ਹੈ?',
    '"{word}" ਨੂੰ ਕਿਵੇਂ ਜੋੜਿਆ ਜਾਂਦਾ ਹੈ?',
    '"{word}" ਦੀ ਸਹੀ ਵਰਤਨੀ ਦੱਸੋ?',
]


def get_punjabi_characters(word: str) -> list[str]:
    """
    Break down a Punjabi word into its constituent Unicode characters.
    """
    return list(word)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as comma-separated characters"""
    chars = get_punjabi_characters(word)
    return ", ".join(chars)


all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 28600

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
    # Use all unique combinations, then randomly sample with replacement
    samples = list(unique_combinations.values())
    unique_word_list = list(set(all_words))
    while len(samples) < target_count:
        word = random.choice(unique_word_list)
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

print(f"S1 Punjabi Spelling: Generated {len(samples)} samples")
