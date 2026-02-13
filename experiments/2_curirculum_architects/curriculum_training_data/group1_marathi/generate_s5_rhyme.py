#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (यमक) questions
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    RHYMING_PAIRS,
)
from prompt_utils import format_qa_pair_marathi  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates (multiple choice format)
TEMPLATES = [
    '"{word}" शी यमक करणारा शब्द कोणता आहे, "{rhyme}" किंवा "{non_rhyme}"?',
    '"{word}" शी कोणता शब्द यमक करतो, "{rhyme}" किंवा "{non_rhyme}"?',
    'कोणता शब्द "{word}" शी यमक करतो, "{rhyme}" किंवा "{non_rhyme}"?',
    '"{word}" सोबत यमक करणारा शब्द "{rhyme}" आणि "{non_rhyme}" यांपैकी कोणता आहे?',
    '"{word}" शी यमक साधणारा शब्द कोणता आहे, "{rhyme}" किंवा "{non_rhyme}"?',
]

# Pre-compute unique words list (OPTIMIZATION)
unique_words = list(set(ALL_WORDS))

samples = []
target_count = 20000
unique_combinations = set()

# Generate samples using rhyming pairs
for word, rhyme_word in RHYMING_PAIRS.items():
    # Find non-rhyming words
    non_rhyming_words = [w for w in unique_words if w != word and w != rhyme_word]

    if not non_rhyming_words:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        non_rhyme = random.choice(non_rhyming_words)
        query = template.format(word=word, rhyme=rhyme_word, non_rhyme=non_rhyme)
        answer = rhyme_word
        key = (word, rhyme_word, non_rhyme, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

# Also generate reverse (rhyme_word -> word)
for rhyme_word, word in RHYMING_PAIRS.items():
    non_rhyming_words = [w for w in unique_words if w != word and w != rhyme_word]

    if not non_rhyming_words:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        non_rhyme = random.choice(non_rhyming_words)
        query = template.format(word=rhyme_word, rhyme=word, non_rhyme=non_rhyme)
        answer = word
        key = (rhyme_word, word, non_rhyme, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

# Sample with replacement to reach target - ONLY using actual rhymes
if samples:
    while len(samples) < target_count:
        samples.append(random.choice(samples))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S5 Rhyming: Generated {len(samples)} samples")
