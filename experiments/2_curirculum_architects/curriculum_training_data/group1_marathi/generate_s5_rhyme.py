#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (यमक) questions
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

# Generate samples using algorithmic rhyming
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
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


# Group words by rhyme suffix (last 2 chars approx rhyme)
# In Marathi, words ending with same matra/char often rhyme
rhyme_groups = defaultdict(list)

# Use suffix length of 2 for better rhymes, fallback to 1 if needed
for word in unique_words:
    if len(word) >= 2:
        suffix = word[-2:]
        rhyme_groups[suffix].append(word)

# Flatten into pairs
generated_pairs = []
for suffix, words in rhyme_groups.items():
    if len(words) >= 2:
        # Generate all pairs in this group
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                generated_pairs.append((words[i], words[j]))

# Shuffle pairs to get random selection
random.shuffle(generated_pairs)

print(f"Generated {len(generated_pairs)} rhyming pairs algorithmically")

for word, rhyme_word in generated_pairs:
    if len(samples) >= target_count:
        break

    # Find non-rhyming words (different suffix)
    # Optimization: Pick random word and check suffix
    non_rhyme = None
    for _ in range(10):
        candidate = random.choice(unique_words)
        if len(candidate) >= 2 and candidate[-2:] != word[-2:]:
            non_rhyme = candidate
            break

    if not non_rhyme:
        continue

    # Forward direction
    template = random.choice(TEMPLATES)
    query = template.format(word=word, rhyme=rhyme_word, non_rhyme=non_rhyme)
    answer = rhyme_word
    key = (word, rhyme_word, non_rhyme)
    if key not in unique_combinations:
        unique_combinations.add(key)
        samples.append((query, answer))

    # Reverse direction
    if len(samples) < target_count:
        template = random.choice(TEMPLATES)
        query = template.format(word=rhyme_word, rhyme=word, non_rhyme=non_rhyme)
        answer = word
        key = (rhyme_word, word, non_rhyme)
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
