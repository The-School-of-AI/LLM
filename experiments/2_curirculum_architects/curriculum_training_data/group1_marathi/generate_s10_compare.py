#!/usr/bin/env python3
"""
Generate Statement 10: Word Comparison (शब्द तुलना) questions
Target: 11,000 pairs (5.5% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
from prompt_utils import (  # noqa: E402
    format_qa_pair_marathi,
    get_marathi_grapheme_clusters,
)

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates
TEMPLATES_LONGER = [
    'कोणता शब्द लांब आहे, "{word1}" किंवा "{word2}"?',
    '"{word1}" आणि "{word2}" यांपैकी कोणता शब्द लांब आहे?',
    'कोणता शब्द अधिक लांब आहे, "{word1}" किंवा "{word2}"?',
    '"{word1}" आणि "{word2}" यांपैकी लांब शब्द कोणता आहे?',
]

TEMPLATES_SHORTER = [
    'कोणता शब्द लहान आहे, "{word1}" किंवा "{word2}"?',
    '"{word1}" आणि "{word2}" यांपैकी कोणता शब्द लहान आहे?',
    'कोणता शब्द अधिक लहान आहे, "{word1}" किंवा "{word2}"?',
    '"{word1}" आणि "{word2}" यांपैकी लहान शब्द कोणता आहे?',
]

# Pre-compute word lengths (OPTIMIZATION - cache expensive operation)
unique_words = list(set(ALL_WORDS))
word_lengths = {}
for word in unique_words:
    clusters = get_marathi_grapheme_clusters(word)
    word_lengths[word] = len(clusters)


def get_word_length(word: str) -> int:
    """Get the length of a word in grapheme clusters (cached)"""
    return word_lengths.get(word, 0)


samples = []
target_count = 11000
unique_combinations = set()

# Generate samples efficiently - limit iterations
word_list = unique_words
max_pairs_to_generate = min(
    target_count * 2, len(word_list) * (len(word_list) - 1) // 2
)
pairs_generated = 0

for i, word1 in enumerate(word_list):
    if pairs_generated >= max_pairs_to_generate:
        break
    for j, word2 in enumerate(word_list):
        if i >= j:  # Avoid duplicates
            continue
        if pairs_generated >= max_pairs_to_generate:
            break

        len1 = get_word_length(word1)
        len2 = get_word_length(word2)

        # Skip equal-length pairs - can't compare when lengths are equal
        if len1 == len2:
            continue

        if len1 > len2:
            longer_word = word1
            shorter_word = word2
        else:
            longer_word = word2
            shorter_word = word1

        # Generate longer questions
        for template_idx, template in enumerate(TEMPLATES_LONGER):
            query = template.format(word1=word1, word2=word2)
            answer = longer_word
            key = (word1, word2, template_idx, "longer")
            if key not in unique_combinations:
                unique_combinations.add(key)
                samples.append((query, answer))

        # Generate shorter questions
        for template_idx, template in enumerate(TEMPLATES_SHORTER):
            query = template.format(word1=word1, word2=word2)
            answer = shorter_word
            key = (word1, word2, template_idx, "shorter")
            if key not in unique_combinations:
                unique_combinations.add(key)
                samples.append((query, answer))

        pairs_generated += 1

# Sample with replacement to reach target
while len(samples) < target_count:
    word1 = random.choice(word_list)
    word2 = random.choice([w for w in word_list if w != word1])

    len1 = get_word_length(word1)
    len2 = get_word_length(word2)

    # Skip equal-length pairs - can't compare when lengths are equal
    if len1 == len2:
        continue

    if len1 > len2:
        longer_word = word1
        shorter_word = word2
    else:
        longer_word = word2
        shorter_word = word1

    if random.random() < 0.5:
        template = random.choice(TEMPLATES_LONGER)
        answer = longer_word
    else:
        template = random.choice(TEMPLATES_SHORTER)
        answer = shorter_word

    query = template.format(word1=word1, word2=word2)
    samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S10 Word Comparison: Generated {len(samples)} samples")
