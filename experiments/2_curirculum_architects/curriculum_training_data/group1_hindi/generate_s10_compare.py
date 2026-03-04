#!/usr/bin/env python3
"""
Generate Statement 10: Word Comparison (शब्द तुलना) questions
Target: 11,000 pairs (5.5% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.generate_s1_spelling import get_hindi_grapheme_clusters  # noqa: E402
from group1_hindi.hindi_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates
TEMPLATES_LONGER = [
    'कौन सा शब्द लंबा है, "{word1}" या "{word2}"?',
    '"{word1}" और "{word2}" में से कौन सा शब्द लंबा है?',
    'कौन सा शब्द अधिक लंबा है, "{word1}" या "{word2}"?',
    '"{word1}" और "{word2}" में से लंबा शब्द कौन सा है?',
]

TEMPLATES_SHORTER = [
    'कौन सा शब्द छोटा है, "{word1}" या "{word2}"?',
    '"{word1}" और "{word2}" में से कौन सा शब्द छोटा है?',
    'कौन सा शब्द अधिक छोटा है, "{word1}" या "{word2}"?',
    '"{word1}" और "{word2}" में से छोटा शब्द कौन सा है?',
]

# Pre-compute word lengths (OPTIMIZATION - cache expensive operation)
unique_words = list(set(ALL_WORDS))

# Exclude adjectives that are used in comparison queries to avoid confusion
EXCLUDED_COMPARISON_WORDS = {
    "बड़ा",
    "छोटा",
    "लंबा",
    "नाटा",
    "ऊंचा",
    "नीचा",
    "मोटा",
    "पतला",
    "भारी",
    "हल्का",
    "तेज",
    "धीमा",
}

# Filter out excluded words
unique_words = [w for w in unique_words if w not in EXCLUDED_COMPARISON_WORDS]

word_lengths = {}
for word in unique_words:
    clusters = get_hindi_grapheme_clusters(word)
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

# Only use unique combinations - NO sampling with replacement
unique_count = len(samples)

if unique_count < target_count:
    print(f"Warning: Only {unique_count} unique combinations (target: {target_count})")
else:
    samples = samples[:target_count]

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(
    f"S10 Word Comparison: Generated {len(samples)} unique samples (target: {target_count})"
)
