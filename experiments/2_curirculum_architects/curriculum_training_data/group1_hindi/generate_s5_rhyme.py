#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (तुकबंदी) questions
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.hindi_vocabulary import ALL_WORDS_UNIQUE, RHYMING_PAIRS  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates (multiple choice format)
TEMPLATES = [
    '"{word}" से तुकबंदी करने वाला शब्द कौन सा है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से कौन सा शब्द तुकबंदी करता है, "{rhyme}" या "{non_rhyme}"?',
    'कौन सा शब्द "{word}" से तुकबंदी करता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" के साथ तुकबंदी करने वाला शब्द "{rhyme}" और "{non_rhyme}" में से कौन सा है?',
    '"{word}" से राइम करने वाला शब्द कौन सा है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से मेल खाने वाला शब्द बताइए, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" के साथ कौन सा शब्द मिलता है, "{rhyme}" या "{non_rhyme}"?',
    'बताइए "{word}" से कौन सा शब्द तुकबंदी करता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" शब्द से मेल खाता है कौन सा, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" के साथ राइम करता है कौन, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से तुकबंदी कौन करता है, "{rhyme}" या "{non_rhyme}"?',
    'कौन सा शब्द "{word}" के साथ राइम करता है, "{rhyme}" या "{non_rhyme}"?',
    # Additional 10 templates
    '"{word}" के साथ कौन मिलता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से तुक मिलाता है कौन, "{rhyme}" या "{non_rhyme}"?',
    'बताओ "{word}" से कौन तुकबंदी करता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" के साथ तुक मिलाने वाला कौन है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" शब्द से कौन तुक मिलता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से मिलने वाला शब्द कौन है, "{rhyme}" या "{non_rhyme}"?',
    'कौन "{word}" के साथ मेल खाता है, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" के साथ तुकबंदी होती है किस से, "{rhyme}" या "{non_rhyme}"?',
    '"{word}" से कौन राइम बनाता है, "{rhyme}" या "{non_rhyme}"?',
    'बताइए "{word}" के साथ कौन मिलता है, "{rhyme}" या "{non_rhyme}"?',
]

# Pre-compute unique words list (OPTIMIZATION)
unique_words = list(set(ALL_WORDS))

samples = []
target_count = 20000
unique_combinations = set()

# Generate samples using rhyming pairs
for word, rhyme_word in RHYMING_PAIRS:
    # Find non-rhyming words (OPTIMIZED - use pre-computed list)
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
for rhyme_word, word in RHYMING_PAIRS:
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

# Only use unique combinations - NO sampling with replacement
unique_count = len(samples)

if unique_count < target_count:
    print(f"Warning: Only {unique_count} unique combinations (target: {target_count})")
else:
    samples = samples[:target_count]

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S5 Rhyming: Generated {len(samples)} unique samples (target: {target_count})")
