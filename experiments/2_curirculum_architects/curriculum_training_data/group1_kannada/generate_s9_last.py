#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (ಕೊನೆಯ ಅಕ್ಷರ) questions - Kannada
Target: 17,200 pairs (8.6% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import get_kannada_grapheme_clusters  # noqa: E402
from group1_kannada.kannada_grammar import get_genitive_suffix  # noqa: E402
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Question templates: {suffix} = get_genitive_suffix(word). Varied phrasings.
TEMPLATES = [
    '"{word}" {suffix} ಕೊನೆಯ ಅಕ್ಷರ ಏನು?',
    '"{word}" ಯಾವ ಅಕ್ಷರದಿಂದ ಕೊನೆಗೊಳ್ಳುತ್ತದೆ?',
    '"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರ ಏನು?',
    '"{word}" {suffix} ಕಡೆಯ ಅಕ್ಷರ ಏನು?',
    '"{word}" ಯಾವ ಅಕ್ಷರದಲ್ಲಿ ಮುಗಿಯುತ್ತದೆ?',
    '"{word}" {suffix} ಕೊನೆಯಲ್ಲಿ ಯಾವ ಅಕ್ಷರವಿದೆ?',
    '"{word}" ಪದದಲ್ಲಿ ಕೊನೆಯಲ್ಲಿ ಬರುವ ಅಕ್ಷರ ಯಾವುದು?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 19200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    last_cluster = clusters[-1]
    suffix = get_genitive_suffix(word)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word, suffix=suffix)
        answer = last_cluster
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    last_cluster = clusters[-1]
    template = random.choice(TEMPLATES)
    suffix = get_genitive_suffix(word)
    query = template.format(word=word, suffix=suffix)
    answer = last_cluster
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S9 Last Letter (Kannada): Generated {len(samples)} samples")
