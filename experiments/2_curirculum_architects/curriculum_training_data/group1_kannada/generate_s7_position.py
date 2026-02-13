#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (ಅಕ್ಷರದ ಸ್ಥಿತಿ) questions - Kannada
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

# Position names in Kannada
POSITIONS = [
    ("ಮೊದಲನೇ", "1"),
    ("ಎರಡನೇ", "2"),
    ("ಮೂರನೇ", "3"),
    ("ನಾಲ್ಕನೇ", "4"),
    ("ಐದನೇ", "5"),
    ("ಆರನೇ", "6"),
    ("ಏಳನೇ", "7"),
    ("ಎಂಟನೇ", "8"),
    ("ಒಂಬತ್ತನೇ", "9"),
    ("ಹತ್ತನೇ", "10"),
]

# Question templates: {suffix} = get_genitive_suffix(word). Varied phrasings.
TEMPLATES = [
    '"{word}" {suffix}ಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ?',
    '"{word}" {suffix}ಲ್ಲಿ "{char}" ಅಕ್ಷರ ಎಲ್ಲಿ ಇದೆ?',
    '"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ?',
    '"{word}" {suffix}ಲ್ಲಿ "{char}" ಯಾವ ಸ್ಥಾನದಲ್ಲಿ ಸಿಗುತ್ತದೆ?',
    '"{char}" ಅಕ್ಷರ "{word}" ಪದದಲ್ಲಿ ಎಂಥ ಸ್ಥಾನದಲ್ಲಿದೆ?',
    '"{word}" ಪದದಲ್ಲಿ "{char}" ಎಂಥ ಸ್ಥಾನದಲ್ಲಿ ಬರುತ್ತದೆ?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 17200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    for cluster in clusters:
        cluster_positions = [i + 1 for i, c in enumerate(clusters) if c == cluster]
        if not cluster_positions:
            continue

        pos_num = cluster_positions[0]
        if pos_num <= len(POSITIONS):
            pos_name, pos_str = POSITIONS[pos_num - 1]
        else:
            pos_name = f"{pos_num}ನೇ"
            pos_str = str(pos_num)

        suffix = get_genitive_suffix(word)
        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word, char=cluster, suffix=suffix)
            answer = pos_name if random.random() < 0.5 else pos_str
            key = (word, cluster, template_idx)
            if key not in unique_combinations:
                unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    cluster = random.choice(clusters)
    cluster_positions = [i + 1 for i, c in enumerate(clusters) if c == cluster]
    if not cluster_positions:
        continue

    pos_num = cluster_positions[0]
    if pos_num <= len(POSITIONS):
        pos_name, pos_str = POSITIONS[pos_num - 1]
    else:
        pos_name = f"{pos_num}ನೇ"
        pos_str = str(pos_num)

    template = random.choice(TEMPLATES)
    suffix = get_genitive_suffix(word)
    query = template.format(word=word, char=cluster, suffix=suffix)
    answer = pos_name if random.random() < 0.5 else pos_str
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S7 Position of Letter (Kannada): Generated {len(samples)} samples")
