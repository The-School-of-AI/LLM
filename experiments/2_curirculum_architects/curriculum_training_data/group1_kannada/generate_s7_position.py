#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (ಅಕ್ಷರದ ಸ್ಥಿತಿ) questions - Kannada
Target: 17,200 pairs (8.6% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_grammar import get_genitive_suffix  # noqa: E402
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_kannada, int_to_kannada  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Position names in Kannada; numeric answer uses Kannada digits (೧–೧೦)
POSITIONS = [
    ("ಮೊದಲನೇ", "೧"),
    ("ಎರಡನೇ", "೨"),
    ("ಮೂರನೇ", "೩"),
    ("ನಾಲ್ಕನೇ", "೪"),
    ("ಐದನೇ", "೫"),
    ("ಆರನೇ", "೬"),
    ("ಏಳನೇ", "೭"),
    ("ಎಂಟನೇ", "೮"),
    ("ಒಂಬತ್ತನೇ", "೯"),
    ("ಹತ್ತನೇ", "೧೦"),
]

# Single canonical phrasing to avoid duplicate questions (e.g. "ಎಲ್ಲಿ ಇದೆ" vs "ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ").
TEMPLATE = '"{word}" {suffix}ಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ?'

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 21200

# Generate samples: one (word, cluster) -> one question (canonical phrasing only)
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
            pos_name = f"{int_to_kannada(pos_num)}ನೇ"
            pos_str = int_to_kannada(pos_num)

        suffix = get_genitive_suffix(word)
        query = TEMPLATE.format(word=word, char=cluster, suffix=suffix)
        answer = pos_name if random.random() < 0.5 else pos_str
        key = (word, cluster)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
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

    suffix = get_genitive_suffix(word)
    query = TEMPLATE.format(word=word, char=cluster, suffix=suffix)
    answer = pos_name if random.random() < 0.5 else pos_str
    if (query, answer) not in seen_qa:
        seen_qa.add((query, answer))
        samples.append((query, answer))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S7 Position of Letter (Kannada): Generated {len(samples)} samples")
