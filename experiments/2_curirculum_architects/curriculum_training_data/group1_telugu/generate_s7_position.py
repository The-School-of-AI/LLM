#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (అక్షరం స్థానం) questions - Telugu
Target: 18,000 pairs (9% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import get_telugu_grapheme_clusters  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Position names in Telugu
POSITIONS = [
    ("మొదటి", "1"),
    ("రెండవ", "2"),
    ("మూడవ", "3"),
    ("నాల్గవ", "4"),
    ("ఐదవ", "5"),
    ("ఆరవ", "6"),
    ("ఏడవ", "7"),
    ("ఎనిమిదవ", "8"),
    ("తొమ్మిదవ", "9"),
    ("పదవ", "10"),
]

# Templates — Telugu uses invariant లో (no genitive suffix needed)
TEMPLATES = [
    '"{word}" లో "{char}" అక్షరం ఏ స్థానంలో ఉంది?',
    '"{word}" లో "{char}" అక్షరం ఎక్కడ ఉంది?',
    '"{word}" పదంలో "{char}" అక్షరం ఏ స్థానంలో ఉంది?',
    '"{word}" లో "{char}" ఏ స్థానంలో వస్తుంది?',
    '"{char}" అక్షరం "{word}" పదంలో ఏ స్థానంలో ఉంది?',
    '"{word}" పదంలో "{char}" ఎంతవ స్థానంలో వస్తుంది?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 18000

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_telugu_grapheme_clusters(word)
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
            pos_name = f"{pos_num}వ"
            pos_str = str(pos_num)

        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word, char=cluster)
            answer = pos_name if random.random() < 0.5 else pos_str
            key = (word, cluster, template_idx)
            if key not in unique_combinations:
                unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_telugu_grapheme_clusters(word)
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
        pos_name = f"{pos_num}వ"
        pos_str = str(pos_num)

    template = random.choice(TEMPLATES)
    query = template.format(word=word, char=cluster)
    answer = pos_name if random.random() < 0.5 else pos_str
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S7 Position of Letter (Telugu): Generated {len(samples)} samples")
