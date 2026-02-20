#!/usr/bin/env python3
"""
Generate Statement 7: Position of Letter (అక్షరం స్థానం) questions - Telugu
Target: 18,000 pairs (9% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.generate_s1_spelling import (  # noqa: E402
    get_telugu_grapheme_clusters,
)
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402
from group1_telugu.telugu_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)

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
    '"{word}" పదంలో "{char}" ఎన్నవ అక్షరం?',
    '"{word}" పదంలో "{char}" అక్షరం ఏ స్థానంలో ఉంది?',
    '"{word}" లో "{char}" ఏ స్థానంలో వస్తుంది?',
    '"{char}" అక్షరం "{word}" పదంలో ఏ స్థానంలో ఉంది?',
    '"{word}" పదంలో "{char}" అక్షరం ఎన్నవ స్థానంలో ఉంది?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS + list(ALL_WORDS_UNIQUE)
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

# Track seen lines for dedup
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

max_attempts = target_count * 10
attempts = 0
while len(samples) < target_count and attempts < max_attempts:
    attempts += 1
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
    q = template.format(word=word, char=cluster)
    a = pos_name if random.random() < 0.5 else pos_str
    if (q, a) not in seen_lines:
        seen_lines.add((q, a))
        samples.append((q, a))

# Final dedup
unique_samples = []
final_seen = set()
for q, a in samples:
    if (q, a) not in final_seen:
        final_seen.add((q, a))
        unique_samples.append((q, a))
samples = unique_samples

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S7 Position of Letter (Telugu): Generated {len(samples)} samples")
