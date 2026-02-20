#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (అక్షర గణన) questions - Telugu
Target: 26,000 pairs (13% of 200,000)
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

# Telugu vowels and consonants
VOWELS = set(chr(c) for c in range(0x0C05, 0x0C15))  # అ through ఔ
CONSONANTS = set(chr(c) for c in range(0x0C15, 0x0C3A))  # క through హ

TEMPLATES = [
    ('"{word}" పదంలో ఎన్ని అక్షరాలు ఉన్నాయి?', "count"),
    ('"{word}" పదంలోని మొత్తం అక్షరాల సంఖ్య ఎంత?', "count"),
    ('"{word}" పదంలో ఎన్ని అక్షరాలు కనిపిస్తాయి?', "count"),
    ('"{word}" పదంలో ఎన్ని వర్ణాలు ఉన్నాయి?', "count"),
    ('"{word}" పదం రెండు అక్షరాల పదమా?', "two_letter_yes_no"),
    ('"{word}" పదంలోని అక్షరాలను లెక్కించండి?', "count"),
    ('"{word}" పదంలో ఉన్న అక్షరాలు ఎన్ని?', "count"),
    ('"{word}" పదంలో ఎన్ని స్వరాలు ఉన్నాయి?', "vowel_count"),
    ('"{word}" పదంలో సంయుక్తాక్షరాలు కలిపి ఎన్ని అక్షరాలు ఉన్నాయి?', "count"),
    ('"{word}" పదంలోని అక్షరాల లెక్క చెప్పండి?', "count"),
    ('"{word}" పదంలోని మొత్తం అక్షరాల సంఖ్య ఎంత?', "count"),
    ('"{word}" పదంలో ఎన్ని అక్షరాలు ఉన్నాయో లెక్కించండి?', "count"),
    ('"{word}" పదం మూడు అక్షరాల పదమా?', "three_letter_yes_no"),
    ('"{word}" పదంలోని అక్షరాల మొత్తం ఎంత?', "count"),
    ('"{word}" పదంలోని అక్షరాల సంఖ్య తెలియజేయండి?', "count"),
    ('"{word}" పదంలో కేవలం రెండు అక్షరాలు ఉన్నాయా?', "two_letter_yes_no"),
    ('"{word}" పదంలోని అక్షరాలను లెక్కించండి?', "count"),
    ('"{word}" పదంలో ఎన్ని వ్యంజనాలు ఉన్నాయి?', "consonant_count"),
    ('"{word}" పదంలోని అక్షరాల సంఖ్య ఎంత?', "count"),
    ('"{word}" పదంలో సంయుక్తాక్షరాలతో కలిపి ఎన్ని అక్షరాలు ఉన్నాయి?', "count"),
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS + list(ALL_WORDS_UNIQUE)
samples = []
target_count = 26000
unique_combinations = {}

for word in set(all_words):
    clusters = get_telugu_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, (template, answer_type) in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = ""
        if answer_type == "count":
            answer = str(cluster_count)
        elif answer_type == "two_letter_yes_no":
            answer = "అవును" if cluster_count == 2 else "కాదు"
        elif answer_type == "three_letter_yes_no":
            answer = "అవును" if cluster_count == 3 else "కాదు"
        elif answer_type == "vowel_count":
            vowels_in_word = [c for c in clusters if c[0] in VOWELS]
            answer = str(len(vowels_in_word))
        elif answer_type == "consonant_count":
            consonants_in_word = [c for c in clusters if c[0] in CONSONANTS]
            answer = str(len(consonants_in_word))
        else:
            continue

        key = (word, template_idx)
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
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue
    template, answer_type = random.choice(TEMPLATES)
    q = template.format(word=word)
    a = ""
    if answer_type == "count":
        a = str(cluster_count)
    elif answer_type == "two_letter_yes_no":
        a = "అవును" if cluster_count == 2 else "కాదు"
    elif answer_type == "three_letter_yes_no":
        a = "అవును" if cluster_count == 3 else "కాదు"
    elif answer_type == "vowel_count":
        vowels_in_word = [c for c in clusters if c[0] in VOWELS]
        a = str(len(vowels_in_word))
    elif answer_type == "consonant_count":
        consonants_in_word = [c for c in clusters if c[0] in CONSONANTS]
        a = str(len(consonants_in_word))
    else:
        continue
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

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S4 Letter Count (Telugu): Generated {len(samples)} samples")
