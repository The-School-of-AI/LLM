#!/usr/bin/env python3
"""
Generate Statement 2: Letter at Position (अक्षर स्थिति) questions
Target: 25,800 pairs (12.9% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair_hindi  # noqa: E402
from group1_hindi.hindi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE, MEDIUM_WORDS_UNIQUE, HARD_WORDS_UNIQUE, ALL_WORDS_UNIQUE
)
from group1_hindi.generate_s1_spelling import get_hindi_grapheme_clusters  # noqa: E402

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Position names in Hindi
POSITIONS = [
    ("पहला", 1),
    ("दूसरा", 2),
    ("तीसरा", 3),
    ("चौथा", 4),
    ("पांचवां", 5),
    ("छठा", 6),
    ("सातवां", 7),
    ("आठवां", 8),
    ("नौवां", 9),
    ("दसवां", 10),
]

# Question templates
TEMPLATES = [
    '"{word}" का {position} अक्षर क्या है?',
    '"{word}" में {position} अक्षर क्या है?',
    '"{word}" शब्द का {position} अक्षर क्या है?',
    '"{word}" में {position} स्थान पर कौन सा अक्षर है?',
    '"{word}" का {position} अक्षर बताइए?',
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_hindi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue
    
    for template_idx, template in enumerate(TEMPLATES):
        for pos_name, pos_num in POSITIONS:
            if pos_num <= len(clusters):
                query = template.format(word=word, position=pos_name)
                answer = clusters[pos_num - 1]  # 0-indexed
                key = (word, template_idx, pos_num)
                if key not in unique_combinations:
                    unique_combinations[key] = (query, answer)

# Use unique combinations, then sample with replacement to reach target
samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_hindi_grapheme_clusters(word)
    if len(clusters) == 0:
        continue
    template = random.choice(TEMPLATES)
    pos_name, pos_num = random.choice(POSITIONS)
    if pos_num <= len(clusters):
        query = template.format(word=word, position=pos_name)
        answer = clusters[pos_num - 1]
        samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S2 Letter Position: Generated {len(samples)} samples")
