#!/usr/bin/env python3
"""
Generate Statement 6: Classification (వర్గీకరణ) questions - Telugu
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.telugu_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Question templates (వ్యక్తి=person, జంతువు=animal, వస్తువు=object)
TEMPLATES = [
    '"{word}" వ్యక్తి, జంతువు లేదా వస్తువు?',
    '"{word}" ఏమిటి, వ్యక్తి, జంతువు లేదా వస్తువు?',
    '"{word}" పదం ఏ వర్గంలోకి వస్తుంది, వ్యక్తి, జంతువు లేదా వస్తువు?',
    '"{word}" ను వ్యక్తి, జంతువు లేదా వస్తువుగా వర్గీకరించండి?',
    '"{word}" అనేది వ్యక్తి, జంతువు లేదా వస్తువు?',
    '"{word}" ఏ రకం, వ్యక్తి, జంతువు లేదా వస్తువు?',
    '"{word}" పదం యొక్క వర్గం ఏమిటి?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    return "వస్తువు"


samples = []
target_count = 20000
all_words = []
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words.extend(word_list)

all_words = all_words * 20
unique_combinations = {}

for word in set(all_words):
    category = classify_word(word)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    category = classify_word(word)
    template = random.choice(TEMPLATES)
    query = template.format(word=word)
    answer = category
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S6 Classification (Telugu): Generated {len(samples)} samples")
