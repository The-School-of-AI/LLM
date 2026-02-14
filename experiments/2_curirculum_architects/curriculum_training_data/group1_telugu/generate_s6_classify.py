#!/usr/bin/env python3
"""
Generate Statement 6: Classification (వర్గీకరణ) questions - Telugu
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.telugu_vocabulary import CLASSIFICATION_CATEGORIES, ALL_WORDS_UNIQUE  # noqa: E402
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
# Use ALL_WORDS_UNIQUE for maximum word coverage
# Words not in CLASSIFICATION_CATEGORIES default to "వస్తువు"
all_words = list(ALL_WORDS_UNIQUE)
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

# Track seen lines for dedup
seen_lines = set()
for q, a in samples:
    seen_lines.add((q, a))

max_attempts = target_count * 10
attempts = 0
while len(samples) < target_count and attempts < max_attempts:
    attempts += 1
    word = random.choice(list(set(all_words)))
    category = classify_word(word)
    template = random.choice(TEMPLATES)
    q = template.format(word=word)
    a = category
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

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S6 Classification (Telugu): Generated {len(samples)} samples")
