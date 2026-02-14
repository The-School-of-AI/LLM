#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (ప్రాస) questions - Telugu
Target: 20,000 pairs (10% of 200,000). ప్రాస = rhyme.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_telugu.telugu_vocabulary import ALL_WORDS_UNIQUE, RHYMING_PAIRS  # noqa: E402
from group1_telugu.prompt_utils_telugu import format_qa_pair_telugu  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates (multiple choice)
TEMPLATES = [
    '"{word}" పదానికి ప్రాస పదం ఏది, "{option1}" లేదా "{option2}"?',
    '"{word}" పదానికి ఏ పదం ప్రాసబద్ధమవుతుంది, "{option1}" లేదా "{option2}"?',
    '"{word}" తో ప్రాస పదం ఏమిటి, "{option1}" లేదా "{option2}"?',
    '"{option1}" మరియు "{option2}" లో "{word}" పదానికి ప్రాసమయ్యేది ఏది?',
    '"{word}" పదానికి ప్రాస అయ్యే పదం ఏది, "{option1}" లేదా "{option2}"?',
    'ఏ పదం "{word}" తో ప్రాస అవుతుంది, "{option1}" లేదా "{option2}"?',
    '"{option1}" మరియు "{option2}" లలో "{word}" కు ప్రాసబద్ధమైనది ఏది?',
]

unique_words = list(set(ALL_WORDS))

samples = []
target_count = 20000
unique_combinations = set()

# Generate samples using rhyming pairs
for word, rhyme_word in RHYMING_PAIRS.items():
    non_rhyming_words = [w for w in unique_words if w != word and w != rhyme_word]
    if not non_rhyming_words:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        non_rhyme = random.choice(non_rhyming_words)
        option1, option2 = random.sample([rhyme_word, non_rhyme], 2)
        query = template.format(word=word, option1=option1, option2=option2)
        answer = rhyme_word
        key = (word, rhyme_word, non_rhyme, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

# Reverse (rhyme_word -> word)
for rhyme_word, word in RHYMING_PAIRS.items():
    non_rhyming_words = [w for w in unique_words if w != word and w != rhyme_word]
    if not non_rhyming_words:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        non_rhyme = random.choice(non_rhyming_words)
        option1, option2 = random.sample([word, non_rhyme], 2)
        query = template.format(word=rhyme_word, option1=option1, option2=option2)
        answer = word
        key = (rhyme_word, word, non_rhyme, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

while len(samples) < target_count:
    if not RHYMING_PAIRS:
        break
    word = random.choice(list(RHYMING_PAIRS.keys()))
    rhyme_word = RHYMING_PAIRS[word]

    non_rhyming_words = [w for w in unique_words if w != word and w != rhyme_word]
    if not non_rhyming_words:
        continue

    template = random.choice(TEMPLATES)
    non_rhyme = random.choice(non_rhyming_words)
    option1, option2 = random.sample([rhyme_word, non_rhyme], 2)
    query = template.format(word=word, option1=option1, option2=option2)
    answer = rhyme_word
    samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_telugu(query, answer) + "\n")

print(f"S5 Rhyming (Telugu): Generated {len(samples)} samples")
