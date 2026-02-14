#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (ಪ್ರಾಸ) questions - Kannada
Target: 20,000 pairs (10% of 200,000). ಪ್ರಾಸ = rhyme (correct Kannada term).
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_vocabulary import ALL_WORDS_UNIQUE, RHYMING_PAIRS  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates (multiple choice). Use {option1}/{option2} so answer position is random. Varied phrasings.
TEMPLATES = [
    '"{word}" ಪದವಿಗೆ ಪ್ರಾಸ ಪದ ಯಾವುದು, "{option1}" ಅಥವಾ "{option2}"?',
    '"{word}" ಪದವಿಗೆ ಯಾವ ಪದ ಪ್ರಾಸ ಮಾಡುತ್ತದೆ, "{option1}" ಅಥವಾ "{option2}"?',
    '"{word}" ಜೊತೆ ಪ್ರಾಸ ಪದ ಯಾವುದು, "{option1}" ಅಥವಾ "{option2}"?',
    '"{word}" ನೊಂದಿಗೆ ಪ್ರಾಸ ಮಾಡುವ ಪದ "{option1}" ಮತ್ತು "{option2}" ನಲ್ಲಿ ಯಾವುದು?',
    '"{word}" ಪದವಿಗೆ ಪ್ರಾಸ ಆಗುವ ಪದ ಯಾವುದು, "{option1}" ಅಥವಾ "{option2}"?',
    '"{option1}" ಮತ್ತು "{option2}" ನಲ್ಲಿ "{word}" ಪದವಿಗೆ ಪ್ರಾಸವಾಗುವುದು ಯಾವುದು?',
    'ಯಾವ ಪದ "{word}" ಜೊತೆ ಪ್ರಾಸ ಹೊಂದುತ್ತದೆ, "{option1}" ಅಥವಾ "{option2}"?',
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
    # Only use RHYMING_PAIRS - both words are real vocabulary words
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
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S5 Rhyming (Kannada): Generated {len(samples)} samples")
