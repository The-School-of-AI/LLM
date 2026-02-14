#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (ಅಕ್ಷರ ಗಣನೆ) questions - Kannada
Target: 25,800 pairs (12.9% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import get_kannada_grapheme_clusters  # noqa: E402
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

VOWELS = set([chr(c) for c in range(0x0C85, 0x0C91) if c not in [0x0C8C, 0x0C8E]])
CONSONANTS = set([chr(c) for c in range(0x0C95, 0x0CB9) if chr(c) not in ['ಱ', 'ೞ']])

# User-specified Letter Count templates. answer_type: "count" (number) or "two_letter_yes_no" or "consonant_count" or "vowel_count".
TEMPLATES = [
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
    ('"{word}" ಪದದ ಒಟ್ಟು ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳನ್ನು ಕಾಣಬಹುದು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ವರ್ಣಗಳಿವೆ?', "count"),
    ('"{word}" ಪದವು ಎರಡು ಅಕ್ಷರದ ಪದವೇ?', "two_letter_yes_no"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಎಣಿಸಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳೆಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವರಗಳಿವೆ?', "vowel_count"),  # Answer: vowel count
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರಗಳನ್ನು ಸೇರಿಸಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಲೆಕ್ಕ ಕೊಡಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒಟ್ಟು ಮಾತ್ರೆಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "count"), # Matre count ~ grapheme count
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ ಎಂದು ಎಣಿಸಿ?', "count"),
    ('"{word}" ಪದವು ಮೂರು ಅಕ್ಷರದ ಪದವೇ?', "three_letter_yes_no"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳ ಮೊತ್ತ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ತಿಳಿಸಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಕೇವಲ ಎರಡು ಅಕ್ಷರಗಳಿವೆಯೇ?', "two_letter_yes_no"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಎಣಿಕೆ ಮಾಡಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ವ್ಯಂಜನಗಳು ಇವೆ?', "consonant_count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರವನ್ನು ಸೇರಿಸಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 25800
unique_combinations = {}

for word in set(all_words):
    clusters = get_kannada_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, (template, answer_type) in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = ""
        if answer_type == "count":
            answer = f"{cluster_count} ಅಕ್ಷರಗಳು"
            if "ಮಾತ್ರೆಗಳ ಸಂಖ್ಯೆ" in template: # Specific formatting for matra count
                # For now, matra count is same as grapheme count. Refine if specific matra rules are given.
                answer = f"{cluster_count} ಮಾತ್ರೆಗಳು"
        elif answer_type == "two_letter_yes_no":
            answer = "ಹೌದು" if cluster_count == 2 else "ಇಲ್ಲ"
        elif answer_type == "three_letter_yes_no":
            answer = "ಹೌದು" if cluster_count == 3 else "ಇಲ್ಲ"
        elif answer_type == "vowel_count":
            vowels_in_word = [c for c in clusters if c[0] in VOWELS] # Check first char of cluster
            answer = f"{len(vowels_in_word)} ಸ್ವರಗಳು"
        elif answer_type == "consonant_count":
            consonants_in_word = [c for c in clusters if c[0] in CONSONANTS] # Check first char of cluster
            answer = f"{len(consonants_in_word)} ವ್ಯಂಜನಗಳು"
        else:
            continue # Should not happen

        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    word = random.choice(list(set(all_words)))
    clusters = get_kannada_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue
    template, answer_type = random.choice(TEMPLATES)
    query = template.format(word=word)
    answer = ""
    if answer_type == "count":
        answer = f"{cluster_count} ಅಕ್ಷರಗಳು"
        if "ಮಾತ್ರೆಗಳ ಸಂಖ್ಯೆ" in template:
            answer = f"{cluster_count} ಮಾತ್ರೆಗಳು"
    elif answer_type == "two_letter_yes_no":
        answer = "ಹೌದು" if cluster_count == 2 else "ಇಲ್ಲ"
    elif answer_type == "three_letter_yes_no":
        answer = "ಹೌದು" if cluster_count == 3 else "ಇಲ್ಲ"
    elif answer_type == "vowel_count":
        vowels_in_word = [c for c in clusters if c[0] in VOWELS]
        answer = f"{len(vowels_in_word)} ಸ್ವರಗಳು"
    elif answer_type == "consonant_count":
        consonants_in_word = [c for c in clusters if c[0] in CONSONANTS]
        answer = f"{len(consonants_in_word)} ವ್ಯಂಜನಗಳು"
    else:
        continue
    samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S4 Letter Count (Kannada): Generated {len(samples)} samples")
