#!/usr/bin/env python3
"""
Generate Statement 8: Number Spelling (ಸಂಖ್ಯೆ ಕಾಗುಣಿತ) questions - Kannada
Target: 10,000 pairs (5% of 200,000)
ಕಾಗುಣಿತ = spelling; ಅಕ್ಷರ = letter.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
    get_varnavichcheda_str,
)
from group1_kannada.kannada_grammar import get_genitive_suffix  # noqa: E402
from group1_kannada.kannada_grammar import get_kannada_aksharas  # noqa: E402
from group1_kannada.kannada_vocabulary import NUMBERS  # noqa: E402
from prompt_utils import format_qa_pair_kannada, int_to_kannada  # noqa: E402

HALANT = "\u0ccd"  # Virama (ottakshara check)

# Question templates - number to name (ಹೆಸರು). Use {num} for digit, {num_k} for Kannada digit.
TEMPLATES_NAME = [
    ("{num} {suffix} ಹೆಸರು ಏನು?", False),
    ("{num} {suffix} ಕನ್ನಡ ಹೆಸರು ಏನು?", False),
    ("{num} ಅನ್ನು ಕನ್ನಡದಲ್ಲಿ ಏನು ಎನ್ನುತ್ತಾರೆ?", False),
    ("{num} ಸಂಖ್ಯೆಯ ಹೆಸರು ಏನು?", False),
    ("{num} ಅಂಕೆಯ ಹೆಸರು ಏನು?", False),
    ("{num_k} {suffix} ಅಕ್ಷರ ರೂಪವೇನು?", True),  # 6: Kannada digit in question
    ("{num_k} ಅಂಕೆಯನ್ನು ಪದಗಳಲ್ಲಿ ಹೇಗೆ ಬರೆಯುವುದು?", True),  # 7
    ("{num_k} ಸಂಖ್ಯೆಗೆ ಸಮನಾದ ಕನ್ನಡ ಪದ ಯಾವುದು?", True),  # 8
    ("{num_k} {suffix} ಪೂರ್ಣ ಹೆಸರು ತಿಳಿಸಿ?", True),  # 9
]

TEMPLATES_SPELLING = [
    ('"{word}" ಪದದ ಕಾಗುಣಿತ ಏನು?', "spelling"),
    ('"{word}" ಪದವನ್ನು ಹೇಗೆ ಬರೆಯುವುದು?', "spelling"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳು ಯಾವುವು?', "spelling"),
    ('"{word}" ಪದದ ಕಾಗುಣಿತ ತಿಳಿಸಿ?', "spelling"),
    ('ಈ ಸಂಖ್ಯೆಯ ಪದ "{word}" ಅಕ್ಷರ ಅಕ್ಷರವಾಗಿ ಬರೆಯಿರಿ?', "spelling"),
    ('"{word}" ಪದವನ್ನು ವರ್ಣವಿಚ್ಛೇದ ಮಾಡಿ ಬರೆಯಿರಿ?', "varnaviccheda"),  # 10
    ('"{word}" ಸಂಖ್ಯಾವಾಚಕ ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳೆಷ್ಟು?', "akshara_count"),  # 11
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಒತ್ತಕ್ಷರ ಬಳಕೆಯಾಗಿದೆ?', "ottakshara_in_number"),  # 12
]

samples = []
target_count = 12000
unique_combinations = {}

# Number to name (suffix for numeral is ರ)
for num in range(1, 101):
    if num <= len(NUMBERS):
        word = NUMBERS[num - 1]
    else:
        continue
    suffix = get_genitive_suffix(str(num))
    num_k = int_to_kannada(num)
    for template_idx, (template, use_k) in enumerate(TEMPLATES_NAME):
        fmt = {"num": num, "num_k": num_k, "suffix": suffix}
        query = template.format(**fmt)
        answer = word
        key = (num, template_idx, "name")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Name to spelling / varnaviccheda / akshara_count / ottakshara
for word in NUMBERS:
    aksharas = get_kannada_aksharas(word)
    clusters = get_kannada_grapheme_clusters(word)
    if len(aksharas) == 0:
        continue

    for template_idx, (template, answer_type) in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        if answer_type == "spelling":
            answer = ", ".join(aksharas)
        elif answer_type == "varnaviccheda":
            answer = get_varnavichcheda_str(word)
        elif answer_type == "akshara_count":
            answer = int_to_kannada(len(clusters))
        elif answer_type == "ottakshara_in_number":
            ott = [c for c in clusters if HALANT in c]
            if not ott:
                continue
            answer = ott[-1]  # e.g. ಎಪ್ಪತ್ತು → ತ್ತು
        else:
            answer = ", ".join(aksharas)
        key = (word, template_idx, answer_type)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Deduplicate by (query, answer) - some templates produce same Q&A (e.g. duplicate template text)
samples = []
seen_qa = set()
for q, a in unique_combinations.values():
    if (q, a) not in seen_qa:
        seen_qa.add((q, a))
        samples.append((q, a))
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    if random.random() < 0.5:
        num = random.randint(1, 100)
        if num <= len(NUMBERS):
            word = NUMBERS[num - 1]
            template, _ = random.choice(TEMPLATES_NAME)
            suffix = get_genitive_suffix(str(num))
            num_k = int_to_kannada(num)
            query = template.format(num=num, num_k=num_k, suffix=suffix)
            answer = word
            if (query, answer) not in seen_qa:
                seen_qa.add((query, answer))
                samples.append((query, answer))
                no_progress = 0
            else:
                no_progress += 1
    else:
        word = random.choice(NUMBERS)
        aksharas = get_kannada_aksharas(word)
        clusters = get_kannada_grapheme_clusters(word)
        if len(aksharas) > 0:
            template, answer_type = random.choice(TEMPLATES_SPELLING)
            query = template.format(word=word)
            if answer_type == "spelling":
                answer = ", ".join(aksharas)
            elif answer_type == "varnaviccheda":
                answer = get_varnavichcheda_str(word)
            elif answer_type == "akshara_count":
                answer = int_to_kannada(len(clusters))
            elif answer_type == "ottakshara_in_number":
                ott = [c for c in clusters if HALANT in c]
                if ott:
                    answer = ott[-1]
                else:
                    no_progress += 1
                    continue
            else:
                answer = ", ".join(aksharas)
            if (query, answer) not in seen_qa:
                seen_qa.add((query, answer))
                samples.append((query, answer))
                no_progress = 0
            else:
                no_progress += 1
        else:
            no_progress += 1

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S8 Number Spelling (Kannada): Generated {len(samples)} samples")
