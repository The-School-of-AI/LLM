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
from group1_kannada.kannada_grammar import get_kannada_aksharas  # noqa: E402
from group1_kannada.kannada_grammar import get_genitive_suffix  # noqa: E402
from group1_kannada.kannada_vocabulary import NUMBERS  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Question templates - number to name (ಹೆಸರು). Numeral takes ರ (72 ರ, 17 ರ).
TEMPLATES_NAME = [
    "{num} {suffix} ಹೆಸರು ಏನು?",
    "{num} {suffix} ಕನ್ನಡ ಹೆಸರು ಏನು?",
    "{num} ಅನ್ನು ಕನ್ನಡದಲ್ಲಿ ಏನು ಎನ್ನುತ್ತಾರೆ?",
    "{num} ಸಂಖ್ಯೆಯ ಹೆಸರು ಏನು?",
    "{num} ಅಂಕೆಯ ಹೆಸರು ಏನು?",
]

TEMPLATES_SPELLING = [
    '"{word}" ಪದದ ಕಾಗುಣಿತ ಏನು?',
    '"{word}" ಪದವನ್ನು ಹೇಗೆ ಬರೆಯುವುದು?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳು ಯಾವುವು?',
    '"{word}" ಪದದ ಕಾಗುಣಿತ ತಿಳಿಸಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳು ಯಾವುವು?',
    'ಈ ಸಂಖ್ಯೆಯ ಪದ "{word}" ಅಕ್ಷರ ಅಕ್ಷರವಾಗಿ ಬರೆಯಿರಿ?',
]

samples = []
target_count = 10000
unique_combinations = {}

# Number to name (suffix for numeral is ರ)
for num in range(1, 101):
    if num <= len(NUMBERS):
        word = NUMBERS[num - 1]
    else:
        continue
    suffix = get_genitive_suffix(str(num))  # ರ for numerals
    for template_idx, template in enumerate(TEMPLATES_NAME):
        query = template.format(num=num, suffix=suffix)
        answer = word
        key = (num, template_idx, "name")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Name to spelling (akshara-level, per Kannada linguistics)
for word in NUMBERS:
    aksharas = get_kannada_aksharas(word)
    if len(aksharas) == 0:
        continue

    for template_idx, template in enumerate(TEMPLATES_SPELLING):
        query = template.format(word=word)
        answer = ", ".join(aksharas)
        key = (word, template_idx, "spelling")
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
while len(samples) < target_count:
    if random.random() < 0.5:
        num = random.randint(1, 100)
        if num <= len(NUMBERS):
            word = NUMBERS[num - 1]
            template = random.choice(TEMPLATES_NAME)
            suffix = get_genitive_suffix(str(num))
            query = template.format(num=num, suffix=suffix)
            answer = word
            samples.append((query, answer))
    else:
        word = random.choice(NUMBERS)
        aksharas = get_kannada_aksharas(word)
        if len(aksharas) > 0:
            template = random.choice(TEMPLATES_SPELLING)
            query = template.format(word=word)
            answer = ", ".join(aksharas)
            samples.append((query, answer))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s8.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S8 Number Spelling (Kannada): Generated {len(samples)} samples")
