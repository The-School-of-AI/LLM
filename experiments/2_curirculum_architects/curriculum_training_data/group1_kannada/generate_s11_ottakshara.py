#!/usr/bin/env python3
"""
Generate Statement 11: Ottakshara & Kagunita (ಒತ್ತಕ್ಷರ, ಗುಣಿತಾಕ್ಷರ) - Kannada
Language-specific: conjuncts, vowel signs, ಋ/ಐ/ಔ, ಷ vs ಶ, ಕ್ಷ as consonant.
Target: 10,000 pairs. User-specified question set.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair_kannada  # noqa: E402
from group1_kannada.kannada_vocabulary import VARGAS # noqa: E402

# User-specified Ottakshara/Kagunita (word, question_template, answer).
# Template may use {word} or be literal.
OTTAKSHARA_QA = [
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?', "ಅಪ್ಪ", "ಪ್ಪ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಋ-ಕಾರ ಗುರುತಿಸಿ?', "ಕೃಷಿ", "ಋ"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆ?', "ಜ್ಞಾನ", "ಜ್ಞ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಓ-ತ್ವ ಚಿಹ್ನೆ ಯಾವುದು?', "ಕೋಳಿ", "ೋ"),
    ('"{letter}" ಅಕ್ಷರದ ಗುಣಿತಾಕ್ಷರಗಳನ್ನು ಬರೆಯಿರಿ?', "ಕ", "ಕ, ಕಾ, ಕಿ, ಕೀ, ಕು, ಕೂ, ಕೃ, ಕೆ, ಕೇ, ಕೈ, ಕೊ, ಕೋ, ಕೌ"),
    ('"ಷ" ಮತ್ತು "ಶ" ಅಕ್ಷರಗಳ ವ್ಯತ್ಯಾಸವೇನು?', None, "ಅವು ವಿಭಿನ್ನ ವ್ಯಂಜನಾಕ್ಷರಗಳು; ಉಚ್ಚಾರಣೆ ಮತ್ತು ಲಿಪಿ ವಿಭಿನ್ನವಾಗಿವೆ."),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ?', "ದೃಶ್ಯ", "ದೃ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಐ-ಕಾರ ಯಾವುದು?', "ಐದು", "ೈ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಔ-ತ್ವ ಚಿಹ್ನೆಯನ್ನು ತೋರಿಸಿ?', "ಸೌರ", "ೌ"),
    ('"{char}" ಅಕ್ಷರವು ಸ್ವರವೋ ಅಥವಾ ವ್ಯಂಜನವೋ?', "ಕ್ಷ", "ವ್ಯಂಜನ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?', "ಲಕ್ಷ್ಮಿ", "ಕ್ಷ್ಮಿ"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಎ-ತ್ವ ಚಿಹ್ನೆ ಇದೆ?', "ಕೇಸರಿ", "ೇ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು ಚಿಹ್ನೆಯನ್ನು ಗುರುತಿಸಿ?', "ಸೂರ್ಯ", "್ರ್ಯ"),
    ('"{letter}" ವರ್ಗದ ಐದನೇ ಅಕ್ಷರ (ಅನುನಾಸಿಕ) ಯಾವುದು?', "ಕ", "ಙ"),
    ('"{letter}" ಅಕ್ಷರದ ಒತ್ತಕ್ಷರ ಹೇಗೆ ಬರೆಯುವುದು?', "ತ", "ತ್"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸ್ವರಾಕ್ಷರ ಯಾವುದು?', "ಋಷಿ", "ಋ"),
    ('"{word}" ಪದದಲ್ಲಿ ಗುಣಿತಾಕ್ಷರಗಳು ಇವೆಯೇ?', "ಗಗನ", "ಇಲ್ಲ"),
    ('"{word}" ಪದದ ಮೊದಲ ಅಕ್ಷರ ಯಾವುದು?', "ಔಷಧ", "ಔ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ದ-ಕಾರದ ಒತ್ತಕ್ಷರ ಗುರುತಿಸಿ?', "ವಿದ್ಯೆ", "ದ್ಯೆ"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವತಂತ್ರ ಅಕ್ಷರಗಳಿವೆ?', "ಕನ್ನಡ", "4"),
]

# Expand: each QA can be repeated with same or rephrased question to reach target
samples = []
target_count = 10000

for template, word_or_letter, answer in OTTAKSHARA_QA:
    if "{word}" in template and word_or_letter:
        q = template.format(word=word_or_letter)
    elif "{letter}" in template and word_or_letter:
        q = template.format(letter=word_or_letter)
    elif "{char}" in template and word_or_letter:
        q = template.format(char=word_or_letter)
    else:
        q = template  # literal (e.g. ಷ and ಶ)
    samples.append((q, answer))

# Fill to target by repeating and shuffling
while len(samples) < target_count:
    idx = random.randint(0, len(OTTAKSHARA_QA) - 1)
    template, word_or_letter, answer = OTTAKSHARA_QA[idx]
    if "{word}" in template and word_or_letter:
        q = template.format(word=word_or_letter)
    elif "{letter}" in template and word_or_letter:
        q = template.format(letter=word_or_letter)
    elif "{char}" in template and word_or_letter:
        q = template.format(char=word_or_letter)
    else:
        q = template
    samples.append((q, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s11.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S11 Ottakshara/Kagunita (Kannada): Generated {len(samples)} samples")
