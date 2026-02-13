#!/usr/bin/env python3
"""
Generate Statement 2: Letter Position (ಅಕ್ಷರ ಸ್ಥಿತಿ) questions - Kannada
User-specified templates: first/last/Nth/middle letter, position of char, at-end yes/no, fifth exists.
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

# Position names for "position of char" answers
POSITION_NAMES = [
    ("ಮೊದಲನೇ", 1),
    ("ಎರಡನೇ", 2),
    ("ಮೂರನೇ", 3),
    ("ನಾಲ್ಕನೇ", 4),
    ("ಐದನೇ", 5),
    ("ಆರನೇ", 6),
    ("ಏಳನೇ", 7),
    ("ಎಂಟನೇ", 8),
    ("ಒಂಬತ್ತನೇ", 9),
    ("ಹತ್ತನೇ", 10),
]

VOWELS = set([chr(c) for c in range(0x0C85, 0x0C91) if c not in [0x0C8C, 0x0C8E]]) # ಅ-ಔ, excluding deprecated
CONSONANTS = set([chr(c) for c in range(0x0C95, 0x0CB9) if chr(c) not in ['ಱ', 'ೞ']]) # ಕ-ಹ (excluding old/deprecated chars)

# User-specified Letter Position templates with generation type.
# Types: first, last, second, third, fourth, fifth, sixth, middle, position_of, at_end,
#        fifth_exists, second_from_end, second_and_fourth, first_vowel_or_consonant
TEMPLATES = [
    ('"{word}" ಪದದ ಮೊದಲ ಅಕ್ಷರ ಯಾವುದು?', "first"),
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರ ಯಾವುದು?', "last"),
    ('"{word}" ಪದದಲ್ಲಿ ಮೂರನೇ ಅಕ್ಷರ ಯಾವುದು?', "third"),
    ('"{word}" ಪದದ ಎರಡನೇ ಅಕ್ಷರ ಯಾವುದು?', "second"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ?', "position_of"),
    ('"{word}" ಪದದ ಮಧ್ಯದ ಅಕ್ಷರ ಯಾವುದು?', "middle"),
    ('"{word}" ಪದದ ನಾಲ್ಕನೇ ಅಕ್ಷರ ಯಾವುದು?', "fourth"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಕೊನೆಯಲ್ಲಿದೆಯೇ?', "at_end"),
    ('"{word}" ಪದದ ಆರಂಭದ ಅಕ್ಷರ ಯಾವುದು?', "first"),
    ('"{word}" ಪದದಲ್ಲಿ ಐದನೇ ಅಕ್ಷರ ಇದೆಯೇ?', "fifth_exists"),
    ('"{word}" ಪದದಲ್ಲಿ ಐದನೇ ಸ್ಥಾನದಲ್ಲಿರುವ ಅಕ್ಷರ ಯಾವುದು?', "fifth"),
    ('"{word}" ಪದದ ಕೊನೆಯಿಂದ ಎರಡನೇ ಅಕ್ಷರ ಯಾವುದು?', "second_from_end"),
    ('"{word}" ಪದದ ನಾಲ್ಕನೇ ಅಕ್ಷರ ತಿಳಿಸಿ?', "fourth"),
    ('"{word}" ಪದದಲ್ಲಿ ಮಧ್ಯದ ಅಕ್ಷರ ಯಾವುದು?', "middle"),
    ('"{word}" ಪದದ ಆರನೇ ಅಕ್ಷರವನ್ನು ಗುರುತಿಸಿ?', "sixth"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಜಾಗದಲ್ಲಿದೆ?', "position_of"),
    ('"{word}" ಪದದ ಎರಡನೇ ಮತ್ತು ನಾಲ್ಕನೇ ಅಕ್ಷರಗಳು ಯಾವುವು?', "second_and_fourth"),
    ('"{word}" ಪದದ ಮೊದಲನೇ ಅಕ್ಷರ ಸ್ವರವೇ ಅಥವಾ ವ್ಯಂಜನವೇ?', "first_vowel_or_consonant"),
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರ ಯಾವುದು?', "last"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರವು ಎಷ್ಟನೇ ಅಕ್ಷರ?', "position_of"),
]

EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70
all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
unique_words = list(set(all_words))

samples = []
target_count = 25800
seen = set()


def get_position_name(pos_1based: int) -> str:
    if 1 <= pos_1based <= len(POSITION_NAMES):
        return POSITION_NAMES[pos_1based - 1][0]
    return f"{pos_1based}ನೇ"


for word in unique_words:
    clusters = get_kannada_grapheme_clusters(word)
    n = len(clusters)
    if n == 0:
        continue

    for template, ttype in TEMPLATES:
        if ttype == "first":
            q = template.format(word=word)
            a = clusters[0]
            key = (word, "first")
        elif ttype == "last":
            q = template.format(word=word)
            a = clusters[-1]
            key = (word, "last")
        elif ttype == "second":
            q = template.format(word=word)
            if n >= 2:
                a = clusters[1]
            else:
                a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
            key = (word, "second", template)
        elif ttype == "third":
            q = template.format(word=word)
            if n >= 3:
                a = clusters[2]
            else:
                a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
            key = (word, "third", template)
        elif ttype == "fourth":
            q = template.format(word=word)
            if n >= 4:
                a = clusters[3]
            else:
                a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
            key = (word, "fourth", template)
        elif ttype == "fifth":
            q = template.format(word=word)
            if n >= 5:
                a = clusters[4]
            else:
                a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
            key = (word, "fifth", template)
        elif ttype == "sixth":
            q = template.format(word=word)
            if n >= 6:
                a = clusters[5]
            else:
                a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
            key = (word, "sixth", template)
        elif ttype == "middle":
            q = template.format(word=word)
            mid = n // 2
            a = clusters[mid]
            key = (word, "middle", template)
        elif ttype == "position_of":
            for c in clusters:
                pos_1 = next((i + 1 for i, x in enumerate(clusters) if x == c), None)
                if pos_1 is None:
                    continue
                q = template.format(word=word, char=c)
                if "ಅಕ್ಷರವು ಎಷ್ಟನೇ ಅಕ್ಷರ?" in template: # Specific phrasing from user
                    a = f"{get_position_name(pos_1)} ಅಕ್ಷರ"
                else:
                    a = get_position_name(pos_1)
                key = (word, "position_of", c, template)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue # Continue to next template after iterating all chars
        elif ttype == "at_end":
            for c in clusters:
                q = template.format(word=word, char=c)
                a = "ಹೌದು" if clusters[-1] == c else "ಇಲ್ಲ"
                key = (word, "at_end", c, template)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "fifth_exists":
            q = template.format(word=word)
            if n >= 5:
                a = f"ಹೌದು, {clusters[4]}"
            else:
                a = "ಇಲ್ಲ"
            key = (word, "fifth_exists", template)
        elif ttype == "second_from_end" and n >= 2:
            q = template.format(word=word)
            a = clusters[-2]
            key = (word, "second_from_end", template)
        elif ttype == "second_and_fourth" and n >= 4:
            q = template.format(word=word)
            a = f"{clusters[1]}, {clusters[3]}"
            key = (word, "second_and_fourth", template)
        elif ttype == "first_vowel_or_consonant":
            first_char = clusters[0]
            q = template.format(word=word)
            if first_char in VOWELS:
                a = "ಸ್ವರ"
            elif first_char in CONSONANTS:
                a = "ವ್ಯಂಜನ"
            else:
                continue # Skip if not a clear vowel or consonant
            key = (word, "first_vowel_or_consonant", template)
        else:
            continue

        if key and key not in seen:
            seen.add(key)
            samples.append((q, a))

# Fill to target with random samples
while len(samples) < target_count:
    word = random.choice(unique_words)
    clusters = get_kannada_grapheme_clusters(word)
    n = len(clusters)
    if n == 0:
        continue

    template, ttype = random.choice(TEMPLATES)
    q, a = None, None  # Initialize q and a

    if ttype == "first":
        q = template.format(word=word)
        a = clusters[0]
    elif ttype == "last":
        q = template.format(word=word)
        a = clusters[-1]
    elif ttype == "second":
        q = template.format(word=word)
        if n >= 2:
            a = clusters[1]
        else:
            a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
    elif ttype == "third":
        q = template.format(word=word)
        if n >= 3:
            a = clusters[2]
        else:
            a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
    elif ttype == "fourth":
        q = template.format(word=word)
        if n >= 4:
            a = clusters[3]
        else:
            a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
    elif ttype == "fifth":
        q = template.format(word=word)
        if n >= 5:
            a = clusters[4]
        else:
            a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
    elif ttype == "sixth":
        q = template.format(word=word)
        if n >= 6:
            a = clusters[5]
        else:
            a = f"ಇಲ್ಲ, ಇದರಲ್ಲಿರುವುದು {n} ಅಕ್ಷರಗಳು"
    elif ttype == "middle":
        q = template.format(word=word)
        a = clusters[n // 2]
    elif ttype == "position_of":
        c = random.choice(clusters)
        pos_1 = next(i + 1 for i, x in enumerate(clusters) if x == c)
        q = template.format(word=word, char=c)
        if "ಅಕ್ಷರವು ಎಷ್ಟನೇ ಅಕ್ಷರ?" in template:
            a = f"{get_position_name(pos_1)} ಅಕ್ಷರ"
        else:
            a = get_position_name(pos_1)
    elif ttype == "at_end":
        c = random.choice(clusters)
        q = template.format(word=word, char=c)
        a = "ಹೌದು" if clusters[-1] == c else "ಇಲ್ಲ"
    elif ttype == "fifth_exists":
        q = template.format(word=word)
        a = f"ಹೌದು, {clusters[4]}" if n >= 5 else "ಇಲ್ಲ"
    elif ttype == "second_from_end" and n >= 2:
        q = template.format(word=word)
        a = clusters[-2]
    elif ttype == "second_and_fourth" and n >= 4:
        q = template.format(word=word)
        a = f"{clusters[1]}, {clusters[3]}"
    elif ttype == "first_vowel_or_consonant":
        first_char = clusters[0]
        q = template.format(word=word)
        if first_char in VOWELS:
            a = "ಸ್ವರ"
        elif first_char in CONSONANTS:
            a = "ವ್ಯಂಜನ"
        else:
            q, a = None, None  # Reset q, a if not a clear vowel/consonant
    else:
        q, a = None, None  # Reset q, a for other unhandled types

    if q is not None and a is not None:
        samples.append((q, a))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S2 Letter Position (Kannada): Generated {len(samples)} samples")
