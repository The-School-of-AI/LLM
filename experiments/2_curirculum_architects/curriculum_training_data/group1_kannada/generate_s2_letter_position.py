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
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import (  # noqa: E402
    format_qa_pair_kannada,
    int_to_kannada,
    int_to_kannada_word,
)

HALANT = "\u0ccd"  # Virama for ottakshara check
# Mahaprana (aspirated) consonants: ಖ, ಛ, ಠ, ಥ, ಫ, ಘ, ಝ, ಢ, ಧ, ಭ
MAHAPRANA = set("ಖಛಠಥಫಘಝಢಧಭ")
VOWEL_SIGNS = set("ಾಿೀುೂೃೄೆೇೈೊೋೌ")  # For gunitakshara (conjunct) check

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

VOWELS = set(
    [chr(c) for c in range(0x0C85, 0x0C91) if c not in [0x0C8C, 0x0C8E]]
)  # ಅ-ಔ, excluding deprecated
CONSONANTS = set(
    [chr(c) for c in range(0x0C95, 0x0CB9) if chr(c) not in ["ಱ", "ೞ"]]
)  # ಕ-ಹ (excluding old/deprecated chars)

# User-specified Letter Position templates with generation type.
# Types: first, last, second, third, fourth, fifth, sixth, middle, position_of, at_end,
#        fifth_exists, second_from_end, second_and_fourth, first_vowel_or_consonant,
#        position_of_numeric, char_is_first, third_from_end, char_in_middle, aarambhika,
#        second_vowel_or_gunita, last_is_ottakshara, second_alpa_mahaprana, char_repeated
TEMPLATES = [
    ('"{word}" ಪದದ ಮೊದಲ ಅಕ್ಷರ ಯಾವುದು?', "first"),
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರ ಯಾವುದು?', "last"),
    ('"{word}" ಪದದಲ್ಲಿ ಮೂರನೇ ಅಕ್ಷರ ಯಾವುದು?', "third"),
    ('"{word}" ಪದದ ಎರಡನೇ ಅಕ್ಷರ ಯಾವುದು?', "second"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಯಾವ ಸ್ಥಾನದಲ್ಲಿದೆ?', "position_of"),
    ('"{word}" ಪದದ ಮಧ್ಯದ ಅಕ್ಷರ ಯಾವುದು?', "middle"),
    ('"{word}" ಪದದ ನಾಲ್ಕನೇ ಅಕ್ಷರ ಯಾವುದು?', "fourth"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರ ಕೊನೆಯಲ್ಲಿದೆಯೇ?', "at_end"),
    ('"{word}" ಪದದಲ್ಲಿ ಐದನೇ ಅಕ್ಷರ ಇದೆಯೇ?', "fifth_exists"),
    ('"{word}" ಪದದಲ್ಲಿ ಐದನೇ ಸ್ಥಾನದಲ್ಲಿರುವ ಅಕ್ಷರ ಯಾವುದು?', "fifth"),
    ('"{word}" ಪದದ ಕೊನೆಯಿಂದ ಎರಡನೇ ಅಕ್ಷರ ಯಾವುದು?', "second_from_end"),
    ('"{word}" ಪದದ ಆರನೇ ಅಕ್ಷರವನ್ನು ಗುರುತಿಸಿ?', "sixth"),
    ('"{word}" ಪದದ ಎರಡನೇ ಮತ್ತು ನಾಲ್ಕನೇ ಅಕ್ಷರಗಳು ಯಾವುವು?', "second_and_fourth"),
    ('"{word}" ಪದದ ಮೊದಲನೇ ಅಕ್ಷರ ಸ್ವರವೇ ಅಥವಾ ವ್ಯಂಜನವೇ?', "first_vowel_or_consonant"),
    # New templates 15-24
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರವು ಎಷ್ಟನೇ ಸ್ಥಾನದಲ್ಲಿದೆ?', "position_of_numeric"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರವು ಮೊದಲ ಅಕ್ಷರವೇ?', "char_is_first"),
    ('"{word}" ಪದದ ಕೊನೆಯಿಂದ ಮೂರನೇ ಅಕ್ಷರ ಯಾವುದು?', "third_from_end"),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರವು ಮಧ್ಯದಲ್ಲಿ ಬಂದಿದೆಯೇ?', "char_in_middle"),
    ('"{word}" ಪದದ ಆರಂಭಿಕ ಅಕ್ಷರ ಯಾವುದು?', "aarambhika"),
    ('"{word}" ಪದದ 2ನೇ ಅಕ್ಷರ ಸ್ವರವೋ ಅಥವಾ ಗುಣಿತಾಕ್ಷರವೋ?', "second_vowel_or_gunita"),
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರವು ಒತ್ತಕ್ಷರವೇ?', "last_is_ottakshara"),
    (
        '"{word}" ಪದದ ಎರಡನೇ ಅಕ್ಷರವು ಅಲ್ಪಪ್ರಾಣವೇ ಅಥವಾ ಮಹಾಪ್ರಾಣವೇ?',
        "second_alpa_mahaprana",
    ),
    ('"{word}" ಪದದಲ್ಲಿ "{char}" ಅಕ್ಷರವು ಒಂದಕ್ಕಿಂತ ಹೆಚ್ಚು ಬಾರಿ ಇದೆಯೇ?', "char_repeated"),
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
    return f"{int_to_kannada(pos_1based)}ನೇ"


# Position names for "Nth letter doesn't exist" answer format
POSITION_NAME_FOR_NTH = {
    "second": "ಎರಡನೇ",
    "third": "ಮೂರನೇ",
    "fourth": "ನಾಲ್ಕನೇ",
    "fifth": "ಐದನೇ",
    "sixth": "ಆರನೇ",
}


def _answer_nth_letter_missing(word: str, n: int, ttype: str) -> str:
    """Format when requested Nth letter doesn't exist. E.g. ಕಾಣಿಕೆ ನಾಲ್ಕನೇ → ಕೇವಲ ಮೂರು ಅಕ್ಷರಗಳಿವೆ, ನಾಲ್ಕನೇ ಅಕ್ಷರ ಇಲ್ಲ."""
    pos_name = POSITION_NAME_FOR_NTH.get(ttype, f"{int_to_kannada(n + 1)}ನೇ")
    n_word = int_to_kannada_word(n)
    return f'"{word}" ಎಂಬ ಪದದಲ್ಲಿ ಕೇವಲ {n_word} ಅಕ್ಷರಗಳಿವೆ, {pos_name} ಅಕ್ಷರ ಇಲ್ಲ.'


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
            a = clusters[1] if n >= 2 else _answer_nth_letter_missing(word, n, "second")
            key = (word, "second", template)
        elif ttype == "third":
            q = template.format(word=word)
            a = clusters[2] if n >= 3 else _answer_nth_letter_missing(word, n, "third")
            key = (word, "third", template)
        elif ttype == "fourth":
            q = template.format(word=word)
            a = clusters[3] if n >= 4 else _answer_nth_letter_missing(word, n, "fourth")
            key = (word, "fourth", template)
        elif ttype == "fifth":
            q = template.format(word=word)
            a = clusters[4] if n >= 5 else _answer_nth_letter_missing(word, n, "fifth")
            key = (word, "fifth", template)
        elif ttype == "sixth":
            q = template.format(word=word)
            a = clusters[5] if n >= 6 else _answer_nth_letter_missing(word, n, "sixth")
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
                a = get_position_name(pos_1)
                key = (word, "position_of", c)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "position_of_numeric":
            for c in clusters:
                pos_1 = next((i + 1 for i, x in enumerate(clusters) if x == c), None)
                if pos_1 is None:
                    continue
                q = template.format(word=word, char=c)
                a = int_to_kannada(pos_1)
                key = (word, "position_of_numeric", c)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "at_end":
            for c in clusters:
                q = template.format(word=word, char=c)
                a = "ಹೌದು" if clusters[-1] == c else "ಇಲ್ಲ"  # ಕೊನೆಯಲ್ಲಿದೆಯೇ? → presence → ಇಲ್ಲ
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
                a = "ಅಲ್ಲ"
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
                continue
            key = (word, "first_vowel_or_consonant", template)
        elif ttype == "char_is_first":
            for c in clusters:
                q = template.format(word=word, char=c)
                a = "ಹೌದು" if clusters[0] == c else "ಅಲ್ಲ"
                key = (word, "char_is_first", c)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "third_from_end" and n >= 3:
            q = template.format(word=word)
            a = clusters[-3]
            key = (word, "third_from_end", template)
        elif ttype == "char_in_middle":
            mid_idx = n // 2
            for c in clusters:
                q = template.format(word=word, char=c)
                a = (
                    "ಹೌದು" if clusters[mid_idx] == c else "ಇಲ್ಲ"
                )  # ಮಧ್ಯದಲ್ಲಿ ಬಂದಿದೆಯೇ? → presence → ಇಲ್ಲ
                key = (word, "char_in_middle", c)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        elif ttype == "aarambhika":
            q = template.format(word=word)
            a = clusters[0]
            key = (word, "aarambhika", template)
        elif ttype == "second_vowel_or_gunita" and n >= 2:
            c2 = clusters[1]
            q = template.format(word=word)
            if c2[0] in VOWELS:
                a = "ಸ್ವರ"
            elif HALANT in c2 or any(ch in VOWEL_SIGNS for ch in c2[1:]):
                a = f"ಗುಣಿತಾಕ್ಷರ ({c2})"
            else:
                a = "ವ್ಯಂಜನ"
            key = (word, "second_vowel_or_gunita", template)
        elif ttype == "last_is_ottakshara":
            q = template.format(word=word)
            a = "ಹೌದು" if HALANT in clusters[-1] else "ಅಲ್ಲ"
            key = (word, "last_is_ottakshara", template)
        elif ttype == "second_alpa_mahaprana" and n >= 2:
            c2_first = clusters[1][0]
            q = template.format(word=word)
            if c2_first in MAHAPRANA:
                a = f"ಮಹಾಪ್ರಾಣ ({clusters[1]})"
            elif c2_first in CONSONANTS:
                a = f"ಅಲ್ಪಪ್ರಾಣ ({clusters[1]})"
            else:
                continue
            key = (word, "second_alpa_mahaprana", template)
        elif ttype == "char_repeated":
            for c in clusters:
                count = sum(1 for x in clusters if x == c)
                q = template.format(word=word, char=c)
                a = "ಹೌದು" if count > 1 else "ಇಲ್ಲ"
                key = (word, "char_repeated", c)
                if key not in seen:
                    seen.add(key)
                    samples.append((q, a))
            continue
        else:
            continue

        if key and key not in seen:
            seen.add(key)
            samples.append((q, a))

# Fill to target with random samples (deduplicate, no duplicates)
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
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
        a = clusters[1] if n >= 2 else _answer_nth_letter_missing(word, n, "second")
    elif ttype == "third":
        q = template.format(word=word)
        a = clusters[2] if n >= 3 else _answer_nth_letter_missing(word, n, "third")
    elif ttype == "fourth":
        q = template.format(word=word)
        a = clusters[3] if n >= 4 else _answer_nth_letter_missing(word, n, "fourth")
    elif ttype == "fifth":
        q = template.format(word=word)
        a = clusters[4] if n >= 5 else _answer_nth_letter_missing(word, n, "fifth")
    elif ttype == "sixth":
        q = template.format(word=word)
        a = clusters[5] if n >= 6 else _answer_nth_letter_missing(word, n, "sixth")
    elif ttype == "middle":
        q = template.format(word=word)
        a = clusters[n // 2]
    elif ttype == "position_of":
        c = random.choice(clusters)
        pos_1 = next(i + 1 for i, x in enumerate(clusters) if x == c)
        q = template.format(word=word, char=c)
        a = get_position_name(pos_1)
    elif ttype == "position_of_numeric":
        c = random.choice(clusters)
        pos_1 = next(i + 1 for i, x in enumerate(clusters) if x == c)
        q = template.format(word=word, char=c)
        a = int_to_kannada(pos_1)
    elif ttype == "at_end":
        c = random.choice(clusters)
        q = template.format(word=word, char=c)
        a = "ಹೌದು" if clusters[-1] == c else "ಇಲ್ಲ"  # ಕೊನೆಯಲ್ಲಿದೆಯೇ? → presence → ಇಲ್ಲ
    elif ttype == "fifth_exists":
        q = template.format(word=word)
        a = f"ಹೌದು, {clusters[4]}" if n >= 5 else "ಅಲ್ಲ"
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
            q, a = None, None
    elif ttype == "char_is_first":
        c = random.choice(clusters)
        q = template.format(word=word, char=c)
        a = "ಹೌದು" if clusters[0] == c else "ಅಲ್ಲ"
    elif ttype == "third_from_end" and n >= 3:
        q = template.format(word=word)
        a = clusters[-3]
    elif ttype == "char_in_middle":
        c = random.choice(clusters)
        mid_idx = n // 2
        q = template.format(word=word, char=c)
        a = (
            "ಹೌದು" if clusters[mid_idx] == c else "ಇಲ್ಲ"
        )  # ಮಧ್ಯದಲ್ಲಿ ಬಂದಿದೆಯೇ? → presence → ಇಲ್ಲ
    elif ttype == "aarambhika":
        q = template.format(word=word)
        a = clusters[0]
    elif ttype == "second_vowel_or_gunita" and n >= 2:
        c2 = clusters[1]
        q = template.format(word=word)
        if c2[0] in VOWELS:
            a = "ಸ್ವರ"
        elif HALANT in c2 or any(ch in VOWEL_SIGNS for ch in c2[1:]):
            a = f"ಗುಣಿತಾಕ್ಷರ ({c2})"
        else:
            a = "ವ್ಯಂಜನ"
    elif ttype == "last_is_ottakshara":
        q = template.format(word=word)
        a = "ಹೌದು" if HALANT in clusters[-1] else "ಅಲ್ಲ"
    elif ttype == "second_alpa_mahaprana" and n >= 2:
        c2_first = clusters[1][0]
        q = template.format(word=word)
        if c2_first in MAHAPRANA:
            a = f"ಮಹಾಪ್ರಾಣ ({clusters[1]})"
        elif c2_first in CONSONANTS:
            a = f"ಅಲ್ಪಪ್ರಾಣ ({clusters[1]})"
        else:
            q, a = None, None
    elif ttype == "char_repeated":
        c = random.choice(clusters)
        count = sum(1 for x in clusters if x == c)
        q = template.format(word=word, char=c)
        a = "ಹೌದು" if count > 1 else "ಇಲ್ಲ"
    else:
        q, a = None, None

    if q is not None and a is not None and (q, a) not in seen_qa:
        seen_qa.add((q, a))
        samples.append((q, a))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S2 Letter Position (Kannada): Generated {len(samples)} samples")
