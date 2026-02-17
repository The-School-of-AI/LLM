#!/usr/bin/env python3
"""
Generate Statement 4: Letter Count (ಅಕ್ಷರ ಗಣನೆ) questions - Kannada
Target: 25,800 pairs (12.9% of 200,000)
"""

import os
import random
import re
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
from prompt_utils import format_qa_pair_kannada, int_to_kannada  # noqa: E402

HALANT = "\u0ccd"
# Dirgha (long) vowels: ಆ ಈ ಊ ಏ ಐ ಓ ಔ
DIRGHA_SVARAS = set("ಆಈಊಏಐಓಔ")
# Hrasva (short) vowels: ಅ ಇ ಉ ಋ ಎ ಒ
HRASVA_SVARAS = set("ಅಇಉಋಎಒ")
# Mahaprana (aspirated) consonants
MAHAPRANA = set("ಖಛಠಥಫಘಝಢಧಭ")
# Yogavaha: anusvara ಂ, visarga ಃ
ANUSVARA, VISARGA = "\u0c82", "\u0c83"

# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

VOWELS = set([chr(c) for c in range(0x0C85, 0x0C91) if c not in [0x0C8C, 0x0C8E]])
CONSONANTS = set([chr(c) for c in range(0x0C95, 0x0CB9) if chr(c) not in ["ಱ", "ೞ"]])
REPH = "\u0cb0\u0ccd"  # ರ್ (ra + virama) - reph/arka-vattu


def _count_sajatiya(clusters):
    """Same-consonant gemination (e.g. ಪ್ಪ, ಕ್ಕ)."""
    count = 0
    for c in clusters:
        if HALANT in c:
            parts = c.split(HALANT)
            if (
                len(parts) >= 2
                and parts[0]
                and parts[1]
                and parts[0][-1] == parts[1][0]
            ):
                count += 1
    return count


def _count_vijatiya(clusters):
    """Different-consonant conjunct (e.g. ಸ್ತ, ಕ್ಷ)."""
    count = 0
    for c in clusters:
        if HALANT in c:
            parts = c.split(HALANT)
            if len(parts) >= 2 and parts[0] and parts[1]:
                if parts[0][-1] != parts[1][0]:
                    count += 1
    return count


def _count_yogavaha(word):
    """Count anusvara (ಂ) and visarga (ಃ)."""
    return word.count(ANUSVARA) + word.count(VISARGA)


def _count_arka_vattu(word):
    """Count reph (್ರ) in word."""
    return word.count(REPH)


def _get_arka_vattu_consonants(word: str) -> list[str]:
    """
    Return list of consonants that have arkavattu (ರ್ attaches to them).
    E.g. ಧರ್ಮ -> [ಮ], ತರ್ಕ -> [ಕ], ಸೂರ್ಯ -> [ಯ].
    """
    pattern = REPH + r"([\u0C95-\u0CB9])"
    return re.findall(pattern, word)


# Kannada vowel signs (ಾ ಿ ೀ ು ೂ ೃ ೄ ೆ ೇ ೈ ೊ ೋ ೌ)
VOWEL_SIGNS = set(
    "\u0cbe\u0cbf\u0cc0\u0cc1\u0cc2\u0cc3\u0cc4\u0cc6\u0cc7\u0cc8\u0cca\u0ccb\u0ccc"
)


def _count_varnas(word: str) -> int:
    """
    Count ವರ್ಣಗಳು (basic phonemes) per Kannada grammar.
    ವರ್ಣವಿಚ್ಛೇದನ: decompose each akshara into ಸ್ವರ+ವ್ಯಂಜನ (consonant+halant and vowel).
    E.g. ಪಾವನ: ಪಾ=ಪ್+ಆ, ವ=ವ್+ಅ, ನ=ನ್+ಅ → ಪ್, ಆ, ವ್, ಅ, ನ್, ಅ = 6 ವರ್ಣಗಳು.
    """
    chars = list(word)
    varna_count = 0
    i = 0
    while i < len(chars):
        c = chars[i]
        if 0x0C95 <= ord(c) <= 0x0CB9:  # Consonant
            if i + 1 < len(chars) and chars[i + 1] == HALANT:
                varna_count += 1  # Consonant+halant (conjunct part)
                i += 2
            else:
                # Consonant with vowel (explicit or inherent ಅ)
                varna_count += 2  # consonant+halant + vowel
                i += 1
                # Skip vowel sign if present (already counted as the vowel)
                if i < len(chars) and chars[i] in VOWEL_SIGNS:
                    i += 1
        elif 0x0C85 <= ord(c) <= 0x0C94:  # Independent vowel (ಅ–ಔ)
            varna_count += 1
            i += 1
        elif c in VOWEL_SIGNS:
            # Standalone vowel sign (unusual) - count as 1
            varna_count += 1
            i += 1
        elif c in (ANUSVARA, VISARGA):
            varna_count += 1
            i += 1
        else:
            i += 1
    return varna_count


def _count_matres(word: str) -> int:
    """
    Count ಮಾತ್ರೆಗಳು (prosodic units) per Kannada chandas.
    ಲಘು = 1 matra (akshara with inherent ಅ only).
    ಗುರು = 2 matras (akshara with conjunct/HALANT or explicit vowel sign).
    E.g. ಗುಡ್ಡೆ: ಗು = ೨ (ಗುರು, has ು), ಡ್ಡೆ = ೨ (ಗುರು, conjunct) → ೪ ಮಾತ್ರೆಗಳು.
    """
    clusters = get_kannada_grapheme_clusters(word)
    total = 0
    for akshara in clusters:
        if HALANT in akshara or any(ch in VOWEL_SIGNS for ch in akshara):
            total += 2  # ಗುರು
        else:
            total += 1  # ಲಘು (inherent ಅ only)
    return total


def _count_vyanjanas(word: str) -> int:
    """
    Count ವ್ಯಂಜನಗಳು (consonants) per ವರ್ಣವಿಚ್ಛೇದನ.
    E.g. ನಿರ್ಗಮಿಸಿ: ನ್+ಇ, ರ್+ಗ್+ಅ, ಮ್+ಅ, ಸ್+ಇ → ನ್, ರ್, ಗ್, ಮ್, ಸ್ = ೫ ವ್ಯಂಜನಗಳು.
    """
    chars = list(word)
    count = 0
    i = 0
    while i < len(chars):
        c = chars[i]
        if 0x0C95 <= ord(c) <= 0x0CB9:  # Consonant
            count += (
                1  # Each consonant = 1 vyanjana (consonant+halant in decomposition)
            )
            if i + 1 < len(chars) and chars[i + 1] == HALANT:
                i += 2  # Skip halant
            else:
                i += 1
                if i < len(chars) and chars[i] in VOWEL_SIGNS:
                    i += 1  # Skip vowel sign
        elif 0x0C85 <= ord(c) <= 0x0C94:  # Independent vowel - no consonant
            i += 1
        elif c in VOWEL_SIGNS or c in (ANUSVARA, VISARGA):
            i += 1
        else:
            i += 1
    return count


# User-specified Letter Count templates. answer_type: "count" (number) or "varna_count" or ...
TEMPLATES = [
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
    ('"{word}" ಪದದ ಒಟ್ಟು ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳನ್ನು ಕಾಣಬಹುದು?', "count"),
    (
        '"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ವರ್ಣಗಳಿವೆ?',
        "varna_count",
    ),  # ವರ್ಣ = phoneme; ಅಕ್ಷರ = syllabic
    ('"{word}" ಪದವು ಎರಡು ಅಕ್ಷರದ ಪದವೇ?', "two_letter_yes_no"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಎಣಿಸಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳೆಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವರಗಳಿವೆ?', "vowel_count"),  # Answer: vowel count
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರಗಳನ್ನು ಸೇರಿಸಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಲೆಕ್ಕ ಕೊಡಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒಟ್ಟು ಮಾತ್ರೆಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "matre_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ ಎಂದು ಎಣಿಸಿ?', "count"),
    ('"{word}" ಪದವು ಮೂರು ಅಕ್ಷರದ ಪದವೇ?', "three_letter_yes_no"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳ ಮೊತ್ತ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ತಿಳಿಸಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಕೇವಲ ಎರಡು ಅಕ್ಷರಗಳಿವೆಯೇ?', "two_letter_yes_no"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಎಣಿಕೆ ಮಾಡಿ?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ವ್ಯಂಜನಗಳು ಇವೆ?', "consonant_count"),
    ('"{word}" ಪದದ ಅಕ್ಷರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "count"),
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರವನ್ನು ಸೇರಿಸಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "count"),
    # New templates 21-30
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸಜಾತೀಯ ಒತ್ತಕ್ಷರಗಳಿವೆ?', "sajatiya_ottakshara_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ವಿಜಾತೀಯ ಒತ್ತಕ್ಷರಗಳಿವೆ?', "vijatiya_ottakshara_count"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒಟ್ಟು ಯೋಗವಾಹಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "yogavaha_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಮಹಾಪ್ರಾಣ ಅಕ್ಷರಗಳಿವೆ?', "mahaprana_count"),
    ('"{word}" ಪದದಲ್ಲಿ ದೀರ್ಘ ಸ್ವರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "dirgha_svara_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಹ್ರಸ್ವ ಸ್ವರಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "hrasva_svara_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರವಿಲ್ಲದ ಅಕ್ಷರಗಳು ಎಷ್ಟು?', "no_ottakshara_count"),
    ('"{word}" ಪದವು ನಾಲ್ಕು ಅಕ್ಷರಗಳ ಪದವೇ?', "four_letter_yes_no"),
    ('"{word}" ಪದದಲ್ಲಿ ಅರ್ಕಾವತ್ತುಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು?', "arka_vattu_count"),
    ('"{word}" ಪದದಲ್ಲಿ ಅರ್ಕಾವತ್ತು ಯಾವ ಅಕ್ಷರಕ್ಕೆ ಸೇರಿದೆ?', "arka_vattu_letter"),
    ('"{word}" ಪದದಲ್ಲಿ ಅರ್ಕಾವತ್ತು ಯಾವ ವ್ಯಂಜನಕ್ಕೆ ಸೇರಿದೆ?', "arka_vattu_letter"),
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು (ರೆಫೆ) ಯಾವ ಅಕ್ಷರಕ್ಕೆ ಸೇರಿದೆ?',
        "arka_vattu_letter",
    ),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಗುಣಿತಾಕ್ಷರಗಳನ್ನು ಕಾಣಬಹುದು?', "gunitakshara_count"),
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
# Words that have ಅರ್ಕಾವತ್ತು (ರೆಫೆ/ರ್) - used for arkavattu questions so answers are not always 0
ARKAVATTU_WORDS = [w for w in set(all_words) if _count_arka_vattu(w) > 0]
samples = []
target_count = 25800
unique_combinations = {}

for word in set(all_words):
    clusters = get_kannada_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        continue

    for template_idx, (template, answer_type) in enumerate(TEMPLATES):
        # arka_vattu_letter only for words that have arkavattu (need consonant to name)
        if answer_type == "arka_vattu_letter" and _count_arka_vattu(word) == 0:
            continue
        query = template.format(word=word)
        answer = ""
        if answer_type == "count":
            kc = int_to_kannada(cluster_count)
            answer = f"{kc} ಅಕ್ಷರಗಳು"
        elif answer_type == "two_letter_yes_no":
            answer = "ಹೌದು" if cluster_count == 2 else "ಅಲ್ಲ"
        elif answer_type == "three_letter_yes_no":
            answer = "ಹೌದು" if cluster_count == 3 else "ಅಲ್ಲ"
        elif answer_type == "vowel_count":
            vc = len([c for c in clusters if c[0] in VOWELS])
            answer = f"{int_to_kannada(vc)} ಸ್ವರಗಳು"
        elif answer_type == "consonant_count":
            cc = _count_vyanjanas(word)  # ವರ್ಣವಿಚ್ಛೇದನ level (consonant+halant units)
            answer = f"{int_to_kannada(cc)} ವ್ಯಂಜನಗಳು"
        elif answer_type == "sajatiya_ottakshara_count":
            cnt = _count_sajatiya(clusters)
            answer = int_to_kannada(cnt)
        elif answer_type == "vijatiya_ottakshara_count":
            cnt = _count_vijatiya(clusters)
            answer = int_to_kannada(cnt)
        elif answer_type == "yogavaha_count":
            cnt = _count_yogavaha(word)
            ans = int_to_kannada(cnt)
            answer = f"{ans} (ಅನುಸ್ವಾರ)" if cnt > 0 else ans
        elif answer_type == "mahaprana_count":
            cnt = len([c for c in clusters if c[0] in MAHAPRANA])
            ex = (
                clusters[
                    next((i for i, x in enumerate(clusters) if x[0] in MAHAPRANA), 0)
                ]
                if cnt > 0
                else ""
            )
            answer = f"{int_to_kannada(cnt)} ({ex})" if ex else int_to_kannada(cnt)
        elif answer_type == "dirgha_svara_count":
            cnt = len([c for c in clusters if any(ch in DIRGHA_SVARAS for ch in c)])
            answer = int_to_kannada(cnt)
        elif answer_type == "hrasva_svara_count":
            cnt = len([c for c in clusters if any(ch in HRASVA_SVARAS for ch in c)])
            answer = int_to_kannada(cnt)
        elif answer_type == "no_ottakshara_count":
            cnt = len([c for c in clusters if HALANT not in c])
            answer = int_to_kannada(cnt)
        elif answer_type == "four_letter_yes_no":
            answer = (
                "ಹೌದು"
                if cluster_count == 4
                else f"ಅಲ್ಲ ({int_to_kannada(cluster_count)} ಅಕ್ಷರ)"
            )
        elif answer_type == "arka_vattu_count":
            cnt = _count_arka_vattu(word)
            answer = int_to_kannada(cnt)
        elif answer_type == "arka_vattu_letter":
            cons = _get_arka_vattu_consonants(word)
            if cons:
                answer = (
                    cons[0]
                    if len(cons) == 1
                    else f"{', '.join(cons[:-1])} ಮತ್ತು {cons[-1]}"
                )
            else:
                continue
        elif answer_type == "gunitakshara_count":
            answer = int_to_kannada(cluster_count)
        elif answer_type == "varna_count":
            vc = _count_varnas(word)
            answer = f"{int_to_kannada(vc)} ವರ್ಣಗಳು"
        elif answer_type == "matre_count":
            mc = _count_matres(word)
            answer = f"{int_to_kannada(mc)} ಮಾತ್ರೆಗಳು"
        else:
            continue

        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    template, answer_type = random.choice(TEMPLATES)
    # For arkavattu questions: ~50% from arkavattu words (non-zero), ~50% from all (mix of 0 and non-zero)
    if (
        answer_type in ("arka_vattu_count", "arka_vattu_letter")
        and ARKAVATTU_WORDS
        and random.random() < 0.5
    ):
        word = random.choice(ARKAVATTU_WORDS)
    else:
        word = random.choice(list(set(all_words)))
    clusters = get_kannada_grapheme_clusters(word)
    cluster_count = len(clusters)
    if cluster_count == 0:
        no_progress += 1
        continue
    if answer_type == "arka_vattu_letter" and _count_arka_vattu(word) == 0:
        no_progress += 1
        continue
    query = template.format(word=word)
    answer = ""
    if answer_type == "count":
        kc = int_to_kannada(cluster_count)
        answer = f"{kc} ಅಕ್ಷರಗಳು"
    elif answer_type == "two_letter_yes_no":
        answer = "ಹೌದು" if cluster_count == 2 else "ಅಲ್ಲ"
    elif answer_type == "three_letter_yes_no":
        answer = "ಹೌದು" if cluster_count == 3 else "ಅಲ್ಲ"
    elif answer_type == "vowel_count":
        vc = len([c for c in clusters if c[0] in VOWELS])
        answer = f"{int_to_kannada(vc)} ಸ್ವರಗಳು"
    elif answer_type == "consonant_count":
        cc = _count_vyanjanas(word)  # ವರ್ಣವಿಚ್ಛೇದನ level (consonant+halant units)
        answer = f"{int_to_kannada(cc)} ವ್ಯಂಜನಗಳು"
    elif answer_type == "sajatiya_ottakshara_count":
        answer = int_to_kannada(_count_sajatiya(clusters))
    elif answer_type == "vijatiya_ottakshara_count":
        answer = int_to_kannada(_count_vijatiya(clusters))
    elif answer_type == "yogavaha_count":
        cnt = _count_yogavaha(word)
        ans = int_to_kannada(cnt)
        answer = f"{ans} (ಅನುಸ್ವಾರ)" if cnt > 0 else ans
    elif answer_type == "mahaprana_count":
        cnt = len([c for c in clusters if c[0] in MAHAPRANA])
        ex = (
            clusters[next((i for i, x in enumerate(clusters) if x[0] in MAHAPRANA), 0)]
            if cnt > 0
            else ""
        )
        answer = f"{int_to_kannada(cnt)} ({ex})" if ex else int_to_kannada(cnt)
    elif answer_type == "dirgha_svara_count":
        cnt = len([c for c in clusters if any(ch in DIRGHA_SVARAS for ch in c)])
        answer = int_to_kannada(cnt)
    elif answer_type == "hrasva_svara_count":
        cnt = len([c for c in clusters if any(ch in HRASVA_SVARAS for ch in c)])
        answer = int_to_kannada(cnt)
    elif answer_type == "no_ottakshara_count":
        cnt = len([c for c in clusters if HALANT not in c])
        answer = int_to_kannada(cnt)
    elif answer_type == "four_letter_yes_no":
        answer = (
            "ಹೌದು"
            if cluster_count == 4
            else f"ಅಲ್ಲ ({int_to_kannada(cluster_count)} ಅಕ್ಷರ)"
        )
    elif answer_type == "arka_vattu_count":
        answer = int_to_kannada(_count_arka_vattu(word))
    elif answer_type == "arka_vattu_letter":
        cons = _get_arka_vattu_consonants(word)
        if cons:
            answer = (
                cons[0] if len(cons) == 1 else f"{', '.join(cons[:-1])} ಮತ್ತು {cons[-1]}"
            )
        else:
            no_progress += 1
            continue
    elif answer_type == "gunitakshara_count":
        answer = int_to_kannada(cluster_count)
    elif answer_type == "varna_count":
        vc = _count_varnas(word)
        answer = f"{int_to_kannada(vc)} ವರ್ಣಗಳು"
    elif answer_type == "matre_count":
        mc = _count_matres(word)
        answer = f"{int_to_kannada(mc)} ಮಾತ್ರೆಗಳು"
    else:
        continue
    if (query, answer) not in seen_qa:
        seen_qa.add((query, answer))
        samples.append((query, answer))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s4.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S4 Letter Count (Kannada): Generated {len(samples)} samples")
