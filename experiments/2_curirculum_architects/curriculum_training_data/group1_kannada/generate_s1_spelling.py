#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (ವर्तನಿ) questions - Kannada
Target: 28,600 pairs (14.3% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_grammar import get_kannada_aksharas  # noqa: E402
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
    NUMBERS,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

HALANT = "\u0ccd"
# Vowel sign → independent swara (for varṇa decomposition)
VOWEL_SIGN_TO_SWARA = {
    "\u0cbe": "ಆ",
    "\u0cbf": "ಇ",
    "\u0cc0": "ಈ",
    "\u0cc1": "ಉ",
    "\u0cc2": "ಊ",
    "\u0cc3": "ಋ",
    "\u0cc4": "ೠ",
    "\u0cc6": "ಎ",
    "\u0cc7": "ಏ",
    "\u0cc8": "ಐ",
    "\u0cca": "ಒ",
    "\u0ccb": "ಓ",
    "\u0ccc": "ಔ",
}

# Exclude number words from S1: ವರ್ಣವಿಚ್ಛೇದ/spelling should use other words, not numbers.
NUMBER_WORDS = frozenset(NUMBERS)


# Expand word lists to reach target count (excluding number words)
def _exclude_numbers(words: list) -> list:
    return [w for w in words if w not in NUMBER_WORDS]


EASY_WORDS = _exclude_numbers(EASY_WORDS_UNIQUE) * 50
MEDIUM_WORDS = _exclude_numbers(MEDIUM_WORDS_UNIQUE) * 60
HARD_WORDS = _exclude_numbers(HARD_WORDS_UNIQUE) * 70

# Spelling: sequence of characters in a word (user-specified templates).
TEMPLATES_SPELLING = [
    '"{word}" ಪದವನ್ನು ಅಕ್ಷರಶಃ ಬಿಡಿಸಿ ಬರೆಯಿರಿ?',
    '"{word}" ಪದದ ಸರಿಯಾದ ಕಾಗುಣಿತ ಯಾವುದು?',
    '"{word}" ಎಂಬ ಪದದ ಅಕ್ಷರಗಳ ಜೋಡಣೆ ತಿಳಿಸಿ?',
    '"{word}" ಪದದ ಸ್ಪೆಲ್ಲಿಂಗ್ ಅನ್ನು ಅಕ್ಷರ ಬಿಡದೆ ಹೇಳಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಿ?',
    '"{word}" ಪದವನ್ನು ದೋಷವಿಲ್ಲದೆ ಬರೆಯುವುದು ಹೇಗೆ?',
    '"{word}" ಪದದ ಸ್ಪೆಲ್ಲಿಂಗ್ ಮಾಹಿತಿ ನೀಡಿ?',
    '"{word}" ಪದವನ್ನು ಅಕ್ಷರಗಳಾಗಿ ಒಡೆಯಿರಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರ ರಚನೆ ಏನು?',
    '"{word}" ಪದದ ಸ್ಪೆಲ್ಲಿಂಗ್ ಹೇಳಲು ಸಾಧ್ಯವೇ?',
]
# Letter listing: extract and list components (same answer format: comma-sep characters).
TEMPLATES_LISTING = [
    '"{word}" ಪದದಲ್ಲಿರುವ ಎಲ್ಲಾ ಅಕ್ಷರಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಬೇರೆ ಬೇರೆಯಾಗಿ ಬರೆಯಿರಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮವಾಗಿ ತೋರಿಸಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳ ಪಟ್ಟಿ ನೀಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಒಂದೊಂದಾಗಿ ಹೇಳಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಬರೆಯಿರಿ?',
    '"{word}" ಪದದಲ್ಲಿ ಯಾವ ಯಾವ ಅಕ್ಷರಗಳಿವೆ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾಂಕದಲ್ಲಿ ನೀಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಬಿಡಿಸಿ ಪಟ್ಟಿ ಮಾಡಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಸ್ವರ ಮತ್ತು ವ್ಯಂಜನಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಎಲ್ಲಾ ಸ್ವರಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾನುಗತವಾಗಿ ನೀಡಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಒಂದೊಂದೇ ಅಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ವಿಭಜಿಸಿ ಬರೆಯಿರಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರ ಘಟಕಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಬಿಡಿಸಿ ಪಟ್ಟಿ ರೂಪದಲ್ಲಿ ನೀಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ತೋರಿಸಿಕೊಡಿ?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಪ್ರತ್ಯೇಕವಾಗಿ ತಿಳಿಸಿ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಅಕ್ಷರಗಳು ಯಾವುವು?',
    '"{word}" ಪದದ ಅಕ್ಷರಗಳನ್ನು ಅನುಕ್ರಮವಾಗಿ ಬರೆಯಿರಿ?',
]
# ವರ್ಣವಿಚ್ಛೇದ (letter segmentation): ask for letters/aksharas themselves, not counts.
# Answer: comma-separated aksharas (e.g. ಕ, ನ್, ನ, ಡ).
TEMPLATES_VARNAVICHCHEDA = [
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದ ಮಾಡಿ?',
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದನೆ ತಿಳಿಸಿ?',
    '"{word}" ಪದವನ್ನು ವರ್ಣವಿಚ್ಛೇದದಲ್ಲಿ ಬಿಡಿಸಿ?',
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದ ಪಟ್ಟಿ ನೀಡಿ?',
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದ ಯಾವುದು?',
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದ ಹೇಳಿ?',
    '"{word}" ಪದದ ವರ್ಣವಿಚ್ಛೇದನೆ ಮಾಡಿ ಹೇಳಿ?',
    '"{word}" ಪದವನ್ನು ವರ್ಣಗಳಾಗಿ ಬಿಡಿಸಿ ತಿಳಿಸಿ?',
]

TEMPLATES = TEMPLATES_SPELLING + TEMPLATES_LISTING + TEMPLATES_VARNAVICHCHEDA


def get_kannada_characters(word: str) -> list[str]:
    """
    Break down a Kannada word into its constituent Unicode characters.
    Each Unicode character (consonant, vowel sign, etc.) is separate.
    Used for: Spelling questions (S1, S8)
    """
    return list(word)


def get_kannada_grapheme_clusters(word: str) -> list[str]:
    """
    Get aksharas (syllabic units) for Kannada word.
    Per Kannada linguistics: Ottakshara/conjuncts = 1 unit, Anusvara = part of preceding.
    Used for: Counting, length, position, spelling (S1-S4, S7, S9, S10).
    """
    return get_kannada_aksharas(word)


def get_swara_vyanjana_lists(word: str) -> tuple[list[str], list[str]]:
    """
    Decompose word into ವ್ಯಂಜನಗಳು (consonant+halant) and ಸ್ವರಗಳು (vowels) at varṇa level.
    E.g. ಕತ್ತಲು → ವ್ಯಂಜನಗಳು: ಕ್, ತ್, ತ್, ಲ್. ಸ್ವರಗಳು: ಅ, ಅ, ಉ.
    """
    vyanjanas: list[str] = []
    swaras: list[str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        c = chars[i]
        if 0x0C95 <= ord(c) <= 0x0CB9:  # Consonant
            if i + 1 < len(chars) and chars[i + 1] == HALANT:
                vyanjanas.append(c + HALANT)
                i += 2
            else:
                vyanjanas.append(c + HALANT)
                i += 1
                if i < len(chars) and chars[i] in VOWEL_SIGN_TO_SWARA:
                    swaras.append(VOWEL_SIGN_TO_SWARA[chars[i]])
                    i += 1
                else:
                    swaras.append("ಅ")
        elif 0x0C85 <= ord(c) <= 0x0C94:  # Independent vowel
            swaras.append(c)
            i += 1
        elif c in VOWEL_SIGN_TO_SWARA:
            swaras.append(VOWEL_SIGN_TO_SWARA[c])
            i += 1
        elif c in ("\u0c82", "\u0c83"):  # Anusvara, visarga - skip or add
            i += 1
        else:
            i += 1
    return (vyanjanas, swaras)


def get_varnavichcheda_str(word: str) -> str:
    """
    ವರ್ಣವಿಚ್ಛೇದ: decompose word into varṇas (consonant+halant, vowel) in order.
    E.g. ಬೆರಳು → ಬ್ + ಎ + ರ್ + ಅ + ಳ್ + ಉ
    """
    varnas: list[str] = []
    chars = list(word)
    i = 0
    while i < len(chars):
        c = chars[i]
        if 0x0C95 <= ord(c) <= 0x0CB9:  # Consonant
            if i + 1 < len(chars) and chars[i + 1] == HALANT:
                varnas.append(c + HALANT)
                i += 2
            else:
                varnas.append(c + HALANT)
                i += 1
                if i < len(chars) and chars[i] in VOWEL_SIGN_TO_SWARA:
                    varnas.append(VOWEL_SIGN_TO_SWARA[chars[i]])
                    i += 1
                elif (
                    i < len(chars) and chars[i] == "\u0c82"
                ):  # Anusvara after consonant → ಅಂ
                    varnas.append("ಅಂ")
                    i += 1
                else:
                    varnas.append("ಅ")
        elif 0x0C85 <= ord(c) <= 0x0C94:  # Independent vowel
            varnas.append(c)
            i += 1
        elif c in VOWEL_SIGN_TO_SWARA:
            varnas.append(VOWEL_SIGN_TO_SWARA[c])
            i += 1
        elif c == "\u0c82":  # Anusvara after vowel (e.g. ಅಂ in ಅಂಚು) → ಅಂ
            varnas.append("ಅಂ")
            i += 1
        elif c == "\u0c83":  # Visarga (ಃ) - include in ವರ್ಣವಿಚ್ಛೇದ
            varnas.append("ಃ")
            i += 1
        else:
            i += 1
    return " + ".join(varnas)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as hyphen-separated aksharas (e.g. ಪು-ಸ್ತ-ಕ)"""
    aksharas = get_kannada_aksharas(word)
    return "-".join(aksharas)


def generate_listing_answer(word: str, template: str) -> str:
    """Generate listing answer based on specific template rules."""
    clusters = get_kannada_aksharas(word)

    if (
        "ಕ್ರಮಾನುಗತವಾಗಿ ನೀಡಿ" in template
        or "ಕ್ರಮವಾಗಿ ತೋರಿಸಿ" in template
        or "ಪ್ರತ್ಯೇಕವಾಗಿ ತಿಳಿಸಿ" in template
        or "ಅನುಕ್ರಮವಾಗಿ ಬರೆಯಿರಿ" in template
    ):
        return ", ".join(clusters)
    elif "ಅಕ್ಷರ ಘಟಕಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template:
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ವಿಭಜಿಸಿ ಬರೆಯಿರಿ" in template:
        return get_varnavichcheda_str(word)
    elif "ಅಕ್ಷರಗಳನ್ನು ಬಿಡಿಸಿ ಪಟ್ಟಿ ರೂಪದಲ್ಲಿ ನೀಡಿ" in template:
        # For words like "ಹೃದಯ" -> "ಹೈ, ದ, ಯ" - this might require custom logic or simplified grapheme listing
        # For now, general grapheme listing
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾನುಗತವಾಗಿ ನೀಡಿ" in template:
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳು ಯಾವುವು" in template:
        if len(clusters) == 2:  # For "ಬಾನು" -> "ಬಾ ಮತ್ತು ನು"
            return f"{clusters[0]} ಮತ್ತು {clusters[1]}"
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾಂಕದಲ್ಲಿ ನೀಡಿ" in template:  # For "ಅಮ್ಮ" -> "೧: ಅ, ೨: ಮ್ಮ"
        return ", ".join([f"{i+1}: {c}" for i, c in enumerate(clusters)])
    elif "ಸ್ವರ ಮತ್ತು ವ್ಯಂಜನಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template:
        vyanjanas, swaras = get_swara_vyanjana_lists(word)
        return f"ವ್ಯಂಜನಗಳು: {', '.join(vyanjanas)}. ಸ್ವರಗಳು: {', '.join(swaras)}"
    elif "ಎಲ್ಲಾ ಸ್ವರಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template:
        _, swaras = get_swara_vyanjana_lists(word)
        return ", ".join(swaras)
    elif "ವರ್ಣವಿಚ್ಛೇದ" in template or "ವರ್ಣಗಳಾಗಿ" in template:
        return get_varnavichcheda_str(word)
    else:
        return ", ".join(clusters)


if __name__ == "__main__":
    all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
    samples = []
    target_count = 28600

    # Generate all unique combinations first
    unique_combinations = {}
    for word in set(all_words):
        for template_idx, template in enumerate(TEMPLATES):
            query = template.format(word=word)
            if template in TEMPLATES_SPELLING:
                answer = (
                    get_varnavichcheda_str(word)
                    if "ಅಕ್ಷರಶಃ ಬಿಡಿಸಿ" in template
                    else generate_spelling_answer(word)
                )
            else:  # TEMPLATES_LISTING or TEMPLATES_VARNAVICHCHEDA
                answer = generate_listing_answer(word, template)
            unique_combinations[(word, template_idx)] = (query, answer)

    # If we have enough unique combinations, use them
    if len(unique_combinations) >= target_count:
        samples = list(unique_combinations.values())[:target_count]
    else:
        # Use all unique combinations, then add more unique pairs only (no duplicates)
        samples = list(unique_combinations.values())
        seen_qa = set((q, a) for q, a in samples)
        no_progress_limit = 50000
        no_progress = 0
        while len(samples) < target_count and no_progress < no_progress_limit:
            word = random.choice(list(set(all_words)))
            template_idx = random.randint(0, len(TEMPLATES) - 1)
            template = TEMPLATES[template_idx]
            query = template.format(word=word)
            if template in TEMPLATES_SPELLING:
                answer = (
                    get_varnavichcheda_str(word)
                    if "ಅಕ್ಷರಶಃ ಬಿಡಿಸಿ" in template
                    else generate_spelling_answer(word)
                )
            else:
                answer = generate_listing_answer(word, template)
            if (query, answer) not in seen_qa:
                seen_qa.add((query, answer))
                samples.append((query, answer))
                no_progress = 0
            else:
                no_progress += 1

    # Shuffle for randomness
    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_kannada(query, answer) + "\n")

    print(f"S1 Spelling (Kannada): Generated {len(samples)} samples")
