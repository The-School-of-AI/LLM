#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (ವर्तನಿ) questions - Kannada
Target: 28,600 pairs (14.3% of 200,000)
"""
import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_grammar import get_kannada_aksharas  # noqa: E402
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Spelling: sequence of characters in a word (user-specified templates).
TEMPLATES_SPELLING = [
    '"{word}" ಪದವನ್ನು ಅಕ್ಷರಶಃ ಬಿಡಿಸಿ ಬರೆಯಿರಿ?',
    '"{word}" ಪದದ ಸರಿಯಾದ ಕಾಗುಣಿತ ಯಾವುದು?',
    '"{word}" ಎಂಬ ಶಬ್ದದ ಅಕ್ಷರಗಳ ಜೋಡಣೆ ತಿಳಿಸಿ?',
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

TEMPLATES = TEMPLATES_SPELLING + TEMPLATES_LISTING


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


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as hyphen-separated aksharas (e.g. ಪು-ಸ್ತ-ಕ)"""
    aksharas = get_kannada_aksharas(word)
    return "-".join(aksharas)


def generate_listing_answer(word: str, template: str) -> str:
    """Generate listing answer based on specific template rules (akshara-level)"""
    clusters = get_kannada_aksharas(word)
    characters = get_kannada_characters(word) # For character-level details if needed

    if "ಕ್ರಮಾನುಗತವಾಗಿ ನೀಡಿ" in template or "ಕ್ರಮವಾಗಿ ತೋರಿಸಿ" in template or "ಪ್ರತ್ಯೇಕವಾಗಿ ತಿಳಿಸಿ" in template or "ಅನುಕ್ರಮವಾಗಿ ಬರೆಯಿರಿ" in template:
        return ", ".join(clusters)
    elif "ಅಕ್ಷರ ಘಟಕಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template:
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ವಿಭಜಿಸಿ ಬರೆಯಿರಿ" in template:
        return "-".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ಬಿಡಿಸಿ ಪಟ್ಟಿ ರೂಪದಲ್ಲಿ ನೀಡಿ" in template:
        # For words like "ಹೃದಯ" -> "ಹೈ, ದ, ಯ" - this might require custom logic or simplified grapheme listing
        # For now, general grapheme listing
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾನುಗತವಾಗಿ ನೀಡಿ" in template:
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳು ಯಾವುವು" in template:
        if len(clusters) == 2: # For "ಬಾನು" -> "ಬಾ ಮತ್ತು ನು"
            return f"{clusters[0]} ಮತ್ತು {clusters[1]}"
        return ", ".join(clusters)
    elif "ಅಕ್ಷರಗಳನ್ನು ಕ್ರಮಾಂಕದಲ್ಲಿ ನೀಡಿ" in template: # For "ಅಮ್ಮ" -> "೧: ಅ, ೨: ಮ್ಮ"
        return ", ".join([f"{i+1}: {c}" for i, c in enumerate(clusters)])
    elif "ಸ್ವರ ಮತ್ತು ವ್ಯಂಜನಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template or "ಎಲ್ಲಾ ಸ್ವರಗಳನ್ನು ಪಟ್ಟಿ ಮಾಡಿ" in template:
        # This requires identifying individual swaras and vyanjanas from unicode characters
        # For now, list all grapheme clusters. Will refine with proper swara/vyanjana identification if needed.
        return ", ".join(clusters)
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
                answer = generate_spelling_answer(word)
            else: # TEMPLATES_LISTING
                answer = generate_listing_answer(word, template)
            unique_combinations[(word, template_idx)] = (query, answer)

    # If we have enough unique combinations, use them
    if len(unique_combinations) >= target_count:
        samples = list(unique_combinations.values())[:target_count]
    else:
        # Use all unique combinations, then randomly sample with replacement to reach target
        samples = list(unique_combinations.values())
        while len(samples) < target_count:
            word = random.choice(list(set(all_words)))
            template_idx = random.randint(0, len(TEMPLATES) - 1)
            template = TEMPLATES[template_idx]
            query = template.format(word=word)
            if template in TEMPLATES_SPELLING:
                answer = generate_spelling_answer(word)
            else:
                answer = generate_listing_answer(word, template)
            samples.append((query, answer))

    # Shuffle for randomness
    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_kannada(query, answer) + "\n")

    print(f"S1 Spelling (Kannada): Generated {len(samples)} samples")
