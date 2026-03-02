#!/usr/bin/env python3
"""
Generate Statement 11: Ottakshara & Kagunita (ಒತ್ತಕ್ಷರ, ಗುಣಿತಾಕ್ಷರ) - Kannada
Language-specific: conjuncts, vowel signs, ಋ/ಐ/ಔ, ಷ vs ಶ, ಕ್ಷ as consonant.
Answers for "ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ" list ALL ottakshara in the word (e.g. ದೃಶ್ಯ → ದೃ ಮತ್ತು ಶ್ಯ).
"""

import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_grammar import get_kannada_aksharas  # noqa: E402
from group1_kannada.kannada_vocabulary import ALL_WORDS_UNIQUE, VARGAS  # noqa: E402
from prompt_utils import format_qa_pair_kannada, int_to_kannada  # noqa: E402

# Halant and vowel signs that make an akshara a conjunct (ottakshara)
HALANT = "\u0ccd"
R_VOWEL, RR_VOWEL = "\u0cc3", "\u0cc4"  # ೃ, ೄ
VIRAMA = HALANT
RA = "\u0cb0"

# Vowel signs (Matras) mapping - for matra identification questions
MATRA_MAP = {
    "\u0be7": "ಆ-ಕಾರ",  # ಾ
    "\u0be8": "ಇ-ಕಾರ",  # ಿ
    "\u0be9": "ಈ-ಕಾರ",  # ೀ
    "\u0bea": "ಉ-ಕಾರ",  # ು
    "\u0beb": "ಊ-ಕಾರ",  # ೂ
    "\u0bec": "ಋ-ಕಾರ",  # ೃ
    "\u0bed": "ಎ-ತ್ವ",  # ೆ
    "\u0bee": "ಏ-ತ್ವ",  # ೇ
    "\u0bef": "ಐ-ಕಾರ",  # ೈ
    "\u0bf0": "ಓ-ತ್ವ",  # ೊ
    "\u0bf1": "ಓ-ತ್ವ",  # ೋ
    "\u0bf2": "ಔ-ತ್ವ",  # ೌ
    "\u0c01": "ಅಕ",  # ಁ
    "\u0c02": "ಅನುಸ್ವಾರ",  # ಂ
    "\u0c03": "ವಿಸರ್ಗ",  # ಃ
}

ORDINALS = ["", "ಮೊದಲ", "ಎರಡನೇ", "ಮೂರನೇ", "ನಾಲ್ಕನೇ", "ಐದನೇ"]

# Question templates for rephrasing (from colleague f3a5869)
OTTAKSHARA_TEMPLATES = [
    '"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರವನ್ನು ಗುರುತಿಸಿ.',
    '"{word}" ಪದದಲ್ಲಿ ಯಾವ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆ?',
    '"{word}" ಪದದಲ್ಲಿ ಕಂಡುಬರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?',
]

SAJATIYA_TEMPLATES = [
    '"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರವು ಸಜಾತೀಯವೇ ಅಥವಾ ವಿಜಾತೀಯವೇ?',
    '"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವ ವಿಧವಾಗಿದೆ (ಸಜಾತೀಯ/ವಿಜಾತೀಯ)?',
    '"{word}" ಪದದ ಸಂಯುಕ್ತಾಕ್ಷರವನ್ನು ವಿಂಗಡಿಸಿ (ಸಜಾತೀಯ/ವಿಜಾತೀಯ).',
]

YES_NO_TEMPLATES = [
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರ ಇದೆಯೇ?', "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆಯೇ?', "ಹೌದು"),
    ('"{word}" ಪದವು ಒತ್ತಕ್ಷರರಹಿತವೇ?', "ಇಲ್ಲ"),
]


def get_all_ottakshara_in_word(word: str) -> list[str]:
    """Return all aksharas in word that are conjuncts (contain ್ or ೃ/ೄ)."""
    clusters = get_kannada_grapheme_clusters(word)
    return [c for c in clusters if HALANT in c or R_VOWEL in c or RR_VOWEL in c]


def format_ottakshara_answer(ottakshara_list: list[str]) -> str:
    """Format as 'ದೃ ಮತ್ತು ಶ್ಯ' or 'ದೃ, ಶ್ಯ ಮತ್ತು ಕ್ಷ'."""
    if not ottakshara_list:
        return ""
    if len(ottakshara_list) == 1:
        return ottakshara_list[0]
    if len(ottakshara_list) == 2:
        return f"{ottakshara_list[0]} ಮತ್ತು {ottakshara_list[1]}"
    return f"{', '.join(ottakshara_list[:-1])} ಮತ್ತು {ottakshara_list[-1]}"


def get_ottakshara_component(akshara: str) -> str | None:
    """Extract full Ottakshara (syllable with conjunct) from Akshara."""
    if regex.search(r"[\u0C95-\u0CB9]\u0CCD[\u0C95-\u0CB9]", akshara):
        return akshara
    return None


def is_sajatiya(akshara: str) -> bool:
    """
    Check if akshara is Sajatiya (same consonant conjunct).
    E.g. 'ಪ್ಪ' -> True, 'ಕ್ತ' -> False.
    """
    consonants = regex.findall(r"[\u0C95-\u0CB9]", akshara)
    if len(consonants) >= 2:
        return consonants[0] == consonants[1]
    return False


def analyze_word_for_questions(word: str) -> list[tuple[str, str]]:
    """Generate QA pairs for a single word (vocabulary-based, from colleague f3a5869)."""
    questions = []
    aksharas = get_kannada_aksharas(word)
    clusters = get_kannada_grapheme_clusters(word)
    has_ottakshara = False

    # 1. Ottakshara Identification & Classification
    for ak in aksharas:
        ottakshara = get_ottakshara_component(ak)
        if ottakshara:
            has_ottakshara = True
            template = random.choice(OTTAKSHARA_TEMPLATES)
            questions.append((template.format(word=word), ottakshara))
            classification = "ಸಜಾತೀಯ" if is_sajatiya(ak) else "ವಿಜಾತೀಯ"
            template_cls = random.choice(SAJATIYA_TEMPLATES)
            questions.append((template_cls.format(word=word), classification))

    # 2. Yes/No Questions (Has Ottakshara?)
    if has_ottakshara:
        q_tmpl, ans = random.choice(YES_NO_TEMPLATES)
        questions.append((q_tmpl.format(word=word), ans))

    # 3. Matra Identification
    for cluster in clusters:
        for char in cluster:
            if char in MATRA_MAP:
                matra_name = MATRA_MAP[char]
                q_variations = [
                    f'"{word}" ಪದದಲ್ಲಿರುವ {matra_name} ಯಾವುದು?',
                    f'"{word}" ಪದದಲ್ಲಿ {matra_name} ಇದೆಯೇ? ಹೌದು, ಅದು "{char}".',
                ]
                questions.append((random.choice(q_variations), char))

    # 4. Arkavattu (Repha)
    if regex.search(rf"{RA}{VIRAMA}([\u0C95-\u0CB9])", word):
        for cluster in clusters:
            if RA + VIRAMA in cluster:
                questions.append(
                    (
                        f'"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು ಚಿಹ್ನೆಯನ್ನು ಗುರುತಿಸಿ?',
                        cluster,
                    )
                )
                break

    # 5. Independent Letter Count (Akshara count)
    count = len(aksharas)
    questions.append(
        (
            f'"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವತಂತ್ರ ಅಕ್ಷರಗಳಿವೆ?',
            int_to_kannada(count),
        )
    )

    return questions


def generate_varga_questions() -> list[tuple[str, str]]:
    """Generate questions about Vargas (varga, ordinal, nasal)."""
    qs = []
    for varga_name, letters in VARGAS.items():
        for idx, letter in enumerate(letters):
            ordinal = ORDINALS[idx + 1]
            qs.append((f'"{varga_name}" ವರ್ಗದ {ordinal} ಅಕ್ಷರ ಯಾವುದು?', letter))
            qs.append((f'"{letter}" ಅಕ್ಷರವು ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ?', f'"{varga_name}" ವರ್ಗ'))
            if idx == 4:
                qs.append((f'"{varga_name}" ವರ್ಗದ ಅನುನಾಸಿಕ ಅಕ್ಷರ ಯಾವುದು?', letter))
    return qs


# User-specified Ottakshara/Kagunita (word, question_template, answer).
# For "ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ" / "ಒತ್ತಕ್ಷರ ಯಾವುದು?" use words that contain conjuncts; answer lists all.
OTTAKSHARA_QA = [
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?', "ಅಪ್ಪ", "ಪ್ಪ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಋ-ಕಾರ ಗುರುತಿಸಿ?', "ಕೃಷಿ", "ಋ"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆ?', "ಜ್ಞಾನ", "ಜ್ಞ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಓ-ತ್ವ ಚಿಹ್ನೆ ಯಾವುದು?', "ಕೋಳಿ", "ೋ"),
    (
        '"{letter}" ಅಕ್ಷರದ ಗುಣಿತಾಕ್ಷರಗಳನ್ನು ಬರೆಯಿರಿ?',
        "ಕ",
        "ಕ, ಕಾ, ಕಿ, ಕೀ, ಕು, ಕೂ, ಕೃ, ಕೆ, ಕೇ, ಕೈ, ಕೊ, ಕೋ, ಕೌ",
    ),
    ('"ಷ" ಮತ್ತು "ಶ" ಅಕ್ಷರಗಳ ವ್ಯತ್ಯಾಸವೇನು?', None, "ಅವು ವಿಭಿನ್ನ ವ್ಯಂಜನಾಕ್ಷರಗಳು"),
    # ದೃಶ್ಯ has two ottakshara: ದೃ and ಶ್ಯ — answer lists both
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ?',
        "ದೃಶ್ಯ",
        None,
    ),  # answer computed below
    ('"{word}" ಪದದಲ್ಲಿರುವ ಐ-ಕಾರ ಯಾವುದು?', "ಐದು", "ೈ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಔ-ತ್ವ ಚಿಹ್ನೆಯನ್ನು ತೋರಿಸಿ?', "ಸೌರ", "ೌ"),
    ('"{char}" ಅಕ್ಷರವು ಸ್ವರವೋ ಅಥವಾ ವ್ಯಂಜನವೋ?', "ಕ್ಷ", "ವ್ಯಂಜನ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?', "ಲಕ್ಷ್ಮಿ", "ಕ್ಷ್ಮಿ"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಎ-ತ್ವ ಚಿಹ್ನೆ ಇದೆ?', "ಕೇಸರಿ", "ೇ"),
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು ಚಿಹ್ನೆಯನ್ನು ಗುರುತಿಸಿ?',
        "ಸೂರ್ಯ",
        "೯",
    ),  # ೯ = ಅರ್ಕಾವತ್ತು (ರೆಫೆ) ಚಿಹ್ನೆ
    ('"{letter}" ವರ್ಗದ ಐದನೇ ಅಕ್ಷರ (ಅನುನಾಸಿಕ) ಯಾವುದು?', "ಕ", "ಙ"),
    ('"{letter}" ಅಕ್ಷರದ ಒತ್ತಕ್ಷರ ಹೇಗೆ ಬರೆಯುವುದು?', "ತ", "ತ್ತ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸ್ವರಾಕ್ಷರ ಯಾವುದು?', "ಋಷಿ", "ಋ"),
    ('"{word}" ಪದದಲ್ಲಿ ಗುಣಿತಾಕ್ಷರಗಳು ಇವೆಯೇ?', "ಗಗನ", "ಇಲ್ಲ"),
    ('"{word}" ಪದದ ಮೊದಲ ಅಕ್ಷರ ಯಾವುದು?', "ಔಷಧ", "ಔ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ದ-ಕಾರದ ಒತ್ತಕ್ಷರ ಗುರುತಿಸಿ?', "ವಿದ್ಯೆ", "ದ್ಯೆ"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವತಂತ್ರ ಅಕ್ಷರಗಳಿವೆ?', "ಬಾಲಕ", "೩"),
    # Templates 21-40 (exact from user CSV)
    ('"{word}" ಪದವು ಸಜಾತೀಯವೇ ಅಥವಾ ವಿಜಾತೀಯ ಒತ್ತಕ್ಷರವೇ?', "ಅಮ್ಮ", "ಸಜಾತೀಯ ಒತ್ತಕ್ಷರ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ವಿಜಾತೀಯ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?', "ಪುಸ್ತಕ", "ಸ್ತ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು (ರೆಫೆ) ಯಾವ ಅಕ್ಷರಕ್ಕೆ ಸೇರಿದೆ?', "ಧರ್ಮ", "ಮ"),
    ('"{word}" ಪದವನ್ನು ಅಕ್ಷರಗಳಾಗಿ ಬಿಡಿಸಿ ಬರೆಯಿರಿ?', "ಶಾಲೆ", "ಶ್ + ಆ + ಲ + ಎ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಯೋಗವಾಹ ಯಾವುದು?', "ಸಿಂಹ", "ಅನುಸ್ವಾರ ( ಂ )"),
    ('"{word}" ಪದದಲ್ಲಿ ವಿಸರ್ಗ ಚಿಹ್ನೆ ಇದೆಯೇ?', "ದುಃಖ", "ಹೌದು ( ಃ )"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವು ಯಾವ ವ್ಯಂಜನಕ್ಕೆ ಸೇರಿದೆ?', "ಅಕ್ಕ", "ಕ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ದೀರ್ಘ ಸ್ವರವನ್ನು ಗುರುತಿಸಿ?', "ಆಕಾಶ", "ಆ"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', "ಮೈಸೂರು", "೩"),
    ("\"{word}\" ಪದದಲ್ಲಿ 'ರ' ಅಕ್ಷರದ ಒತ್ತಕ್ಷರ ರೂಪ (ಅರ್ಕಾವತ್ತು) ಇದೆಯೇ?", "ಕರ್ಣ", "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ವ್ಯಂಜನಕ್ಕೆ ಯ ಒತ್ತಕ್ಷರ ಬಂದಿದೆ?', "ಸೂರ್ಯ", "ರ್"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವು ಪೂರ್ಣ ರೂಪ ಬದಲಿಸುವ ಒತ್ತಕ್ಷರವೇ?', "ಅಣ್ಣ", "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ತ ಕಾರದ ಒತ್ತಕ್ಷರವನ್ನು ಗುರುತಿಸಿ?', "ಕತ್ತಿ", "ತ್ತಿ"),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಒತ್ತಕ್ಷರಗಳಿವೆ?', "ನಕ್ಷತ್ರ", "೨"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸಜಾತೀಯ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?', "ಸಕ್ಕರೆ", "ಕ್ಕ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ವಿಜಾತೀಯ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?', "ಚಂದ್ರ", "ಂದ್ರ"),
    ('"{word}" ಪದದಲ್ಲಿ ರ ವ್ಯಂಜನವು ಒತ್ತಕ್ಷರವಾಗಿ ಬಂದಿದೆಯೇ?', "ಪ್ರೇಮ", "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿ ಅರ್ಕಾವತ್ತು ಯಾವ ಅಕ್ಷರದ ಮೊದಲು ಉಚ್ಚಾರವಾಗುತ್ತದೆ?', "ತರ್ಕ", "ಕ"),
    ('"{word}" ಪದದಲ್ಲಿ ಲ ವ್ಯಂಜನಕ್ಕೆ ಯಾವ ಒತ್ತಕ್ಷರ ಸೇರಿದೆ?', "ಕಲ್ಪನೆ", "ಪ-ವತ್ತು"),
    ('"{word}" ಪದದಲ್ಲಿ ಮ ಕಾರಕ್ಕೆ ಮ ಒತ್ತೇ ಬಂದಿದೆಯೇ?', "ನಮ್ಮ", "ಹೌದು"),
    # Templates 41-60
    ('"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರದಲ್ಲಿ ಎಷ್ಟು ವ್ಯಂಜನಗಳಿವೆ?', "ರಾಷ್ಟ್ರ", "೩"),
    ('"{word}" ಪದದಲ್ಲಿ ಷ ಕಾರಕ್ಕೆ ಯಾವ ಒತ್ತಕ್ಷರ ನೀಡಲಾಗಿದೆ?', "ಕೃಷ್ಣ", "ಣ-ವತ್ತು"),
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರವಿಲ್ಲದ ಅಕ್ಷರ ಯಾವುದು?', "ಅಪ್ಪಟ", "ಅ, ಟ"),
    ('"{letter}" ಅಕ್ಷರಕ್ಕೆ ನ ಒತ್ತು ಸೇರಿಸಿ ಬರೆಯಿರಿ?', "ರ", "ರ್ನ"),
    ('"{word}" ಪದದಲ್ಲಿ ವ ಒತ್ತಕ್ಷರ ಯಾವ ಅಕ್ಷರದಲ್ಲಿದೆ?', "ತತ್ವ", "ತ್ವ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವು ತಲೆಕಟ್ಟು ಇಲ್ಲದ ರೂಪವೇ?', "ಅಕ್ಕ", "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿ ಜ ಕಾರಕ್ಕೆ ಯಾವ ಒತ್ತಕ್ಷರ ಬಂದಿದೆ?', "ವಿಜ್ಞಾನ", "ಞ-ವತ್ತು"),
    ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರವು ಪದದ ಯಾವ ಭಾಗದಲ್ಲಿದೆ?', "ಕಪ್ಪೆ", "ಮಧ್ಯ"),
    ('"{word}" ಪದದಲ್ಲಿ ದ ವ್ಯಂಜನಕ್ಕೆ ಯಾವ ಸ್ವರ ಸೇರಿ ಒತ್ತಕ್ಷರವಾಗಿದೆ?', "ವಿದ್ಯೆ", "ಎ"),
    ('"{word}" ಪದದಲ್ಲಿ ನ ಮತ್ತು ಯ ಒತ್ತುಗಳು ಒಟ್ಟಿಗೆ ಬಂದಿವೆಯೇ?', "ನ್ಯೂನತೆ", "ಹೌದು"),
    ('"{word}" ಪದದಲ್ಲಿ ರ ಅಕ್ಷರದ ಒತ್ತಕ್ಷರ ರೂಪ ಯಾವುದು?', "ಪ್ರಗತಿ", "ಕ್ರ-ವತ್ತು"),
    (
        '"{word}" ಪದದಲ್ಲಿ ಯಾವ ವರ್ಗದ ವ್ಯಂಜನಕ್ಕೆ ಅದೇ ವರ್ಗದ ಒತ್ತು ಬಂದಿದೆ?',
        "ಬೊಟ್ಟು",
        "ಟ-ವರ್ಗ",
    ),
    ('"{word}" ಪದದಲ್ಲಿ ಹ ಕಾರದ ಒತ್ತಕ್ಷರ ಎಲ್ಲಿದೆ?', "ಅರ್ಹತೆ", "ರ್ಹ"),
    ('"{word}" ಪದದಲ್ಲಿ ಚ ಕಾರಕ್ಕೆ ಛ ಒತ್ತಕ್ಷರ ಬಂದಿದೆಯೇ?', "ಲಾಂಛನ", "ಹೌದು"),
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ವ್ಯಂಜನ ಮತ್ತು ಒತ್ತಕ್ಷರವನ್ನು ಬೇರ್ಪಡಿಸಿ?',
        "ಸ್ತೋತ್ರ",
        "ಸ್+ತ, ತ್+ರ",
    ),
    ('"{word}" ಪದದಲ್ಲಿ ಕ ಕಾರಕ್ಕೆ ಷ ಒತ್ತು ಸೇರಿದರೆ ಏನಾಗುತ್ತದೆ?', "ಅಕ್ಷರ", "ಕ್ಷ"),
    ('"{word}" ಪದದಲ್ಲಿ ಸ ಕಾರಕ್ಕೆ ತ ಒತ್ತು ಸೇರಿದ ರೂಪ ಯಾವುದು?', "ರಸ್ತೆ", "ಸ್ತೆ"),
    ('"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವು ಮಹಾಪ್ರಾಣವೇ?', "ಬುದ್ಧಿ", "ಹೌದು (ದ್ಧ)"),
    ('"{word}" ಪದದಲ್ಲಿ ಯಾವ ಅಕ್ಷರಕ್ಕೆ ಒತ್ತಕ್ಷರ ನೀಡಬಾರದು?', "ಆನೆ", "ಯಾವುದಕ್ಕೂ ಇಲ್ಲ"),
    ('"{word}" ಪದದಲ್ಲಿ ಲ ವ್ಯಂಜನಕ್ಕೆ ಸಜಾತೀಯ ಒತ್ತು ಸೇರಿದೆಯೇ?', "ಬೆಲ್ಲ", "ಹೌದು"),
]

# Words with ottakshara (conjuncts) from vocabulary - for multi-word expansion
OTTAKSHARA_WORDS = [
    w for w in ALL_WORDS_UNIQUE if w and (HALANT in w or R_VOWEL in w or RR_VOWEL in w)
]


# Multi-word templates: (template_str, filter_fn, answer_fn)
# filter_fn(word) -> bool; answer_fn(word) -> str or None to skip
def _first_ottakshara(w):
    ott = get_all_ottakshara_in_word(w)
    return ott[0] if ott else None


def _all_ottakshara(w):
    ott = get_all_ottakshara_in_word(w)
    return format_ottakshara_answer(ott) if ott else None


def _cluster_count(w):
    return int_to_kannada(len(get_kannada_grapheme_clusters(w)))


def _ottakshara_count(w):
    return int_to_kannada(len(get_all_ottakshara_in_word(w)))


MULTI_WORD_TEMPLATES = [
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?',
        lambda w: bool(get_all_ottakshara_in_word(w)),
        _first_ottakshara,
    ),
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ?',
        lambda w: len(get_all_ottakshara_in_word(w)) >= 1,
        _all_ottakshara,
    ),
    (
        '"{word}" ಪದದಲ್ಲಿ ಯಾವ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆ?',
        lambda w: HALANT in w,
        _first_ottakshara,
    ),
    (
        '"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?',
        lambda w: HALANT in w,
        _first_ottakshara,
    ),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವತಂತ್ರ ಅಕ್ಷರಗಳಿವೆ?', lambda w: True, _cluster_count),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?', lambda w: True, _cluster_count),
    ('"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಒತ್ತಕ್ಷರಗಳಿವೆ?', lambda w: True, _ottakshara_count),
]

# Curated set: only unique QA pairs, no repeats
samples = []
seen_qa = set()

for template, word_or_letter, answer in OTTAKSHARA_QA:
    if "{word}" in template and word_or_letter:
        q = template.format(word=word_or_letter)
        # "ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ" → answer = all ottakshara in word (e.g. ದೃಶ್ಯ → ದೃ ಮತ್ತು ಶ್ಯ)
        if answer is None and "ಒತ್ತಕ್ಷರವನ್ನು ಹೆಸರಿಸಿ" in template:
            ott = get_all_ottakshara_in_word(word_or_letter)
            answer = format_ottakshara_answer(ott) if ott else ""
    elif "{letter}" in template and word_or_letter:
        q = template.format(letter=word_or_letter)
    elif "{char}" in template and word_or_letter:
        q = template.format(char=word_or_letter)
    else:
        q = template  # literal (e.g. ಷ and ಶ)
    if answer is not None:
        key = (q, answer)
        if key not in seen_qa:
            seen_qa.add(key)
            samples.append((q, answer))

# Multi-word expansion: generate from vocabulary for templates that support it
for template_str, filter_fn, answer_fn in MULTI_WORD_TEMPLATES:
    # Use all words for count templates; ottakshara words for identification templates
    word_pool = (
        OTTAKSHARA_WORDS if "ಎಷ್ಟು" not in template_str else list(ALL_WORDS_UNIQUE)
    )
    for word in word_pool:
        if not word or not filter_fn(word):
            continue
        ans = answer_fn(word)
        if ans is None or ans == "":
            continue
        q = template_str.format(word=word)
        key = (q, ans)
        if key not in seen_qa:
            seen_qa.add(key)
            samples.append((q, ans))

# Colleague's vocabulary-based generation (f3a5869, 2054b43): rephrasing, Sajatiya/Vijatiya, yes/no, matra
for word in ALL_WORDS_UNIQUE:
    if not word:
        continue
    for q, a in analyze_word_for_questions(word):
        key = (q, a)
        if key not in seen_qa:
            seen_qa.add(key)
            samples.append((q, a))

# Negation: words WITHOUT ottakshara -> "ಒತ್ತಕ್ಷರ ಇದೆಯೇ?" -> "ಇಲ್ಲ"
simple_words = [
    w
    for w in ALL_WORDS_UNIQUE
    if w and not any(get_ottakshara_component(ak) for ak in get_kannada_aksharas(w))
]
for word in simple_words[:1000]:  # Cap to avoid imbalance
    for q_tmpl, ans in [
        ('"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರ ಇದೆಯೇ?', "ಇಲ್ಲ"),
        ('"{word}" ಪದದಲ್ಲಿ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆಯೇ?', "ಇಲ್ಲ"),
        ('"{word}" ಪದವು ಒತ್ತಕ್ಷರರಹಿತವೇ?', "ಹೌದು"),
    ]:
        q = q_tmpl.format(word=word)
        key = (q, ans)
        if key not in seen_qa:
            seen_qa.add(key)
            samples.append((q, ans))

# Varga-based questions
for q, a in generate_varga_questions():
    key = (q, a)
    if key not in seen_qa:
        seen_qa.add(key)
        samples.append((q, a))

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s11.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S11 Ottakshara/Kagunita (Kannada): Generated {len(samples)} samples")
