#!/usr/bin/env python3
"""
Kannada grammar helpers for dataset generation.
Genitive (ಷಷ್ಠಿ) suffix: ನ, ಯ, or ದ depends on the word ending.
Numerals take ರ (e.g. 72 ರ, 17 ರ).
Akshara segmentation: syllabic units per Kannada linguistics (Ottakshara, Anusvara).
"""

import regex

# Virama (halant): consonant+್ forms half-form; next consonant attaches (conjunct)
_KANNADA_VIRAMA = "\u0ccd"


def _ends_with_virama(s: str) -> bool:
    """True if string ends with virama (halant), indicating an incomplete akshara."""
    return bool(s and s[-1] == _KANNADA_VIRAMA)


def get_kannada_aksharas(word: str) -> list[str]:
    """
    Segment Kannada word into aksharas (syllabic units).
    Per Kannada linguistics: Ottakshara (conjuncts like ಸ್ಪ, ತ್ರೆ) = 1 unit;
    Anusvara (ಂ) = part of preceding letter.
    Uses grapheme clusters + virama merging.
    E.g. ಆಸ್ಪತ್ರೆ -> ['ಆ', 'ಸ್ಪ', 'ತ್ರೆ'] (3 aksharas)
    """
    if not word:
        return []
    clusters = regex.findall(r"\X", word)
    aksharas = []
    i = 0
    while i < len(clusters):
        akshara = clusters[i]
        while i + 1 < len(clusters) and _ends_with_virama(akshara):
            i += 1
            akshara += clusters[i]
        aksharas.append(akshara)
        i += 1
    return aksharas


# Vowel signs that take ಯ (nouns ending in i, ī, e, ē)
_SUFFIX_Y = ("ಿ", "ೀ", "ೆ", "ೇ")  # i, ī, e, ē

# Vowel signs that take ನ (nouns ending in u, ū, ai, o, ō, au)
_SUFFIX_N = ("ು", "ೂ", "ೈ", "ೊ", "ೋ", "ೌ", "ೃ")

# Vowel sign ಆ (ā) and full vowels ಅ, ಆ take ದ
_SUFFIX_D_VOWELS = ("ಅ", "ಆ", "ಾ")

# Halant/virama (್): words ending in ್ take ನ (e.g. ಫೋನ್ ನ, ಬಸ್ ನ)
_KANNADA_HALANT = "\u0ccd"

# Kannada consonant range (U+0C95 to U+0CB9). Consonant ಯ (Ya) takes ನ (e.g. ಉಪಾಧ್ಯಾಯ ನಲ್ಲಿ).
_KANNADA_CONSONANT_YA = "\u0caf"
_KANNADA_CONSONANT_FIRST = "\u0c95"
_KANNADA_CONSONANT_LAST = "\u0cb9"


def get_genitive_suffix(word: str) -> str:
    """
    Return the genitive suffix (ನ, ಯ, or ದ) to use after a Kannada noun.
    - Words ending in ಇ, ಈ, ಎ, ಏ (ಿ, ೀ, ೆ, ೇ) → ಯ  (e.g. ಗುಲಾಬಿ ಯ, ಕರುಣೆ ಯ)
    - Words ending in ಉ, ಊ, ಐ, ಓ, etc. (ು, ೂ, ೈ, ೊ, ೋ, ೌ) → ನ  (e.g. ಜುಲೈ ನ)
    - Words ending in ಅ, ಆ or consonant → ದ  (e.g. ನಕ್ಷತ್ರಮಂಡಲ ದ)
    - If word is only digits (numeral), return ರ  (e.g. 72 ರ, 17 ರ)
    """
    if not word or not isinstance(word, str):
        return "ನ"
    s = word.strip()
    if not s:
        return "ನ"
    # Numerals: string of digits → ರ
    if s.isdigit():
        return "ರ"
    last = s[-1]
    if last in _SUFFIX_Y:
        return "ಯ"
    if last in _SUFFIX_N:
        return "ನ"
    if last == _KANNADA_CONSONANT_YA:
        return "ನ"  # e.g. ಉಪಾಧ್ಯಾಯ ನಲ್ಲಿ
    if last == _KANNADA_HALANT:
        return "ನ"  # e.g. ಫೋನ್ ನ, ಬಸ್ ನ (words ending in ್)
    if last in _SUFFIX_D_VOWELS:
        return "ದ"
    if _KANNADA_CONSONANT_FIRST <= last <= _KANNADA_CONSONANT_LAST:
        return "ದ"
    # Other (e.g. symbols): default ನ
    return "ನ"
