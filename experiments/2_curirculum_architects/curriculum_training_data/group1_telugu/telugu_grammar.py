#!/usr/bin/env python3
"""
Telugu grammar helpers for dataset generation.
Akshara segmentation: syllabic units per Telugu linguistics (conjuncts, anusvara, visarga).
No genitive suffix logic needed — Telugu uses invariant postpositions (లో, యొక్క).
"""

import regex

# Virama (halant): consonant + ్ forms half-form; next consonant attaches (conjunct)
TELUGU_VIRAMA = "\u0c4d"  # ్


def _ends_with_virama(s: str) -> bool:
    """True if string ends with virama (halant), indicating an incomplete akshara."""
    return bool(s and s[-1] == TELUGU_VIRAMA)


def get_telugu_aksharas(word: str) -> list[str]:
    """
    Segment Telugu word into aksharas (syllabic units).
    Per Telugu linguistics: Conjuncts (సంయుక్తాక్షరాలు like స్త, ద్య, క్ష) = 1 unit;
    Anusvara (ం) = part of preceding letter; Visarga (ః) = part of preceding.
    Uses grapheme clusters + virama merging.
    E.g. పుస్తకం -> ['పు', 'స్త', 'కం'] (3 aksharas)
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


# Function to extract Telugu letters (ignoring vowel signs if desired)
def get_telugu_aksharas_with_roots(word, ignore_vowels=False):
    """
    Returns a list of Telugu letters in the word.
    If ignore_vowels=True, vowel signs are skipped.
    """
    letters = []
    for ch in word:
        code = ord(ch)
        # Telugu consonants range
        if (
            0x0C15 <= code <= 0x0C39
            or 0x0C05 <= code <= 0x0C14
            or 0x0C02 <= code <= 0x0C03
        ):
            letters.append(ch)
        # Include vowel signs only if not ignoring
        elif not ignore_vowels and 0x0C3E <= code <= 0x0C4C:
            letters.append(ch)
    return letters
