#!/usr/bin/env python3
"""
Telugu grammar helpers for dataset generation.
Akshara segmentation: syllabic units per Telugu linguistics (conjuncts, anusvara, visarga).
No genitive suffix logic needed — Telugu uses invariant postpositions (లో, యొక్క).
"""
import regex

# Virama (halant): consonant + ్ forms half-form; next consonant attaches (conjunct)
TELUGU_VIRAMA = "\u0C4D"  # ్


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
