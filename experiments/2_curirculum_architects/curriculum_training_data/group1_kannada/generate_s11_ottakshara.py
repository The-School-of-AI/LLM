#!/usr/bin/env python3
"""
Generate Statement 11: Ottakshara & Kagunita (ಒತ್ತಕ್ಷರ, ಗುಣಿತಾಕ್ಷರ) - Kannada
Language-specific: conjuncts, vowel signs, ಋ/ಐ/ಔ, ಷ vs ಶ, ಕ್ಷ as consonant.
Target: 10,000 pairs. User-specified question set.
"""
import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair_kannada, get_kannada_grapheme_clusters  # noqa: E402
from group1_kannada.kannada_vocabulary import ALL_WORDS_UNIQUE, VARGAS  # noqa: E402
from group1_kannada.kannada_grammar import get_kannada_aksharas

# Vowel signs (Matras) mapping
MATRA_MAP = {
    '\u0BE7': 'ಆ-ಕಾರ', # ಾ
    '\u0BE8': 'ಇ-ಕಾರ', # ಿ
    '\u0BE9': 'ಈ-ಕಾರ', # ೀ
    '\u0BEA': 'ಉ-ಕಾರ', # ು
    '\u0BEB': 'ಊ-ಕಾರ', # ೂ
    '\u0BEC': 'ಋ-ಕಾರ', # ೃ
    '\u0BED': 'ಎ-ತ್ವ', # ೆ
    '\u0BEE': 'ಏ-ತ್ವ', # ೇ
    '\u0BEF': 'ಐ-ಕಾರ', # ೈ
    '\u0BF0': 'ಓ-ತ್ವ', # ೊ
    '\u0BF1': 'ಓ-ತ್ವ', # ೋ (Note: Both can be called O-tva or O-kara, using consistent naming)
    '\u0BF2': 'ಔ-ತ್ವ', # ೌ
    '\u0C01': 'ಅಕ',    # ಁ (Gumm) - Anusvara logic usually handles ಂ (\u0C02)
    '\u0C02': 'ಅನುಸ್ವಾರ', # ಂ
    '\u0C03': 'ವಿಸರ್ಗ',   # ಃ
}

# Unicode constants
VIRAMA = '\u0CCD'
RA = '\u0CB0'

ORDINALS = ["", "ಮೊದಲ", "ಎರಡನೇ", "ಮೂರನೇ", "ನಾಲ್ಕನೇ", "ಐದನೇ"]

def get_ottakshara_component(akshara: str) -> str:
    """
    Extract the Ottakshara (subscript) part from a complex Akshara.
    E.g., 'ಪ್ಪ' -> 'ಪ' (base) + '್' + 'ಪ' (sub). Wait, logically 'ಪ್ಪ' is the ottakshara syllable.
    The question 'Ottakshara yavudu?' usually expects the *cluster* or the *subscript*.
    Hardcoded example 'ಅಪ್ಪ' -> 'ಪ್ಪ'. So it expects the full conjunct Akshara.
    """
    # Check if it has Virama + Consonant (but not just Virama at end)
    # Regex for Consonant+Virama+Consonant
    if regex.search(r'[\u0C95-\u0CB9]\u0CCD[\u0C95-\u0CB9]', akshara):
        return akshara
    return None

def get_base_consonant(akshara: str) -> str:
    """Return base consonant of the cluster."""
    # First char usually
    return akshara[0]

def analyze_word_for_questions(word: str) -> list:
    """Generate QA pairs for a single word."""
    questions = []
    
    aksharas = get_kannada_aksharas(word)
    clusters = get_kannada_grapheme_clusters(word) # Use for precise matra finding if needed
    
    # 1. Ottakshara Identification
    for ak in aksharas:
        ottakshara = get_ottakshara_component(ak)
        if ottakshara:
             questions.append((
                 f'"{word}" ಪದದಲ್ಲಿರುವ ಒತ್ತಕ್ಷರ ಯಾವುದು?',
                 ottakshara
             ))
             questions.append((
                 f'"{word}" ಪದದಲ್ಲಿರುವ ಸಂಯುಕ್ತಾಕ್ಷರ ಯಾವುದು?',
                 ottakshara
             ))
    
    # 2. Matra Identification
    for cluster in clusters:
        for char in cluster:
            if char in MATRA_MAP:
                matra_name = MATRA_MAP[char]
                questions.append((
                    f'"{word}" ಪದದಲ್ಲಿರುವ {matra_name} ಯಾವುದು?',
                    char
                ))
    
    # 3. Arkavattu (Repha) - Ra + Virama + Consonant
    # Visually typically Ra+Virama+Consonant (e.g. ರ್ಯ).
    # But in stored unicode it is Ra+Virama+Ya.
    # Check if word contains Ra+Virama+Consonant
    if re_arkavattu := regex.search(fr'{RA}{VIRAMA}([\u0C95-\u0CB9])', word):
        # Result is the cluster containing it.
        # Find the specific cluster
        for cluster in clusters:
            if RA + VIRAMA in cluster:
                # This cluster has Arkavattu
                questions.append((
                   f'"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು ಚಿಹ್ನೆಯನ್ನು ಗುರುತಿಸಿ?',
                   cluster
                ))

    # 4. Independent Letter Count (Akshara count)
    count = len(aksharas)
    questions.append((
        f'"{word}" ಪದದಲ್ಲಿ ಎಷ್ಟು ಸ್ವತಂತ್ರ ಅಕ್ಷರಗಳಿವೆ?',
        str(count)
    ))
    
    return questions

def generate_varga_questions():
    """Generate questions about Vargas."""
    qs = []
    for varga_name, letters in VARGAS.items():
        # "X ವರ್ಗದ Yನೇ ಅಕ್ಷರ ಯಾವುದು?"
        for idx, letter in enumerate(letters):
            ordinal = ORDINALS[idx + 1]
            qs.append((
                f'"{varga_name}" ವರ್ಗದ {ordinal} ಅಕ್ಷರ ಯಾವುದು?',
                letter
            ))
            # Inverse
            qs.append((
                f'"{letter}" ಅಕ್ಷರವು ಯಾವ ವರ್ಗಕ್ಕೆ ಸೇರಿದೆ?',
                f'"{varga_name}" ವರ್ಗ'
            ))
            # Nasal (5th letter)
            if idx == 4:
                qs.append((
                    f'"{varga_name}" ವರ್ಗದ ಅನುನಾಸಿಕ ಅಕ್ಷರ ಯಾವುದು?',
                    letter
                ))
    return qs

# Main Generation
generated_samples = set()

# 1. Vocabulary based
for word in ALL_WORDS_UNIQUE:
    qas = analyze_word_for_questions(word)
    for q, a in qas:
        generated_samples.add((q, a))

# 2. Varga based
for q, a in generate_varga_questions():
    generated_samples.add((q, a))

# 3. Add manual existing ones (if not covered or specific)
OTTAKSHARA_QA_MANUAL = [
    ('"ಷ" ಮತ್ತು "ಶ" ಅಕ್ಷರಗಳ ವ್ಯತ್ಯಾಸವೇನು?', "ಅವು ವಿಭಿನ್ನ ವ್ಯಂಜನಾಕ್ಷರಗಳು; ಉಚ್ಚಾರಣೆ ಮತ್ತು ಲಿಪಿ ವಿಭಿನ್ನವಾಗಿವೆ."),
    ('"{char}" ಅಕ್ಷರವು ಸ್ವರವೋ ಅಥವಾ ವ್ಯಂಜನವೋ?'.format(char="ಕ್ಷ"), "ವ್ಯಂಜನ"), # ಕ್ಷ is technically compound
]
for q, a in OTTAKSHARA_QA_MANUAL:
    generated_samples.add((q, a))

# Convert to list
samples_list = list(generated_samples)

# Fill to target count by repetition if needed (though 10k unique might be hard with 1000 words)
# With 1000 words, maybe 3-4 questions per word -> ~3000-4000 questions.
# We will repeat the list to fill.
final_samples = []
target_count = 10000 
while len(final_samples) < target_count:
    random.shuffle(samples_list)
    needed = target_count - len(final_samples)
    final_samples.extend(samples_list[:needed])

output_file = os.path.join(os.path.dirname(__file__), "group1_s11.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in final_samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S11 Ottakshara/Kagunita (Kannada): Generated {len(final_samples)} samples (Unique: {len(samples_list)})")
