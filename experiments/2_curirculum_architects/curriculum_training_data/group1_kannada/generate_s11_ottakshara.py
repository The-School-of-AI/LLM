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
    '\u0BF1': 'ಓ-ತ್ವ', # ೋ
    '\u0BF2': 'ಔ-ತ್ವ', # ೌ
    '\u0C01': 'ಅಕ',    # ಁ
    '\u0C02': 'ಅನುಸ್ವಾರ', # ಂ
    '\u0C03': 'ವಿಸರ್ಗ',   # ಃ
}

# Unicode constants
VIRAMA = '\u0CCD'
RA = '\u0CB0'

ORDINALS = ["", "ಮೊದಲ", "ಎರಡನೇ", "ಮೂರನೇ", "ನಾಲ್ಕನೇ", "ಐದನೇ"]

# Question Templates for Rephrasing
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

def get_ottakshara_component(akshara: str) -> str:
    """Extract full Ottakshara (syllable with conjunct) from Akshara."""
    if regex.search(r'[\u0C95-\u0CB9]\u0CCD[\u0C95-\u0CB9]', akshara):
        return akshara
    return None

def is_sajatiya(akshara: str) -> bool:
    """
    Check if akshara is Sajatiya (same consonant conjunct).
    Structure: Consonant1 + Virama + Consonant2 [+ Matra]
    Sajatiya if Consonant1 == Consonant2
    """
    # Find all consonants in the akshara
    consonants = regex.findall(r'[\u0C95-\u0CB9]', akshara)
    if len(consonants) >= 2:
        # Check if first two consonants are the same (typical case)
        # e.g. 'ಪ್ಪ' -> ['ಪ', 'ಪ'] -> True
        # e.g. 'ಕ್ತ' -> ['ಕ', 'ತ'] -> False
        return consonants[0] == consonants[1]
    return False

def analyze_word_for_questions(word: str) -> list:
    """Generate QA pairs for a single word."""
    questions = []
    
    aksharas = get_kannada_aksharas(word)
    clusters = get_kannada_grapheme_clusters(word)
    
    has_ottakshara = False

    # 1. Ottakshara Identification & Classification
    for ak in aksharas:
        ottakshara = get_ottakshara_component(ak)
        if ottakshara:
             has_ottakshara = True
             # A. Identification (Rephrased)
             template = random.choice(OTTAKSHARA_TEMPLATES)
             questions.append((template.format(word=word), ottakshara))
             
             # B. Classification (Sajatiya/Vijatiya)
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
                # Rephrase simple prompt
                q_variations = [
                    f'"{word}" ಪದದಲ್ಲಿರುವ {matra_name} ಯಾವುದು?',
                    f'"{word}" ಪದದಲ್ಲಿ {matra_name} ಇದೆಯೇ? ಹೌದು, ಅದು "{char}".'
                ]
                questions.append((random.choice(q_variations), char))
    
    # 4. Arkavattu (Repha)
    if re_arkavattu := regex.search(fr'{RA}{VIRAMA}([\u0C95-\u0CB9])', word):
        for cluster in clusters:
            if RA + VIRAMA in cluster:
                # This cluster has Arkavattu
                questions.append((
                   f'"{word}" ಪದದಲ್ಲಿರುವ ಅರ್ಕಾವತ್ತು ಚಿಹ್ನೆಯನ್ನು ಗುರುತಿಸಿ?',
                   cluster
                ))

    # 5. Independent Letter Count (Akshara count)
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

# 2. Negation (No Ottakshara)
# Find words WITHOUT ottakshara to ask "Does this have ottakshara? -> No"
simple_words = [w for w in ALL_WORDS_UNIQUE if not any(get_ottakshara_component(ak) for ak in get_kannada_aksharas(w))]

# Take a sample of simple words equal to some portion of complex words
# Just add a few hundreds to balance
count_neg = 0
for word in simple_words:
    if count_neg > 1000: break
    generated_samples.add((f'"{word}" ಪದದಲ್ಲಿ ಒತ್ತಕ್ಷರ ಇದೆಯೇ?', "ಇಲ್ಲ"))
    generated_samples.add((f'"{word}" ಪದದಲ್ಲಿ ಸಂಯುಕ್ತಾಕ್ಷರ ಇದೆಯೇ?', "ಇಲ್ಲ"))
    generated_samples.add((f'"{word}" ಪದವು ಒತ್ತಕ್ಷರರಹಿತವೇ?', "ಹೌದು"))
    count_neg += 1

# 3. Varga based
for q, a in generate_varga_questions():
    generated_samples.add((q, a))

# 4. Manual Additions
OTTAKSHARA_QA_MANUAL = [
    ('"ಷ" ಮತ್ತು "ಶ" ಅಕ್ಷರಗಳ ವ್ಯತ್ಯಾಸವೇನು?', "ಅವು ವಿಭಿನ್ನ ವ್ಯಂಜನಾಕ್ಷರಗಳು; ಉಚ್ಚಾರಣೆ ಮತ್ತು ಲಿಪಿ ವಿಭಿನ್ನವಾಗಿವೆ."),
    ('"{char}" ಅಕ್ಷರವು ಸ್ವರವೋ ಅಥವಾ ವ್ಯಂಜನವೋ?'.format(char="ಕ್ಷ"), "ವ್ಯಂಜನ"), # ಕ್ಷ is technically compound
]
for q, a in OTTAKSHARA_QA_MANUAL:
    generated_samples.add((q, a))

# Convert to list
samples_list = list(generated_samples)

# Fill to target count by repetition if needed
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
