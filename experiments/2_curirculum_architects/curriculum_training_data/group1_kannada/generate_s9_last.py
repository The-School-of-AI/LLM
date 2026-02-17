#!/usr/bin/env python3
"""
Generate Statement 9: Last Letter (ಕೊನೆಯ ಅಕ್ಷರ) questions - Kannada
Target: 17,200 pairs (8.6% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_grammar import get_genitive_suffix  # noqa: E402
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

VOWELS = set([chr(c) for c in range(0x0C85, 0x0C91) if c not in [0x0C8C, 0x0C8E]])
CONSONANTS = set([chr(c) for c in range(0x0C95, 0x0CB9) if chr(c) not in ["ಱ", "ೞ"]])
VOWEL_SIGNS = "ಾಿೀುೂೃೄೆೇೈೊೋೌ"  # Dependent vowel signs
HALANT = "\u0ccd"


def get_last_vowel_in_cluster(cluster: str) -> str:
    """Extract vowel/matra from last syllabic cluster (e.g. ಲು -> ಉ, ಹಾ -> ಆ)."""
    for c in cluster:
        if c in VOWELS:
            return c
        if c in VOWEL_SIGNS:
            # Map matra to full vowel: ು->ಉ, ಾ->ಆ, etc.
            m = {
                "ಾ": "ಆ",
                "ಿ": "ಇ",
                "ೀ": "ಈ",
                "ು": "ಉ",
                "ೂ": "ಊ",
                "ೃ": "ಋ",
                "ೆ": "ಎ",
                "ೇ": "ಏ",
                "ೈ": "ಐ",
                "ೊ": "ಒ",
                "ೋ": "ಓ",
                "ೌ": "ಔ",
            }
            return m.get(c, c)
    return ""


def last_is_vowel_or_consonant(cluster: str) -> str:
    """Return ಸ್ವರ, ವ್ಯಂಜನ, or ಗುಣಿತಾಕ್ಷರ."""
    if cluster[0] in VOWELS:
        return "ಸ್ವರ"
    if HALANT in cluster or any(c in cluster[1:] for c in VOWEL_SIGNS):
        return "ಗುಣಿತಾಕ್ಷರ"
    return "ವ್ಯಂಜನ"


# Expand word lists
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Question templates: (template, type). type: "last" | "ends_with" | "last_vc" | "last_vowel"
TEMPLATES = [
    ('"{word}" {suffix} ಕೊನೆಯ ಅಕ್ಷರ ಏನು?', "last"),
    ('"{word}" ಯಾವ ಅಕ್ಷರದಿಂದ ಕೊನೆಗೊಳ್ಳುತ್ತದೆ?', "last"),
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರ ಏನು?', "last"),
    ('"{word}" {suffix} ಕಡೆಯ ಅಕ್ಷರ ಏನು?', "last"),
    ('"{word}" ಯಾವ ಅಕ್ಷರದಲ್ಲಿ ಮುಗಿಯುತ್ತದೆ?', "last"),
    ('"{word}" {suffix} ಕೊನೆಯಲ್ಲಿ ಯಾವ ಅಕ್ಷರವಿದೆ?', "last"),
    ('"{word}" ಪದದಲ್ಲಿ ಕೊನೆಯಲ್ಲಿ ಬರುವ ಅಕ್ಷರ ಯಾವುದು?', "last"),
    ('"{word}" {suffix} ಅಂತ್ಯದಲ್ಲಿರುವ ವರ್ಣ ಯಾವುದು?', "last"),  # 8
    ('"{word}" ಪದದ ಕಡೆಯ ಅಕ್ಷರ ಯಾವುದು ಎಂದು ಗುರುತಿಸಿ?', "last"),  # 9
    ('"{word}" ಪದವು "{char}" ಅಕ್ಷರದಿಂದ ಕೊನೆಯಾಗುತ್ತದೆಯೇ?', "ends_with"),  # 10
    ('"{word}" ಪದದ ಅಂತ್ಯಾಕ್ಷರ ಯಾವುದು?', "last"),  # 11
    ('"{word}" {suffix} ಕೊನೆಯ ಅಕ್ಷರವು ಸ್ವರವೋ ಅಥವಾ ವ್ಯಂಜನವೋ?', "last_vc"),  # 12
    ('"{word}" ಪದದ ಕೊನೆಯ ಅಕ್ಷರದಲ್ಲಿ ಯಾವ ಸ್ವರ ಅಡಗಿದೆ?', "last_vowel"),  # 13
]

all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 19200

# Generate samples
unique_combinations = {}
for word in set(all_words):
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    last_cluster = clusters[-1]
    suffix = get_genitive_suffix(word)
    for template_idx, (template, ttype) in enumerate(TEMPLATES):
        if ttype == "last":
            query = template.format(word=word, suffix=suffix)
            answer = last_cluster
        elif ttype == "ends_with":
            for c in clusters:
                query = template.format(word=word, char=c)
                answer = "ಹೌದು" if last_cluster == c else "ಅಲ್ಲ"
                key = (word, template_idx, c)
                if key not in unique_combinations:
                    unique_combinations[key] = (query, answer)
            continue
        elif ttype == "last_vc":
            vc = last_is_vowel_or_consonant(last_cluster)
            query = template.format(word=word, suffix=suffix)
            answer = vc
        elif ttype == "last_vowel":
            v = get_last_vowel_in_cluster(last_cluster)
            if not v:
                continue
            query = template.format(word=word)
            answer = v
        else:
            query = template.format(word=word, suffix=suffix)
            answer = last_cluster
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

samples = list(unique_combinations.values())
seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    word = random.choice(list(set(all_words)))
    clusters = get_kannada_grapheme_clusters(word)
    if len(clusters) == 0:
        continue

    last_cluster = clusters[-1]
    suffix = get_genitive_suffix(word)
    template, ttype = random.choice(TEMPLATES)
    if ttype == "last":
        query = template.format(word=word, suffix=suffix)
        answer = last_cluster
    elif ttype == "ends_with":
        c = random.choice(clusters)
        query = template.format(word=word, char=c)
        answer = "ಹೌದು" if last_cluster == c else "ಅಲ್ಲ"
    elif ttype == "last_vc":
        query = template.format(word=word, suffix=suffix)
        answer = last_is_vowel_or_consonant(last_cluster)
    elif ttype == "last_vowel":
        v = get_last_vowel_in_cluster(last_cluster)
        if not v:
            no_progress += 1
            continue
        query = template.format(word=word)
        answer = v
    else:
        query = template.format(word=word, suffix=suffix)
        answer = last_cluster
    if (query, answer) not in seen_qa:
        seen_qa.add((query, answer))
        samples.append((query, answer))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S9 Last Letter (Kannada): Generated {len(samples)} samples")
