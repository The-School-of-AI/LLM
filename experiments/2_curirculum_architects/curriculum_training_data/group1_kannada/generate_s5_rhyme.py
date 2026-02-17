#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (ಪ್ರಾಸ) questions - Kannada
Uses actual rhyming words from vocabulary (same last akshara).
Templates: multiple-choice, open-ended, do-they-rhyme (yes/no).
Target: up to 20,000.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_vocabulary import (  # noqa: E402
    ALL_WORDS_UNIQUE,
    BAD_RHYME_PAIRS,
    RHYMING_GROUPS,
    RHYMING_PAIRS,
)
from prompt_utils import format_qa_pair_kannada  # noqa: E402

unique_words = list(set(ALL_WORDS_UNIQUE))
rhyme_set = {(a, b) for a, b in RHYMING_PAIRS.items()} | {
    (b, a) for a, b in RHYMING_PAIRS.items()
}


def do_rhyme(w1: str, w2: str) -> bool:
    """Check if two words rhyme (ಪ್ರಾಸ); False for BAD_RHYME_PAIRS."""
    if (w1, w2) in BAD_RHYME_PAIRS or (w2, w1) in BAD_RHYME_PAIRS:
        return False
    return (w1, w2) in rhyme_set or w2 in RHYMING_GROUPS.get(w1, [])


# Templates using ಪದದ (not ಶಬ್ದದ)
TEMPLATES = [
    (
        '"{word}" ಪದಕ್ಕೆ ಪ್ರಾಸ ಆಗುವ ಪದ ಯಾವುದು, "{option1}" ಅಥವಾ "{option2}"?',
        "multiple_choice",
    ),
    ('"{word}" ಪದಕ್ಕೆ ಪ್ರಾಸಬದ್ಧವಾದ ಪದ ಯಾವುದು?', "open_ended"),
    ('"{word}" ಪದಕ್ಕೆ ಪ್ರಾಸವಾಗುವ ಮತ್ತೊಂದು ಪದ ತಿಳಿಸಿ?', "open_ended"),
    ('"{word1}" ಮತ್ತು "{word2}" ಪದಗಳು ಪ್ರಾಸವಾಗುತ್ತವೆಯೇ?', "do_rhyme_yes_no"),
    ('"{word1}" ಮತ್ತು "{word2}" ಪದಗಳು ಪ್ರಾಸಬದ್ಧವೇ?', "do_rhyme_yes_no"),
]

samples = []
seen_qa = set()


def add_unique(q: str, a: str) -> None:
    key = (q, a)
    if key not in seen_qa:
        seen_qa.add(key)
        samples.append((q, a))


# 1. Multiple-choice: "X ಪದಕ್ಕೆ ಪ್ರಾಸ ಆಗುವ ಪದ ಯಾವುದು, A ಅಥವಾ B?"
seen_mc = set()
for word, rhyme_word in RHYMING_PAIRS.items():
    if word in seen_mc:
        continue
    non_rhyming = [
        w
        for w in unique_words
        if w != word and w != rhyme_word and not do_rhyme(word, w)
    ]
    if not non_rhyming:
        continue
    distractor = random.choice(non_rhyming)
    option1, option2 = random.sample([rhyme_word, distractor], 2)
    q = TEMPLATES[0][0].format(word=word, option1=option1, option2=option2)
    add_unique(q, rhyme_word)
    seen_mc.add(word)
# Reverse
for word, rhyme_word in RHYMING_PAIRS.items():
    if rhyme_word in seen_mc:
        continue
    non_rhyming = [
        w
        for w in unique_words
        if w != rhyme_word and w != word and not do_rhyme(rhyme_word, w)
    ]
    if not non_rhyming:
        continue
    distractor = random.choice(non_rhyming)
    option1, option2 = random.sample([word, distractor], 2)
    q = TEMPLATES[0][0].format(word=rhyme_word, option1=option1, option2=option2)
    add_unique(q, word)
    seen_mc.add(rhyme_word)


# 2. Open-ended: "X ಪದಕ್ಕೆ ಪ್ರಾಸಬದ್ಧವಾದ ಪದ ಯಾವುದು?" - any valid rhyme from RHYMING_GROUPS
for word, rhymes in RHYMING_GROUPS.items():
    if not rhymes:
        continue
    for tmpl_idx in (1, 2):  # both open-ended templates
        answer = random.choice(rhymes)
        q = TEMPLATES[tmpl_idx][0].format(word=word)
        add_unique(q, answer)


# 3. Do they rhyme? (yes) - pairs from same rhyme group
for word, rhymes in RHYMING_GROUPS.items():
    if len(rhymes) < 2:
        continue
    w1 = random.choice(rhymes)
    q = TEMPLATES[3][0].format(word1=word, word2=w1)
    add_unique(q, "ಹೌದು")
    q = TEMPLATES[4][0].format(word1=word, word2=w1)
    add_unique(q, "ಹೌದು")


# 4. Do they rhyme? (no) - word + non-rhyme
for word in list(RHYMING_GROUPS.keys())[:3000]:  # cap to avoid explosion
    non_rhyming = [w for w in unique_words if w != word and not do_rhyme(word, w)]
    if not non_rhyming:
        continue
    other = random.choice(non_rhyming)
    for tmpl_idx in (3, 4):
        q = TEMPLATES[tmpl_idx][0].format(word1=word, word2=other)
        add_unique(q, "ಅಲ್ಲ")  # ಪ್ರಾಸಬದ್ಧವೇ? → quality → ಅಲ್ಲ

random.shuffle(samples)
# Cap at target
target_count = 20000
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S5 Rhyming (Kannada): Generated {len(samples)} samples")
