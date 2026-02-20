#!/usr/bin/env python3
"""
Generate Statement 2: Positional Analysis (Merged S2+S7)
Target: 55,000 pairs (no exact duplicates)
Focus: Identify letter at index X, and Index of letter X.
"""
import os
import random
import sys
import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    ALL_WORDS_UNIQUE,
    ORDINALS,
)
from prompt_utils import format_qa_pair_hindi

# Expand word list to ensure enough samples
WORDS = ALL_WORDS_UNIQUE * 50

TEMPLATES_LETTER_AT_INDEX = [
    '"{word}" শব্দটোৰ {ordinal} আখৰটো কি?',
    '"{word}"ৰ {ordinal} আখৰ কি?',
    '"{word}" শব্দত {ordinal} স্থানত কি আখৰ আছে?',
    '"{word}"ৰ {ordinal} বৰ্ণটো চিনাক্ত কৰক।',
]

TEMPLATES_INDEX_OF_LETTER = [
    '"{word}"ত "{char}" আখৰটো কেই নম্বৰত আছে?',
    '"{word}" শব্দত "{char}" কিমান নম্বৰ স্থানত আছে?',
    '"{word}"ৰ কোনটো স্থানত "{char}" আছে?',
    '"{word}"ত "{char}"ৰ স্থান কি?',
]

def get_assamese_grapheme_clusters(word: str) -> list[str]:
    """Get grapheme clusters for Assamese word using \\X pattern."""
    return regex.findall(r"\X", word)

def main():
    samples = []
    seen = set()  # Track (query, answer) to avoid exact duplicates
    target_count = 55000

    # Pre-calculate clusters for all words to save time
    word_clusters = [(w, get_assamese_grapheme_clusters(w)) for w in set(WORDS) if len(w) > 1]
    max_attempts = target_count * 20  # Safeguard against infinite loop
    attempts = 0

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        word, clusters = random.choice(word_clusters)
        length = len(clusters)

        # Mode 1: Letter at Index
        if random.random() < 0.5:
            idx = random.randint(0, length - 1)
            if idx < len(ORDINALS):
                ordinal = ORDINALS[idx]
                template = random.choice(TEMPLATES_LETTER_AT_INDEX)
                query = template.format(word=word, ordinal=ordinal)
                answer = clusters[idx]
                key = (query, answer)
                if key not in seen:
                    seen.add(key)
                    samples.append((query, answer))

        # Mode 2: Index of Letter
        else:
            target_char = random.choice(clusters)
            indices = [i + 1 for i, char in enumerate(clusters) if char == target_char]
            ans_indices = []
            for i in indices:
                if (i - 1) < len(ORDINALS):
                    ans_indices.append(ORDINALS[i - 1])
                else:
                    ans_indices.append(str(i))
            answer = ", ".join(ans_indices)
            template = random.choice(TEMPLATES_INDEX_OF_LETTER)
            query = template.format(word=word, char=target_char)
            key = (query, answer)
            if key not in seen:
                seen.add(key)
                samples.append((query, answer))

    # Shuffle
    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s2.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S2 Positional: Generated {len(samples)} samples (no duplicates)")

if __name__ == "__main__":
    main()
