#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ਧੁਨੀ ਮਿਲਾਨ) questions for Punjabi
Target: 20,000 pairs
"""

import os
import random
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi


def get_punjabi_first_sound(word: str) -> str:
    clusters = regex.findall(r"\X", word)
    if not clusters:
        return ""
    # Simplified: first grapheme cluster
    return clusters[0]


TEMPLATES = [
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਧੁਨੀ ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"{word1}" ਅਤੇ "{word2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਨਾਲ ਆਰੰਭ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"{word1}" ਅਤੇ "{word2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਧੁਨੀ ਵਾਲਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਧੁਨੀ ਨਾਲ ਆਰੰਭ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
]


def generate_samples(target_count):
    samples = set()
    words = ALL_WORDS_UNIQUE

    # Pre-group by sounds
    word_groups = {}
    for w in words:
        s = get_punjabi_first_sound(w)
        if s not in word_groups:
            word_groups[s] = []
        word_groups[s].append(w)

    sounds = list(word_groups.keys())

    for sound in sounds:
        correct_words = word_groups[sound]
        wrong_words = [w for w in words if get_punjabi_first_sound(w) != sound]
        if not wrong_words:
            continue

        for correct_word in correct_words:
            # Pick a few wrong words to create variety
            for _ in range(5):
                wrong_word = random.choice(wrong_words)
                w1, w2 = (
                    (correct_word, wrong_word)
                    if random.random() < 0.5
                    else (wrong_word, correct_word)
                )
                for template in TEMPLATES:
                    query = template.format(sound=sound, word1=w1, word2=w2)
                    answer = correct_word
                    samples.add((query, answer))
                    if len(samples) >= target_count:
                        return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 20000
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S3 Punjabi Sound Matching: Generated {len(samples)} unique samples")
