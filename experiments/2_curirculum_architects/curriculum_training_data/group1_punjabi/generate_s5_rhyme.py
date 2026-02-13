#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (ਤੁਕਬੰਦੀ) questions for Punjabi
Target: 20,000 pairs
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE, RHYMING_PAIRS
from prompt_utils import format_qa_pair_hindi

TEMPLATES = [
    '"{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਨ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{rhyme}" ਜਾਂ "{non_rhyme}"?',
    '"{word}" ਨਾਲ ਕਿਹੜਾ ਸ਼ਬਦ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ, "{rhyme}" ਜਾਂ "{non_rhyme}"?',
    '"{word}" ਦੇ ਨਾਲ ਤੁਕਬੰਦੀ ਵਾਲਾ ਸ਼ਬਦ "{rhyme}" ਅਤੇ "{non_rhyme}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ, "{rhyme}" ਜਾਂ "{non_rhyme}"?',
    '"{word}" ਨਾਲ ਰਾਈਮ ਕਰਨ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{rhyme}" ਜਾਂ "{non_rhyme}"?',
    '"{rhyme}" ਅਤੇ "{non_rhyme}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ "{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ?',
]


def generate_samples(target_count):
    samples = set()
    pairs = list(RHYMING_PAIRS.items())
    all_words = ALL_WORDS_UNIQUE

    for word, rhyme in pairs:
        non_rhymes = [w for w in all_words if w != rhyme and w != word]
        for _ in range(10):  # variety of non-rhymes
            non_rhyme = random.choice(non_rhymes)
            w1, w2 = (rhyme, non_rhyme) if random.random() < 0.5 else (non_rhyme, rhyme)
            for template in TEMPLATES:
                query = template.format(
                    word=word, rhyme=rhyme, non_rhyme=non_rhyme, w1=w1, w2=w2
                )  # w1,w2 handles switch
                # Wait, template needs {rhyme} and {non_rhyme} but also order variation
                # Fix template above to use {option1} and {option2}
                pass

    # Redoing with better template logic
    TEMPLATES_V2 = [
        '"{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਨ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{o1}" ਜਾਂ "{o2}"?',
        '"{word}" ਨਾਲ ਕਿਹੜਾ ਸ਼ਬਦ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ, "{o1}" ਜਾਂ "{o2}"?',
        '"{word}" ਨਾਲ ਰਾਈਮ ਕਰਨ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{o1}" ਜਾਂ "{o2}"?',
        '"{word}" ਦੇ ਨਾਲ ਤੁਕਬੰਦੀ ਵਾਲਾ ਸ਼ਬਦ "{o1}" ਅਤੇ "{o2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਹੈ?',
        'ਕਿਹੜਾ ਸ਼ਬਦ "{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ, "{o1}" ਜਾਂ "{o2}"?',
        '"{o1}" ਅਤੇ "{o2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਸ਼ਬਦ "{word}" ਨਾਲ ਤੁਕਬੰਦੀ ਕਰਦਾ ਹੈ?',
    ]

    samples = set()
    for word, rhyme in pairs:
        non_rhymes = [w for w in all_words if w != rhyme and w != word]
        for _ in range(20):
            nr = random.choice(non_rhymes)
            o1, o2 = (rhyme, nr) if random.random() < 0.5 else (nr, rhyme)
            for template in TEMPLATES_V2:
                query = template.format(word=word, o1=o1, o2=o2)
                answer = rhyme
                samples.add((query, answer))
                if len(samples) >= target_count:
                    return list(samples)
    return list(samples)


if __name__ == "__main__":
    target_count = 20000
    samples = generate_samples(target_count)
    output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")
    print(f"S5 Punjabi Rhyming: Generated {len(samples)} unique samples")
