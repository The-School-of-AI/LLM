#!/usr/bin/env python3
"""
Generate Statement 3: Phonetic Matching (Sibilants/Wa)
Target: 15,000 pairs
Focus: Identifying words with specific sounds (Sibilants: শ, ষ, স -> 'x' sound; Wa/Ba distinction).
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import ALL_WORDS_UNIQUE
from prompt_utils import format_qa_pair_hindi

TEMPLATES_CONTAINS = [
    'তলৰ কোনটো শব্দত "{sound_char}" আছে: "{word1}" নে "{word2}"?',
    '"{word1}" আৰু "{word2}"ৰ ভিতৰত কোনটোত "{sound_char}" ব্যৱহাৰ হৈছে?',
    '"{sound_char}" থকা শব্দটো বাছনি কৰক: "{word1}", "{word2}"।',
]

TEMPLATES_STARTS_WITH = [
    'কোনটো শব্দ "{sound_char}"ৰে আৰম্ভ হৈছে: "{word1}" নে "{word2}"?',
    '"{sound_char}" আখৰেৰে আৰম্ভ হোৱা শব্দটো কি? "{word1}" নে "{word2}"?',
]


def main():
    samples = []
    target_count = 15000

    # Target characters for phonetic questions
    targets = ["শ", "ষ", "স", "ৱ", "ব"]

    while len(samples) < target_count:
        target_char = random.choice(targets)

        # Find a correct word (contains the char)
        correct_candidates = [w for w in ALL_WORDS_UNIQUE if target_char in w]
        if not correct_candidates:
            continue
        correct_word = random.choice(correct_candidates)

        # Find a distractor (does NOT contain the char)
        distractor_candidates = [w for w in ALL_WORDS_UNIQUE if target_char not in w]
        if not distractor_candidates:
            continue
        distractor_word = random.choice(distractor_candidates)

        # Randomize order
        words = [correct_word, distractor_word]
        random.shuffle(words)
        w1, w2 = words

        # Choose template type
        # Check if correct word STARTS with char for "Starts With" templates
        if correct_word.startswith(target_char):
            template = random.choice(TEMPLATES_CONTAINS + TEMPLATES_STARTS_WITH)
        else:
            template = random.choice(TEMPLATES_CONTAINS)

        query = template.format(sound_char=target_char, word1=w1, word2=w2)
        answer = correct_word

        samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S3 Phonetic: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
