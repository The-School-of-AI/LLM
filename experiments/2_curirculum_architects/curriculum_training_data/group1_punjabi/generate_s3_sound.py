#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ਧੁਨੀ ਮਿਲਾਨ) questions for Punjabi
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_punjabi.punjabi_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates
TEMPLATES = [
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਧੁਨੀ ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"/{sound}/" ਧੁਨੀ ਨਾਲ ਸ਼ੁਰੂ ਹੋਣ ਵਾਲਾ ਸ਼ਬਦ ਕਿਹੜਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਧੁਨੀ ਨਾਲ ਆਰੰਭ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
    '"/{sound}/" ਧੁਨੀ ਨਾਲ ਸ਼ੁਰੂ ਹੋਣ ਵਾਲਾ ਸ਼ਬਦ "{word1}" ਅਤੇ "{word2}" ਵਿੱਚੋਂ ਕਿਹੜਾ ਹੈ?',
    'ਕਿਹੜਾ ਸ਼ਬਦ "/{sound}/" ਨਾਲ ਸ਼ੁਰੂ ਹੁੰਦਾ ਹੈ, "{word1}" ਜਾਂ "{word2}"?',
]


def get_first_sound(word: str) -> str:
    """Get the first sound/character of a Punjabi word"""
    if not word:
        return ""
    return word[0]


# Pre-compute word groups by first sound
unique_words = list(set(ALL_WORDS))
words_by_sound = {}
for word in unique_words:
    sound = get_first_sound(word)
    if sound:
        if sound not in words_by_sound:
            words_by_sound[sound] = []
        words_by_sound[sound].append(word)

# Pre-compute all sounds
all_sounds = list(words_by_sound.keys())

samples = []
target_count = 20000
unique_combinations = set()

# Generate samples efficiently
for word1 in unique_words:
    sound1 = get_first_sound(word1)
    if not sound1 or sound1 not in words_by_sound:
        continue

    # Get non-matching words from other sounds
    non_matching_words = []
    for sound in all_sounds:
        if sound != sound1:
            non_matching_words.extend(words_by_sound[sound])

    if not non_matching_words:
        continue

    for template_idx, template in enumerate(TEMPLATES):
        word2_nonmatch = random.choice(non_matching_words)
        query = template.format(sound=sound1, word1=word1, word2=word2_nonmatch)
        answer = word1
        key = (word1, word2_nonmatch, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

# Sample with replacement to reach target
if len(samples) > target_count:
    samples = samples[:target_count]
else:
    while len(samples) < target_count:
        word1 = random.choice(unique_words)
        sound1 = get_first_sound(word1)
        if not sound1 or sound1 not in words_by_sound:
            continue

        non_matching_words = []
        for sound in all_sounds:
            if sound != sound1:
                non_matching_words.extend(words_by_sound[sound])

        if not non_matching_words:
            continue

        template = random.choice(TEMPLATES)
        word2 = random.choice(non_matching_words)

        query = template.format(sound=sound1, word1=word1, word2=word2)
        answer = word1
        samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S3 Punjabi Sound Matching: Generated {len(samples)} samples")
