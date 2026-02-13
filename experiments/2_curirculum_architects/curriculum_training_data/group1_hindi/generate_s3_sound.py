#!/usr/bin/env python3
"""
Generate Statement 3: Sound Matching (ध्वनि मिलान) questions
Target: 20,000 pairs (10% of 200,000)
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair_hindi  # noqa: E402
from group1_hindi.hindi_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates
TEMPLATES = [
    'कौन सा शब्द "/{sound}/" ध्वनि से शुरू होता है, "{word1}" या "{word2}"?',
    '"/{sound}/" ध्वनि से शुरू होने वाला शब्द कौन सा है, "{word1}" या "{word2}"?',
    'कौन सा शब्द "/{sound}/" ध्वनि से आरंभ होता है, "{word1}" या "{word2}"?',
    '"/{sound}/" ध्वनि से शुरू होने वाला शब्द "{word1}" और "{word2}" में से कौन सा है?',
    'कौन सा शब्द "/{sound}/" से शुरू होता है, "{word1}" या "{word2}"?',
]

def get_first_sound(word: str) -> str:
    """Get the first sound/character of a Hindi word"""
    if not word:
        return ""
    return word[0]

# Pre-compute word groups by first sound (OPTIMIZATION)
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
    
    matching_words = [w for w in words_by_sound[sound1] if w != word1]
    # Get non-matching words from other sounds
    non_matching_words = []
    for sound in all_sounds:
        if sound != sound1:
            non_matching_words.extend(words_by_sound[sound])
    
    if not matching_words or not non_matching_words:
        continue
    
    for template_idx, template in enumerate(TEMPLATES):
        # CRITICAL FIX: Only create pairs where word2 does NOT match the sound
        # Both words matching the sound creates invalid questions (no correct answer)
        # word1 always matches, word2 must be a distractor (doesn't match)
        word2_nonmatch = random.choice(non_matching_words)
        query = template.format(sound=sound1, word1=word1, word2=word2_nonmatch)
        answer = word1  # word1 is correct because it matches the sound
        key = (word1, word2_nonmatch, template_idx)
        if key not in unique_combinations:
            unique_combinations.add(key)
            samples.append((query, answer))

# Sample with replacement to reach target (OPTIMIZED)
while len(samples) < target_count:
    word1 = random.choice(unique_words)
    sound1 = get_first_sound(word1)
    if not sound1 or sound1 not in words_by_sound:
        continue
    
    matching_words = [w for w in words_by_sound[sound1] if w != word1]
    non_matching_words = []
    for sound in all_sounds:
        if sound != sound1:
            non_matching_words.extend(words_by_sound[sound])
    
    if not matching_words or not non_matching_words:
        continue
    
    template = random.choice(TEMPLATES)
    # CRITICAL FIX: word2 must NOT match the sound (must be a distractor)
    # word1 matches the sound, so answer is always word1
    word2 = random.choice(non_matching_words)
    
    query = template.format(sound=sound1, word1=word1, word2=word2)
    answer = word1  # word1 is correct because it matches the sound
    samples.append((query, answer))

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s3.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S3 Sound Matching: Generated {len(samples)} samples")
