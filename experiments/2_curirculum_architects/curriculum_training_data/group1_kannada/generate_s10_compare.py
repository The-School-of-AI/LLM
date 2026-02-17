#!/usr/bin/env python3
"""
Generate Statement 10: Word Comparison (ಪದ ಹೋಲಿಕೆ) questions - Kannada
Target: 11,000 pairs (5.5% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.generate_s1_spelling import (  # noqa: E402
    get_kannada_grapheme_clusters,
)
from group1_kannada.kannada_vocabulary import ALL_WORDS_UNIQUE  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Expand word list
ALL_WORDS = ALL_WORDS_UNIQUE * 30

# Question templates — varied phrasings for word comparison.
TEMPLATES_LONGER = [
    'ಯಾವ ಪದ ಉದ್ದವಾಗಿದೆ, "{word1}" ಅಥವಾ "{word2}"?',
    '"{word1}" ಮತ್ತು "{word2}" ನಲ್ಲಿ ಯಾವ ಪದ ಉದ್ದವಾಗಿದೆ?',
    'ಯಾವ ಪದ ಹೆಚ್ಚು ಉದ್ದವಾಗಿದೆ, "{word1}" ಅಥವಾ "{word2}"?',
    '"{word1}" ಮತ್ತು "{word2}" ನಲ್ಲಿ ಉದ್ದನೆಯ ಪದ ಯಾವುದು?',
    'ಅಕ್ಷರಗಳಲ್ಲಿ ಉದ್ದವಾದ ಪದ "{word1}" ಮತ್ತು "{word2}" ರಲ್ಲಿ ಯಾವುದು?',
]

TEMPLATES_SHORTER = [
    'ಯಾವ ಪದ ಕಿರಿದಾಗಿದೆ, "{word1}" ಅಥವಾ "{word2}"?',
    '"{word1}" ಮತ್ತು "{word2}" ನಲ್ಲಿ ಯಾವ ಪದ ಕಿರಿದಾಗಿದೆ?',
    'ಯಾವ ಪದ ಹೆಚ್ಚು ಕಿರಿದಾಗಿದೆ, "{word1}" ಅಥವಾ "{word2}"?',
    '"{word1}" ಮತ್ತು "{word2}" ನಲ್ಲಿ ಕಿರಿದಾದ ಪದ ಯಾವುದು?',
    'ಅಕ್ಷರಗಳಲ್ಲಿ ಕಿರಿದಾದ ಪದ "{word1}" ಮತ್ತು "{word2}" ರಲ್ಲಿ ಯಾವುದು?',
]

# Pre-compute word lengths
unique_words = list(set(ALL_WORDS))
word_lengths = {}
for word in unique_words:
    clusters = get_kannada_grapheme_clusters(word)
    word_lengths[word] = len(clusters)


def get_word_length(word: str) -> int:
    """Get the length of a word in grapheme clusters (cached)"""
    return word_lengths.get(word, 0)


samples = []
target_count = 13000
unique_combinations = set()

word_list = unique_words
max_pairs_to_generate = min(
    target_count * 2, len(word_list) * (len(word_list) - 1) // 2
)
pairs_generated = 0

for i, word1 in enumerate(word_list):
    if pairs_generated >= max_pairs_to_generate:
        break
    for j, word2 in enumerate(word_list):
        if i >= j:
            continue
        if pairs_generated >= max_pairs_to_generate:
            break

        len1 = get_word_length(word1)
        len2 = get_word_length(word2)

        if len1 == len2:
            continue

        if len1 > len2:
            longer_word = word1
            shorter_word = word2
        else:
            longer_word = word2
            shorter_word = word1

        for template_idx, template in enumerate(TEMPLATES_LONGER):
            query = template.format(word1=word1, word2=word2)
            answer = longer_word
            key = (word1, word2, template_idx, "longer")
            if key not in unique_combinations:
                unique_combinations.add(key)
                samples.append((query, answer))

        for template_idx, template in enumerate(TEMPLATES_SHORTER):
            query = template.format(word1=word1, word2=word2)
            answer = shorter_word
            key = (word1, word2, template_idx, "shorter")
            if key not in unique_combinations:
                unique_combinations.add(key)
                samples.append((query, answer))

        pairs_generated += 1

seen_qa = set((q, a) for q, a in samples)
no_progress_limit = 50000
no_progress = 0
while len(samples) < target_count and no_progress < no_progress_limit:
    word1 = random.choice(word_list)
    word2 = random.choice([w for w in word_list if w != word1])

    len1 = get_word_length(word1)
    len2 = get_word_length(word2)

    if len1 == len2:
        continue

    if len1 > len2:
        longer_word = word1
        shorter_word = word2
    else:
        longer_word = word2
        shorter_word = word1

    if random.random() < 0.5:
        template = random.choice(TEMPLATES_LONGER)
        answer = longer_word
    else:
        template = random.choice(TEMPLATES_SHORTER)
        answer = shorter_word

    query = template.format(word1=word1, word2=word2)
    if (query, answer) not in seen_qa:
        seen_qa.add((query, answer))
        samples.append((query, answer))
        no_progress = 0
    else:
        no_progress += 1

random.shuffle(samples)
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S10 Word Comparison (Kannada): Generated {len(samples)} samples")
