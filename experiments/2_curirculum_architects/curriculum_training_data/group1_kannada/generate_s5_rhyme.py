#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (ಪ್ರಾಸ) questions - Kannada
One question per (question_word): no duplicate stems with only options changed.
Target: up to 2 * len(RHYMING_PAIRS); expand vocabulary to reach 20,000.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_kannada.kannada_vocabulary import ALL_WORDS_UNIQUE, RHYMING_PAIRS  # noqa: E402
from prompt_utils import format_qa_pair_kannada  # noqa: E402

# Pool of non-rhyming distractors (unique words)
unique_words = list(set(ALL_WORDS_UNIQUE))

# One canonical template to avoid same stem with different options
TEMPLATE = '"{word}" ಪದವಿಗೆ ಪ್ರಾಸ ಆಗುವ ಪದ ಯಾವುದು, "{option1}" ಅಥವಾ "{option2}"?'

samples = []
seen_question_word = set()  # at most one question per word (the word we ask rhyme for)

def add_one_rhyme_question(question_word: str, answer_word: str) -> None:
    """Add exactly one question for this (question_word, answer_word) rhyme pair."""
    if question_word in seen_question_word:
        return
    non_rhyming = [w for w in unique_words if w != question_word and w != answer_word]
    if not non_rhyming:
        return
    distractor = random.choice(non_rhyming)
    option1, option2 = random.sample([answer_word, distractor], 2)
    query = TEMPLATE.format(word=question_word, option1=option1, option2=option2)
    seen_question_word.add(question_word)
    samples.append((query, answer_word))

# Forward: one question per word (word -> rhyme_word)
for word, rhyme_word in RHYMING_PAIRS.items():
    add_one_rhyme_question(word, rhyme_word)

# Reverse: one question per rhyme_word (rhyme_word -> word)
for word, rhyme_word in RHYMING_PAIRS.items():
    add_one_rhyme_question(rhyme_word, word)

random.shuffle(samples)
# Cap at target; dataset combiner uses what we produce
target_count = 20000
samples = samples[:target_count]

output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_kannada(query, answer) + "\n")

print(f"S5 Rhyming (Kannada): Generated {len(samples)} samples")
