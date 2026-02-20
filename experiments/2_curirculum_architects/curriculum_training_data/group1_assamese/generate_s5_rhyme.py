#!/usr/bin/env python3
"""
Generate Statement 5: Rhyming (Verb endings)
Target: 15,000 pairs (no exact duplicates, quality distractors)
Focus: Matching words with same ending sounds/suffixes.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    ALL_WORDS_UNIQUE,
    RHYMING_PAIRS,
    RHYMING_GROUPS,
)
from prompt_utils import format_qa_pair_hindi

TEMPLATES_PAIR = [
    '"{word}"ৰ লগত ছন্দ মিল থকা শব্দ এটা কোৱা?',
    '"{word}"ৰ এটা ছন্দোবদ্ধ শব্দ কি?',
    '"{word}"ৰ লগত মিল থকা শব্দ কি?',
]

TEMPLATES_CHOICE = [
    '"{word}"ৰ লগত কোনটো শব্দৰ ছন্দ মিলে: "{option1}" নে "{option2}"?',
    '"{word}"ৰ লগত ছন্দ মিল থকা শব্দটো বাছনি কৰক: "{option1}", "{option2}"।',
]

def _build_rhyme_sets():
    """Build set of words that rhyme with each word (for quality distractor exclusion)."""
    rhyme_with = {}
    for w, r in RHYMING_PAIRS.items():
        rhyme_with.setdefault(w, set()).add(r)
        rhyme_with.setdefault(r, set()).add(w)
    for group in RHYMING_GROUPS.values():
        if len(group) >= 2:
            gset = set(group)
            for w in group:
                rhyme_with.setdefault(w, set()).update(gset - {w})
    return rhyme_with


def main():
    samples = []
    seen = set()  # Track (query, answer) to avoid exact duplicates
    target_count = 15000

    pairs = list(RHYMING_PAIRS.items())
    group_pairs = []
    for group in RHYMING_GROUPS.values():
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(len(group)):
                    if i != j:
                        group_pairs.append((group[i], group[j]))
    all_pairs = pairs + group_pairs
    rhyme_with = _build_rhyme_sets()
    all_words_set = set(ALL_WORDS_UNIQUE)
    max_attempts = target_count * 25
    attempts = 0

    while len(samples) < target_count and attempts < max_attempts:
        attempts += 1
        word, rhyme = random.choice(all_pairs)

        # Strategy 1: Direct question (40%)
        if random.random() < 0.4:
            template = random.choice(TEMPLATES_PAIR)
            query = template.format(word=word)
            answer = rhyme
            key = (query, answer)
            if key not in seen:
                seen.add(key)
                samples.append((query, answer))

        # Strategy 2: Multiple Choice (60%) - use non-rhyming distractors for quality
        else:
            exclude = rhyme_with.get(word, set()) | {word, rhyme}
            valid_distractors = [w for w in all_words_set if w not in exclude]
            if not valid_distractors:
                continue
            distractor = random.choice(valid_distractors)
            template = random.choice(TEMPLATES_CHOICE)
            opts = [rhyme, distractor]
            random.shuffle(opts)
            opt1, opt2 = opts[0], opts[1]
            query = template.format(word=word, option1=opt1, option2=opt2)
            answer = rhyme
            key = (query, answer)
            if key not in seen:
                seen.add(key)
                samples.append((query, answer))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s5.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S5 Rhyming: Generated {len(samples)} samples (no duplicates)")

if __name__ == "__main__":
    main()
