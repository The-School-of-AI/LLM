#!/usr/bin/env python3
"""
Generate Statement 10: Synonyms & Antonyms
Target: 30,000 pairs
Focus: Semantic relationship mapping.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import ANTONYMS, SYNONYMS
from prompt_utils import format_qa_pair_hindi

TEMPLATES_SYNONYM = [
    '"{word}"ৰ সমাৰ্থক শব্দ কি?',
    '"{word}"ৰ এটা প্ৰতিশব্দ কি?',
    '"{word}"ৰ প্ৰতিশব্দটো ক’ব পাৰিব নেকি?',
    '"{word}"ৰ আন এটা অৰ্থ কি হ\'ব পাৰে?',
    '"{word}" শব্দটোৰ সলনি আন কি শব্দ ব্যৱহাৰ কৰিব পাৰি?',
    '"{word}"ক আন কি নামেৰে জনা যায়?',
    '"{word}"ৰ নিচিনা অৰ্থ থকা এটা শব্দ কওক?',
]

TEMPLATES_ANTONYM = [
    '"{word}"ৰ বিপৰীত শব্দ কি?',
    '"{word}"ৰ বিপৰীতাৰ্থক শব্দটো কি?',
    # Conversational
    '"{word}"ৰ ওলোটা অৰ্থ কি?',
    '"{word}"ৰ ওলোটা শব্দটো কি হ’ব?',
    '"{word}"ৰ বিপৰীতে কি বহিব?',
    # Simple
    '"{word}"ৰ বিপৰীতটো কি?',
]


def main():
    samples = []
    target_count = 30000

    syn_keys = list(SYNONYMS.keys())
    ant_keys = list(ANTONYMS.keys())

    while len(samples) < target_count:
        if random.random() < 0.6:
            # Synonyms
            word = random.choice(syn_keys)
            syn_list = SYNONYMS[word]
            syn = random.choice(syn_list)

            if random.random() < 0.7:
                # Ask for synonym
                template = random.choice(TEMPLATES_SYNONYM[:3])
                query = template.format(word=word)
                answer = syn
            else:
                # Verification
                template = TEMPLATES_SYNONYM[3]
                query = template.format(word=word, syn=syn)
                answer = "হয়"  # Yes

            samples.append((query, answer))

        else:
            # Antonyms
            word = random.choice(ant_keys)
            ant = ANTONYMS[word]

            template = random.choice(TEMPLATES_ANTONYM)
            query = template.format(word=word)
            answer = ant
            samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S10 Semantics: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
