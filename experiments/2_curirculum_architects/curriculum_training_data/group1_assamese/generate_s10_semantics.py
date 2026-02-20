#!/usr/bin/env python3
"""
Generate Statement 10: Synonyms & Antonyms
Target: All unique pairs possible (~634)
Focus: Semantic relationship mapping.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    SYNONYMS,
    ANTONYMS
)
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

# Yes/no verification: "Is {syn} another meaning of {word}?" → Answer: হয়
TEMPLATES_SYNONYM_VERIFY = [
    '"{word}"ৰ আন এটা অৰ্থ "{syn}" হ\'ব পাৰে নেকি?',
    '"{syn}" "{word}"ৰ সমাৰ্থক শব্দ হয় নেকি?',
    '"{syn}" "{word}"ৰ প্ৰতিশব্দ হয় নেকি?',
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

    # Synonyms: all (word, syn) pairs × all templates
    for word, syn_list in SYNONYMS.items():
        for syn in syn_list:
            for template in TEMPLATES_SYNONYM:
                query = template.format(word=word)
                samples.append((query, syn))
            for template in TEMPLATES_SYNONYM_VERIFY:
                query = template.format(word=word, syn=syn)
                samples.append((query, "হয়"))

    # Antonyms: all (word, ant) pairs × all templates
    for word, ant in ANTONYMS.items():
        for template in TEMPLATES_ANTONYM:
            query = template.format(word=word)
            samples.append((query, ant))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s10.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S10 Semantics: Generated {len(samples)} samples (all unique)")

if __name__ == "__main__":
    main()
