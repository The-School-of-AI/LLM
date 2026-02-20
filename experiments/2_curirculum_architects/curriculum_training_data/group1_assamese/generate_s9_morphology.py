#!/usr/bin/env python3
"""
Generate Statement 9: Morphology (Roots + Suffixes)
Target: All unique pairs possible (~9,498)
Focus: Root word extraction and suffix identification.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import VERBS, SUFFIXES
# Use EASY_OBJECTS/EASY_PEOPLE if available, else OBJECTS/PEOPLE (cleaned vocab)
from group1_assamese.assamese_vocabulary import EASY_OBJECTS, EASY_PEOPLE
from prompt_utils import format_qa_pair_hindi

# Define specific suffix categories based on SUFFIXES dict
HUMAN_SUFFIXES = ["সকল", "হঁত", "জন", "জনী"]
# Note: "বোৰ", "বিলাক", "মখা" are in 'plural' but mostly for objects/general. 
# "টো", "টি" can be both but safe for objects. 
# "খন", "ডাল", etc are classifiers for objects.
OBJECT_SUFFIXES = ["বোৰ", "বিলাক", "মখা", "টো", "টি", "খন", "খনি", "ডাল", "চটা", "গছ", "পাত", "জোপা"]
CASE_SUFFIXES = ["ৰ", "লৈ", "ত", "ৰে", "ৰপৰা", "লৈকে", "ক"]

# Define allowed combinations
# (Word List, List of allowed suffix lists)
COMBINATIONS = [
    (EASY_PEOPLE, [HUMAN_SUFFIXES, CASE_SUFFIXES, ["বোৰ", "বিলাক"]]), # People can take plural/case
    (EASY_OBJECTS, [OBJECT_SUFFIXES, CASE_SUFFIXES]), # Objects avoid human suffixes
    (VERBS, [HUMAN_SUFFIXES, OBJECT_SUFFIXES, CASE_SUFFIXES]) # Nominalized verbs can be flexible
]

TEMPLATES_ROOT = [
    '"{inflected}" শব্দটোৰ মূল শব্দ কি?',
    '"{inflected}"ৰ মূল কি?',
    '"{inflected}" - ইয়াৰ মূল শব্দটো বাছনি কৰক।',
]

TEMPLATES_SUFFIX = [
    '"{inflected}" শব্দটোত কি বিভক্তি/প্ৰত্যয় যোগ হৈছে?',
    '"{inflected}"ৰ শেষত কি যোগ হৈছে?',
    '"{root}"ৰ লগত "{suffix}" যোগ কৰিলে কি হ\'ব?', # Synthesis
]

def main():
    samples = []

    for word_list, allowed_suffix_groups in COMBINATIONS:
        all_suffixes = []
        for g in allowed_suffix_groups:
            all_suffixes.extend(g)
        for root in word_list:
            for suffix in all_suffixes:
                inflected = root + suffix
                # Identify Root (3 templates)
                for template in TEMPLATES_ROOT:
                    query = template.format(inflected=inflected)
                    samples.append((query, root))
                # Identify Suffix (2 templates)
                for template in TEMPLATES_SUFFIX[:2]:
                    query = template.format(inflected=inflected)
                    samples.append((query, suffix))
                # Synthesis (1 template)
                query = TEMPLATES_SUFFIX[2].format(root=root, suffix=suffix)
                samples.append((query, inflected))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s9.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S9 Morphology: Generated {len(samples)} samples (all unique)")

if __name__ == "__main__":
    main()
