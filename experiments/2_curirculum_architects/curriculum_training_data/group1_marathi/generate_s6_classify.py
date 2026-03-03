#!/usr/bin/env python3
"""
Generate Statement 6: Classification (वर्गीकरण) questions
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_marathi.marathi_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402
from prompt_utils import format_qa_pair_marathi  # noqa: E402

# Question templates
TEMPLATES = [
    '"{word}" हा व्यक्ती, प्राणी किंवा वस्तू आहे?',
    '"{word}" काय आहे, व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" शब्द कोणत्या श्रेणीत येतो, व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" कोणत्या श्रेणीत ठेवता येईल, व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" कोणत्या प्रकारची गोष्ट आहे, व्यक्ती, प्राणी किंवा वस्तू?',
    'दिलेल्या पर्यायांपैकी "{word}" काय आहे: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" या शब्दाचे वर्गीकरण करा: व्यक्ती, प्राणी किंवा वस्तू?',
    'खालीलपैकी "{word}" कशात मोडते: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" ही संज्ञा कशासाठी वापरली जाते: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" कोणत्या वर्गात विभागता येईल: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" साठी योग्य वर्ग निवडा: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" हे व्यक्ती, प्राणी किंवा वस्तू यांपैकी काय आहे?',
    '"{word}" चा संबंध कशाशी आहे: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" कशाचे उदाहरण आहे: व्यक्ती, प्राणी किंवा वस्तू?',
    '"{word}" कोणत्या गटात येते: व्यक्ती, प्राणी किंवा वस्तू?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    # Default to वस्तू if not found
    return "वस्तू"


samples = []
target_count = 20000
all_words_set = set()
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words_set.update(word_list)

unique_combinations = []

# Generate all possible unique combinations
all_words_list = list(all_words_set)
random.shuffle(all_words_list)

for word in all_words_list:
    category = classify_word(word)
    # Shuffle templates for each word to spread them out
    current_templates = list(enumerate(TEMPLATES))
    random.shuffle(current_templates)

    for template_idx, template in current_templates:
        query = template.format(word=word)
        answer = category
        unique_combinations.append((query, answer))
        if len(unique_combinations) >= target_count:
            break
    if len(unique_combinations) >= target_count:
        break

samples = unique_combinations

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_marathi(query, answer) + "\n")

print(f"S6 Classification: Generated {len(samples)} samples")
