#!/usr/bin/env python3
"""
Generate Statement 6: Classification (वर्गीकरण) questions
Target: 20,000 pairs (10% of 200,000)
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.hindi_vocabulary import CLASSIFICATION_CATEGORIES  # noqa: E402
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Question templates
TEMPLATES = [
    '"{word}" एक व्यक्ति, जानवर या वस्तु है?',
    '"{word}" क्या है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" शब्द किस श्रेणी में आता है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" को किस श्रेणी में रखा जा सकता है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस प्रकार की चीज़ है, व्यक्ति, जानवर या वस्तु?',
    'बताइए "{word}" किस वर्ग का है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस श्रेणी में आता है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" को कैसे वर्गीकृत करें, व्यक्ति, जानवर या वस्तु?',
    '"{word}" का वर्गीकरण क्या है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस वर्ग से संबंधित है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" शब्द को वर्गीकृत करें, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस प्रकार है, व्यक्ति, जानवर या वस्तु?',
    # Additional 10 templates
    'बताओ "{word}" क्या है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस वर्ग में है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" को कहां रखें, व्यक्ति, जानवर या वस्तु?',
    '"{word}" की श्रेणी क्या है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" शब्द का वर्ग बताइए, व्यक्ति, जानवर या वस्तु?',
    '"{word}" कौन सी श्रेणी का है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" का प्रकार क्या है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" किस समूह में आता है, व्यक्ति, जानवर या वस्तु?',
    '"{word}" को किस वर्ग में डालें, व्यक्ति, जानवर या वस्तु?',
    'बताइए "{word}" क्या होता है, व्यक्ति, जानवर या वस्तु?',
]


def classify_word(word: str) -> str:
    """Classify a word into category"""
    for category, word_list in CLASSIFICATION_CATEGORIES.items():
        if word in word_list:
            return category
    # Default to वस्तु if not found
    return "वस्तु"


samples = []
target_count = 20000
all_words = []
for word_list in CLASSIFICATION_CATEGORIES.values():
    all_words.extend(word_list)

# Expand word list
all_words = all_words * 20
unique_combinations = {}

# Generate samples
for word in set(all_words):
    category = classify_word(word)
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = category
        key = (word, template_idx)
        if key not in unique_combinations:
            unique_combinations[key] = (query, answer)

# Generate unique samples only - NO sampling with replacement
samples = list(unique_combinations.values())
unique_count = len(samples)

# If we have fewer unique combinations than target, generate warning
if unique_count < target_count:
    print(
        f"Warning: Only {unique_count} unique combinations possible (target: {target_count})"
    )
    print("  Consider adding more words or templates to reach target")
else:
    # If we have more than target, take only what we need
    samples = samples[:target_count]

random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s6.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(
    f"S6 Classification: Generated {len(samples)} unique samples (target: {target_count})"
)
