#!/usr/bin/env python3
"""
Generate Statement 1: Spelling (वर्तनी) questions
Target: 28,600 pairs (14.3% of 200,000)
"""

import os
import random
import sys

import regex  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.hindi_vocabulary import (  # noqa: E402
    EASY_WORDS_UNIQUE,
    HARD_WORDS_UNIQUE,
    MEDIUM_WORDS_UNIQUE,
)
from prompt_utils import format_qa_pair_hindi  # noqa: E402

# Expand word lists to reach target count
EASY_WORDS = EASY_WORDS_UNIQUE * 50
MEDIUM_WORDS = MEDIUM_WORDS_UNIQUE * 60
HARD_WORDS = HARD_WORDS_UNIQUE * 70

# Hindi question templates for spelling (only Hindi words, no English transliterations)
TEMPLATES = [
    '"{word}" की वर्तनी क्या है?',
    '"{word}" को कैसे लिखते हैं?',
    '"{word}" के अक्षर क्या हैं?',
    '"{word}" का वर्तनी बताइए?',
    '"{word}" शब्द की वर्तनी क्या है?',
    '"{word}" का सही वर्तनी क्या है?',
    '"{word}" शब्द को कैसे लिखा जाता है?',
    '"{word}" की वर्तनी लिखिए?',
    '"{word}" का वर्तनी क्या होता है?',
    '"{word}" शब्द का वर्तनी क्या है?',
    '"{word}" को कैसे वर्तनी करते हैं?',
    '"{word}" की सही वर्तनी बताइए?',
    '"{word}" का वर्तनी क्या है?',
    '"{word}" शब्द की वर्तनी बताइए?',
    '"{word}" को कैसे लिखा जाता है?',
    # Additional 10 templates for 200K
    '"{word}" की स्पेलिंग क्या है?',
    '"{word}" को किस तरह लिखते हैं?',
    '"{word}" में कौन कौन से अक्षर हैं?',
    '"{word}" का सही तरीके से वर्तनी बताइए?',
    '"{word}" शब्द को लिखने का तरीका क्या है?',
    '"{word}" की वर्तनी बताओ?',
    '"{word}" कैसे लिखा जाता है?',
    '"{word}" का स्पेलिंग बताइए?',
    '"{word}" को वर्तनी करो?',
    '"{word}" शब्द की स्पेलिंग क्या है?',
]


def get_hindi_characters(word: str) -> list[str]:
    """
    Break down a Hindi word into its constituent Unicode characters.
    Each Unicode character (consonant, vowel, matra, nukta) is separate.
    This matches the spelling format where each character is shown separately.

    Example: "पानी" → ['प', 'ा', 'न', 'ी'] (4 Unicode chars)
    Example: "मूली" → ['म', 'ू', 'ल', 'ी'] (4 Unicode chars)
    Example: "जड़" → ['ज', 'ड', '़'] (3 Unicode chars - nukta is separate)
    Example: "कमल" → ['क', 'म', 'ल'] (3 Unicode chars)

    Used for: Spelling questions (S1, S8)
    """
    # Simply return each Unicode character separately
    # This matches the spelling format and token counting logic
    return list(word)


def get_hindi_grapheme_clusters(word: str) -> list[str]:
    """
    Get grapheme clusters for Hindi word (for counting/length/position).
    Uses regex library's \\X pattern (Unicode UAX#29 compliant).
    Each grapheme cluster = 1 अक्षर for counting/position questions.

    Example: "मुर्गी" → ['मु', 'र्गी'] (2 clusters)
    Example: "पानी" → ['पा', 'नी'] (2 clusters)
    Example: "कमल" → ['क', 'म', 'ल'] (3 clusters)
    Example: "विद्यालय" → ['वि', 'द्या', 'ल', 'य'] (4 clusters)

    Used for: Counting, length, and position questions (S2, S4, S7, S9, S10)
    """
    return regex.findall(r"\X", word)


def generate_spelling_answer(word: str) -> str:
    """Generate spelling answer as comma-separated characters"""
    chars = get_hindi_characters(word)
    return ", ".join(chars)


all_words = EASY_WORDS + MEDIUM_WORDS + HARD_WORDS
samples = []
target_count = 35000  # Increased from 28600 for 200K push

# Generate all unique combinations first
unique_combinations = {}
for word in set(all_words):  # Use unique words
    for template_idx, template in enumerate(TEMPLATES):
        query = template.format(word=word)
        answer = generate_spelling_answer(word)
        unique_combinations[(word, template_idx)] = (query, answer)

# Only use unique combinations - NO sampling with replacement
samples = list(unique_combinations.values())
unique_count = len(samples)

if unique_count < target_count:
    print(f"Warning: Only {unique_count} unique combinations (target: {target_count})")
else:
    samples = samples[:target_count]

# Shuffle for randomness
random.shuffle(samples)

output_file = os.path.join(os.path.dirname(__file__), "group1_s1.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for query, answer in samples:
        f.write(format_qa_pair_hindi(query, answer) + "\n")

print(f"S1 Spelling: Generated {len(samples)} unique samples (target: {target_count})")
