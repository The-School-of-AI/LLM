#!/usr/bin/env python3
"""
Generate Statement 7: Numeric Mastery (Ordinals)
Target: 15,000 pairs
Focus: Numbers, Ordinals.
"""

import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt_utils import format_qa_pair_hindi

TEMPLATES_SPELLING = [
    '"{num}" সংখ্যাটোৰ বানান কি?',
    "{num} - এই সংখ্যাটো আখৰেৰে লিখক।",
    '"{num}"ক আখৰেৰে কেনেকৈ লিখিব?',
]

TEMPLATES_ORDINAL = [
    '"{num}" ৰ ক্ৰমিক সংখ্যা কি?',
    '"{num}"ৰ ক্ৰমিক ৰূপটো কি?',
    '"{num}" নম্বৰ স্থানক কি বুলি কয়?',
]


def main():
    samples = []
    target_count = 15000

    # Create a mapping of digit -> word if possible, but we mostly have words.
    # We can ask "What is spelling of X" where X is a digit.
    # Since NUMBERS_BASE are words ("এক"), we need digits.
    # Let's map indices to digits.

    # Mapping for NUMBERS_BASE (assuming ordered 1..19, then 20, 30...)
    # This is tricky because the list is mixed.
    # Let's define a safe map for generation.

    # Arabic digit -> Assamese word
    digit_map = {
        "1": "এক",
        "2": "দুই",
        "3": "তিনি",
        "4": "চাৰি",
        "5": "পাঁচ",
        "6": "ছয়",
        "7": "সাত",
        "8": "আঠ",
        "9": "ন",
        "10": "দহ",
        "11": "এঘাৰ",
        "12": "বাৰ",
        "13": "তেৰ",
        "14": "চৈধ্য",
        "15": "পোন্ধৰ",
        "16": "ষোল্ল",
        "17": "সোতৰ",
        "18": "ওঠৰ",
        "19": "ঊনৈছ",
        "20": "বিছ",
        "30": "ত্ৰিছ",
        "40": "চল্লিছ",
        "50": "পঞ্চাছ",
        "60": "ষাঠি",
        "70": "সত্তৰ",
        "80": "আশী",
        "90": "নব্বৈ",
        "100": "এশ",
    }
    # Arabic digit -> Bengali-Assamese numeral (U+09E6–U+09EF)
    ASSAMESE_NUMERAL = {
        "0": "০",
        "1": "১",
        "2": "২",
        "3": "৩",
        "4": "৪",
        "5": "৫",
        "6": "৬",
        "7": "৭",
        "8": "৮",
        "9": "৯",
    }

    def to_assamese_numeral(s: str) -> str:
        return "".join(ASSAMESE_NUMERAL.get(c, c) for c in s)

    ordinal_map = {
        "1": "প্ৰথম",
        "2": "দ্বিতীয়",
        "3": "তৃতীয়",
        "4": "চতুৰ্থ",
        "5": "পঞ্চম",
        "6": "ষষ্ঠ",
        "7": "সপ্তম",
        "8": "অষ্টম",
        "9": "নৱম",
        "10": "দশম",
    }

    while len(samples) < target_count:
        if random.random() < 0.6:
            # Number Spelling
            digit = random.choice(list(digit_map.keys()))
            word = digit_map[digit]
            template = random.choice(TEMPLATES_SPELLING)
            num_display = to_assamese_numeral(digit)
            query = template.format(num=num_display)
            answer = word
            samples.append((query, answer))
        else:
            # Ordinals
            digit = random.choice(list(ordinal_map.keys()))
            word = ordinal_map[digit]
            template = random.choice(TEMPLATES_ORDINAL)
            num_display = to_assamese_numeral(digit)
            query = template.format(num=num_display)
            answer = word
            samples.append((query, answer))

    random.shuffle(samples)
    samples = samples[:target_count]

    output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S7 Numeric: Generated {len(samples)} samples")


if __name__ == "__main__":
    main()
