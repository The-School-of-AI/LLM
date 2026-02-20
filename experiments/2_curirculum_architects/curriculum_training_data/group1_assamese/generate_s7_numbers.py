#!/usr/bin/env python3
"""
Generate Statement 7: Numeric Mastery (Ordinals)
Target: 114 unique pairs (max possible with current digit/ordinal maps)
Focus: Numbers, Ordinals.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_assamese.assamese_vocabulary import (
    NUMBERS_BASE,
    ORDINALS
)
from prompt_utils import format_qa_pair_hindi

TEMPLATES_SPELLING = [
    '"{num}" সংখ্যাটোৰ বানান কি?',
    '"{num}" - এই সংখ্যাটো আখৰেৰে লিখক।',
    '"{num}"ক আখৰেৰে কেনেকৈ লিখিব?',
]

TEMPLATES_ORDINAL = [
    '"{num}" ৰ ক্ৰমিক সংখ্যা কি?',
    '"{num}"ৰ ক্ৰমিক ৰূপটো কি?',
    '"{num}" নম্বৰ স্থানক কি বুলি কয়?',
]

def main():
    samples = []
    
    # Arabic digit -> Assamese word (1-20, 30, 40...100)
    digit_map = {
        "1": "এক", "2": "দুই", "3": "তিনি", "4": "চাৰি", "5": "পাঁচ",
        "6": "ছয়", "7": "সাত", "8": "আঠ", "9": "ন", "10": "দহ",
        "11": "এঘাৰ", "12": "বাৰ", "13": "তেৰ", "14": "চৈধ্য", "15": "পোন্ধৰ",
        "16": "ষোল্ল", "17": "সোতৰ", "18": "ওঠৰ", "19": "ঊনৈছ", "20": "বিছ",
        "30": "ত্ৰিছ", "40": "চল্লিছ", "50": "পঞ্চাছ", "60": "ষাঠি",
        "70": "সত্তৰ", "80": "আশী", "90": "নব্বৈ", "100": "এশ",
    }
    # Arabic digit -> Bengali-Assamese numeral (U+09E6–U+09EF)
    ASSAMESE_NUMERAL = {"0": "০", "1": "১", "2": "২", "3": "৩", "4": "৪", "5": "৫", "6": "৬", "7": "৭", "8": "৮", "9": "৯"}

    def to_assamese_numeral(s: str) -> str:
        return "".join(ASSAMESE_NUMERAL.get(c, c) for c in s)
    
    ordinal_map = {
        "1": "প্ৰথম", "2": "দ্বিতীয়", "3": "তৃতীয়", "4": "চতুৰ্থ", "5": "পঞ্চম",
        "6": "ষষ্ঠ", "7": "সপ্তম", "8": "অষ্টম", "9": "নৱম", "10": "দশম"
    }

    # Generate all unique (query, answer) pairs
    for digit, word in digit_map.items():
        num_display = to_assamese_numeral(digit)
        for template in TEMPLATES_SPELLING:
            query = template.format(num=num_display)
            samples.append((query, word))
    for digit, word in ordinal_map.items():
        num_display = to_assamese_numeral(digit)
        for template in TEMPLATES_ORDINAL:
            query = template.format(num=num_display)
            samples.append((query, word))

    random.shuffle(samples)

    output_file = os.path.join(os.path.dirname(__file__), "group1_s7.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for query, answer in samples:
            f.write(format_qa_pair_hindi(query, answer) + "\n")

    print(f"S7 Numeric: Generated {len(samples)} samples (all unique)")

if __name__ == "__main__":
    main()
