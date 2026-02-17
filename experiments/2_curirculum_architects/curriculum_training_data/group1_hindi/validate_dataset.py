#!/usr/bin/env python3
"""
Deep validation of generated Hindi dataset
Checks both uniqueness and correctness of each statement type
"""

import os
import re
import sys

import regex

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from group1_hindi.hindi_vocabulary import (  # noqa: E402
    CLASSIFICATION_CATEGORIES,
    RHYMING_PAIRS,
)


def get_hindi_grapheme_clusters(word: str) -> list[str]:
    """Get grapheme clusters for Hindi word"""
    return regex.findall(r"\X", word)


def get_hindi_characters(word: str) -> list[str]:
    """Get Unicode characters for spelling"""
    return list(word)


def check_rhyming(word1: str, word2: str) -> bool:
    """Check if two words rhyme (simple check: last 2+ characters match)"""
    if len(word1) < 2 or len(word2) < 2:
        return False
    # Check if words end with same sound (last 1-2 characters)
    return word1[-1] == word2[-1] or word1[-2:] == word2[-2:]


def parse_qa_file(filename: str) -> list[tuple[str, str]]:
    """Parse a group1_sX.txt file and extract Q&A pairs"""
    pairs = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "Q? A।"
            match = re.match(r"(.+?)\? (.+?)।", line)
            if match:
                query, answer = match.groups()
                pairs.append((query.strip(), answer.strip()))
    return pairs


def validate_s1_spelling(pairs: list[tuple[str, str]]) -> dict:
    """Validate S1: Spelling questions"""
    errors = []
    for query, answer in pairs:
        # Extract word from query (in quotes)
        word_match = re.search(r'"(.+?)"', query)
        if not word_match:
            errors.append(f"Could not extract word from: {query}")
            continue

        word = word_match.group(1)
        chars = get_hindi_characters(word)
        expected_answer = ", ".join(chars)

        if answer != expected_answer:
            errors.append(f"Word: {word}, Expected: {expected_answer}, Got: {answer}")

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s2_letter_position(pairs: list[tuple[str, str]]) -> dict:
    """Validate S2: Letter Position questions"""
    errors = []
    for query, answer in pairs:
        # Extract word from query
        word_match = re.search(r'"(.+?)"', query)
        if not word_match:
            continue

        word = word_match.group(1)
        clusters = get_hindi_grapheme_clusters(word)

        # Extract position from query
        positions = {
            "पहला": 0,
            "दूसरा": 1,
            "तीसरा": 2,
            "चौथा": 3,
            "पांचवां": 4,
            "छठा": 5,
            "सातवां": 6,
            "आठवां": 7,
            "नौवां": 8,
            "दसवां": 9,
        }

        position_idx = None
        for pos_name, idx in positions.items():
            if pos_name in query:
                position_idx = idx
                break

        if position_idx is not None and position_idx < len(clusters):
            expected = clusters[position_idx]
            if answer != expected:
                errors.append(
                    f"Word: {word}, Position: {position_idx+1}, Expected: {expected}, Got: {answer}"
                )

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s3_sound_matching(pairs: list[tuple[str, str]]) -> dict:
    """Validate S3: Sound Matching questions"""
    errors = []
    for query, answer in pairs:
        # Extract sound and words from query
        sound_match = re.search(r"/(.+?)/", query)
        word_matches = re.findall(r'"(.+?)"', query)

        if not sound_match or len(word_matches) != 2:
            continue

        sound = sound_match.group(1)
        word1, word2 = word_matches

        # Check if answer starts with the sound
        if not answer.startswith(sound):
            errors.append(f"Sound: {sound}, Answer: {answer} doesn't start with sound")

        # Check if answer is one of the two words
        if answer not in [word1, word2]:
            errors.append(f"Answer: {answer} not in options: {word1}, {word2}")

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s4_count(pairs: list[tuple[str, str]]) -> dict:
    """Validate S4: Letter Count questions"""
    errors = []
    for query, answer in pairs:
        word_match = re.search(r'"(.+?)"', query)
        if not word_match:
            continue

        word = word_match.group(1)
        clusters = get_hindi_grapheme_clusters(word)
        expected_count = str(len(clusters))

        if answer != expected_count:
            errors.append(
                f"Word: {word}, Expected count: {expected_count}, Got: {answer}"
            )

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s5_rhyming(pairs: list[tuple[str, str]]) -> dict:
    """Validate S5: Rhyming questions"""
    errors = []
    rhyme_quality = []
    pair_set = set(RHYMING_PAIRS)

    for query, answer in pairs:
        # Extract words from query
        word_matches = re.findall(r'"(.+?)"', query)
        if len(word_matches) < 2:
            continue

        # First word is the target, answer should rhyme with it
        target_word = word_matches[0]

        # Check if answer rhymes with target
        if not check_rhyming(target_word, answer):
            errors.append(f"Target: {target_word}, Answer: {answer} - doesn't rhyme")

        # Check if answer exists in known rhyming data.
        is_valid_pair = (target_word, answer) in pair_set or (
            answer,
            target_word,
        ) in pair_set

        rhyme_quality.append(is_valid_pair)

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "valid_pairs": sum(rhyme_quality),
        "valid_pair_rate": (
            sum(rhyme_quality) / len(rhyme_quality) * 100 if rhyme_quality else 0
        ),
        "sample_errors": errors[:5],
    }


def validate_s6_classification(pairs: list[tuple[str, str]]) -> dict:
    """Validate S6: Classification questions"""
    errors = []
    for query, answer in pairs:
        word_match = re.search(r'"(.+?)"', query)
        if not word_match:
            continue

        word = word_match.group(1)

        # Check if word exists in the category
        found = False
        for category, word_list in CLASSIFICATION_CATEGORIES.items():
            if word in word_list and answer == category:
                found = True
                break

        if not found:
            # Check if answer is valid category but word might not be in list
            if answer not in CLASSIFICATION_CATEGORIES:
                errors.append(f"Word: {word}, Invalid category: {answer}")

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s7_position(pairs: list[tuple[str, str]]) -> dict:
    """Validate S7: Position of Letter questions"""
    errors = []
    for query, answer in pairs:
        char_matches = re.findall(r'"(.+?)"', query)

        if len(char_matches) < 2:
            continue

        word = char_matches[0]
        char = char_matches[1]
        clusters = get_hindi_grapheme_clusters(word)

        # Find position of character
        try:
            pos = clusters.index(char) + 1  # 1-indexed
            position_names = {
                1: "पहला",
                2: "दूसरा",
                3: "तीसरा",
                4: "चौथा",
                5: "पांचवां",
                6: "छठा",
                7: "सातवां",
                8: "आठवां",
                9: "नौवां",
                10: "दसवां",
            }

            expected = position_names.get(pos, str(pos))
            # Answer can be word form or numeric
            if answer != expected and answer != str(pos):
                errors.append(
                    f"Word: {word}, Char: {char}, Position: {pos}, Expected: {expected} or {pos}, Got: {answer}"
                )
        except ValueError:
            errors.append(f"Word: {word}, Char: {char} not found in word")

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s9_last_letter(pairs: list[tuple[str, str]]) -> dict:
    """Validate S9: Last Letter questions"""
    errors = []
    for query, answer in pairs:
        word_match = re.search(r'"(.+?)"', query)
        if not word_match:
            continue

        word = word_match.group(1)
        clusters = get_hindi_grapheme_clusters(word)
        expected_last = clusters[-1] if clusters else ""

        if answer != expected_last:
            errors.append(
                f"Word: {word}, Expected last: {expected_last}, Got: {answer}"
            )

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def validate_s10_comparison(pairs: list[tuple[str, str]]) -> dict:
    """Validate S10: Word Comparison questions"""
    errors = []
    for query, answer in pairs:
        word_matches = re.findall(r'"(.+?)"', query)
        if len(word_matches) < 2:
            continue

        word1, word2 = word_matches[0], word_matches[1]
        len1 = len(get_hindi_grapheme_clusters(word1))
        len2 = len(get_hindi_grapheme_clusters(word2))

        # Determine which should be the answer based on query
        if "लंबा" in query or "बड़ा" in query:  # asking for longer
            expected = word1 if len1 > len2 else word2
        elif "छोटा" in query:  # asking for shorter
            expected = word1 if len1 < len2 else word2
        else:
            continue

        if answer != expected:
            errors.append(
                f"Words: {word1}({len1}) vs {word2}({len2}), Expected: {expected}, Got: {answer}"
            )

    return {
        "total": len(pairs),
        "errors": len(errors),
        "error_rate": len(errors) / len(pairs) * 100 if pairs else 0,
        "sample_errors": errors[:5],
    }


def main():
    base_path = os.path.dirname(__file__)

    print("=" * 80)
    print("DEEP VALIDATION OF HINDI DATASET")
    print("=" * 80)

    validators = {
        "S1 (Spelling)": ("group1_s1.txt", validate_s1_spelling),
        "S2 (Letter Position)": ("group1_s2.txt", validate_s2_letter_position),
        "S3 (Sound Matching)": ("group1_s3.txt", validate_s3_sound_matching),
        "S4 (Letter Count)": ("group1_s4.txt", validate_s4_count),
        "S5 (Rhyming)": ("group1_s5.txt", validate_s5_rhyming),
        "S6 (Classification)": ("group1_s6.txt", validate_s6_classification),
        "S7 (Position)": ("group1_s7.txt", validate_s7_position),
        "S9 (Last Letter)": ("group1_s9.txt", validate_s9_last_letter),
        "S10 (Comparison)": ("group1_s10.txt", validate_s10_comparison),
    }

    all_valid = True

    for name, (filename, validator) in validators.items():
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            print(f"\n❌ {name}: File not found: {filename}")
            continue

        pairs = parse_qa_file(filepath)
        result = validator(pairs)

        status = "✅" if result["errors"] == 0 else "⚠️"
        print(f"\n{status} {name}")
        print(f"   Total pairs: {result['total']:,}")
        print(f"   Errors: {result['errors']:,} ({result['error_rate']:.2f}%)")

        if "valid_pairs" in result:
            print(
                f"   Valid pairs: {result['valid_pairs']:,} ({result['valid_pair_rate']:.1f}%)"
            )

        if result["errors"] > 0:
            all_valid = False
            print("   Sample errors:")
            for error in result["sample_errors"]:
                print(f"     • {error}")

    print("\n" + "=" * 80)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED! Dataset is correct and valid.")
    else:
        print("⚠️  SOME VALIDATIONS FAILED. Review errors above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
