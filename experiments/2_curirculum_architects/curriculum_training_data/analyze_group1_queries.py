#!/usr/bin/env python3
"""
Analyze group1.json to categorize queries by the 10 statement types from group1_plan.md
"""

import json
import re
from collections import Counter, defaultdict


def categorize_query(query: str) -> str:
    """Categorize a prompt into one of the 10 statement types."""
    query_lower = query.lower()

    # Statement 1: Spelling
    spelling_patterns = [
        r"spell\s+['\"]",
        r"what is the spelling of",
        r"write the spelling of",
        r"can you spell",
        r"what's the spelling of",
        r"how do you spell",
        r"tell me the spelling of",
        r"give me the spelling of",
        r"what is .* spelled as",
    ]
    for pattern in spelling_patterns:
        if re.search(pattern, query_lower):
            return "Statement 1: Spelling"

    # Statement 2: Letter at Position
    position_patterns = [
        r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+letter",
        r"(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\s+letter",
        r"letter.*(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)",
        r"letter.*(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)",
    ]
    for pattern in position_patterns:
        if re.search(pattern, query_lower):
            return "Statement 2: Letter at Position"

    # Statement 3: Sound Matching
    sound_patterns = [
        r"starts with the sound",
        r"begins with sound",
        r"starts with sound",
        r"pick the word that begins with sound",
        r"which word starts with",
    ]
    for pattern in sound_patterns:
        if re.search(pattern, query_lower):
            return "Statement 3: Sound Matching"

    # Statement 4: Letter Count
    count_patterns = [
        r"how many (alphabets|letters) are (there in|in)",
        r"count the letters in",
        r"letter count of",
        r"how many letters",
        r"what is the length of word",
    ]
    for pattern in count_patterns:
        if re.search(pattern, query_lower):
            return "Statement 4: Letter Count"

    # Statement 5: Rhyming
    rhyming_patterns = [
        r"what word rhymes with",
        r"tell me a word that rhymes with",
        r"give me a rhyming word for",
        r"find a rhyme for",
        r"rhymes with",
    ]
    for pattern in rhyming_patterns:
        if re.search(pattern, query_lower):
            return "Statement 5: Rhyming"

    # Statement 6: Classification
    classification_patterns = [
        r"is .* a (person|animal|thing)",
        r"classify .* as (person|animal|thing)",
        r"what is .* - a (person|animal|thing)",
        r"tell me if .* is a (person|animal|thing)",
        r"what category is .* - (person|animal|thing)",
    ]
    for pattern in classification_patterns:
        if re.search(pattern, query_lower):
            return "Statement 6: Classification"

    # Statement 7: Position of Letter
    position_letter_patterns = [
        r"at what location does letter",
        r"where is the letter",
        r"what position is .* in",
        r"find the position of letter",
        r"at what positions does",
    ]
    for pattern in position_letter_patterns:
        if re.search(pattern, query_lower):
            return "Statement 7: Position of Letter"

    # Statement 8: Number Spelling
    number_patterns = [
        r"spell\s+\d+",
        r"spelling of\s+\d+",
        r"spell\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)",
        r"how do you spell\s+\d+",
        r"how do you spell\s+(one|two|three|four|five|six|seven|eight|nine|ten)",
    ]
    for pattern in number_patterns:
        if re.search(pattern, query_lower):
            return "Statement 8: Number Spelling"

    # Statement 9: Last Letter
    last_letter_patterns = [
        r"what letter does .* end with",
        r"what is the last letter of",
        r"which letter does .* end with",
        r"tell me the ending letter of",
        r"find the final letter in",
    ]
    for pattern in last_letter_patterns:
        if re.search(pattern, query_lower):
            return "Statement 9: Last Letter"

    # Statement 10: Word Comparison
    comparison_patterns = [
        r"which word is longer",
        r"is .* longer than",
        r"compare the length of",
        r"which is the longer word",
        r"tell me which word has more letters",
    ]
    for pattern in comparison_patterns:
        if re.search(pattern, query_lower):
            return "Statement 10: Word Comparison"

    return "Uncategorized"


def extract_words_from_query(query):
    """Extract all words mentioned in single quotes from a query."""
    words = re.findall(r"['\"]([^'\"]+)['\"]", query)
    return [w.lower() for w in words]


def analyze_dataset(file_path):
    """Analyze the dataset and generate a comprehensive report."""
    print("Loading dataset...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backward compatibility: older datasets were dicts of {query: answer}
    prompts = list(data.keys()) if isinstance(data, dict) else list(data)

    total_queries = len(prompts)
    print(f"Total prompts: {total_queries:,}\n")

    # Categorize all queries
    categories = Counter()
    category_examples = defaultdict(list)
    word_usage = Counter()

    print("Categorizing queries...")
    for query in prompts:
        category = categorize_query(query)
        categories[category] += 1

        # Store examples (up to 3 per category)
        if len(category_examples[category]) < 3:
            category_examples[category].append(query)

        # Extract words for word usage analysis
        words = extract_words_from_query(query)
        for word in words:
            word_usage[word] += 1

    # Print report
    print("\n" + "=" * 80)
    print("QUERY TYPE ANALYSIS BASED ON GROUP1_PLAN.MD")
    print("=" * 80)

    # Expected counts from plan
    expected_counts = {
        "Statement 1: Spelling": 10000,
        "Statement 2: Letter at Position": 9000,
        "Statement 3: Sound Matching": 7000,
        "Statement 4: Letter Count": 9000,
        "Statement 5: Rhyming": 7000,
        "Statement 6: Classification": 7000,
        "Statement 7: Position of Letter": 6000,
        "Statement 8: Number Spelling": 3500,
        "Statement 9: Last Letter": 6000,
        "Statement 10: Word Comparison": 5500,
    }

    print("\nCategory Distribution:")
    print("-" * 80)
    print(
        f"{'Category':<40} {'Actual':>10} {'Expected':>10} {'Difference':>12} {'%':>8}"
    )
    print("-" * 80)

    total_actual = 0
    total_expected = sum(expected_counts.values())

    for category in sorted(categories.keys()):
        actual = categories[category]
        expected = expected_counts.get(category, 0)
        diff = actual - expected
        percentage = (actual / total_queries * 100) if total_queries > 0 else 0
        total_actual += actual

        status = "✓" if abs(diff) <= 100 else "⚠" if abs(diff) <= 500 else "✗"
        print(
            f"{category:<40} {actual:>10,} {expected:>10,} {diff:>+12,} {percentage:>7.2f}% {status}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<40} {total_actual:>10,} {total_expected:>10,} {total_actual - total_expected:>+12,}"
    )

    # Show examples for each category
    print("\n" + "=" * 80)
    print("EXAMPLES BY CATEGORY")
    print("=" * 80)

    for category in sorted(categories.keys()):
        if category == "Uncategorized":
            continue
        print(f"\n{category} ({categories[category]:,} queries):")
        for i, query in enumerate(category_examples[category][:3], 1):
            print(f"  Example {i}:")
            print(f"    Query:  {query}")

    # Word usage statistics
    print("\n" + "=" * 80)
    print("WORD USAGE STATISTICS")
    print("=" * 80)

    unique_words = len(word_usage)
    total_word_occurrences = sum(word_usage.values())
    avg_uses = total_word_occurrences / unique_words if unique_words > 0 else 0

    print(f"\nUnique words: {unique_words:,}")
    print(f"Total word occurrences: {total_word_occurrences:,}")
    print(f"Average uses per word: {avg_uses:.1f}")

    if word_usage:
        top_word, top_count = word_usage.most_common(1)[0]
        print(
            f"Top word ('{top_word}'): Used {top_count:,} times ({top_count/total_queries*100:.2f}% of all queries)"
        )

    print("\nTop 20 Most Repeated Words:")
    for i, (word, count) in enumerate(word_usage.most_common(20), 1):
        percentage = (count / total_queries * 100) if total_queries > 0 else 0
        print(f"  {i:2d}. {word:15s} - {count:5,} times ({percentage:5.2f}%)")

    # Uncategorized queries analysis
    if categories.get("Uncategorized", 0) > 0:
        print("\n" + "=" * 80)
        print(f"UNCATEGORIZED QUERIES ({categories['Uncategorized']:,})")
        print("=" * 80)
        uncategorized = [q for q in prompts if categorize_query(q) == "Uncategorized"]
        print("\nSample uncategorized queries:")
        for i, query in enumerate(uncategorized[:10], 1):
            print(f"  {i}. Query: {query}")


if __name__ == "__main__":
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "group1.json")
    analyze_dataset(file_path)
