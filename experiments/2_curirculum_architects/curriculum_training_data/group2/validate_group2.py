#!/usr/bin/env python3
"""
Validate Group 2 distribution
"""

import json
from collections import defaultdict

# Load the data
with open("../group2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Backward compatibility: older datasets were dicts of {query: answer}
all_prompts = list(data.keys()) if isinstance(data, dict) else list(data)

print(f"Total prompts: {len(all_prompts):,}\n")

# Get some examples of each type
examples = defaultdict(list)

# Object words
OBJECTS = [
    "apple",
    "apples",
    "orange",
    "oranges",
    "banana",
    "bananas",
    "mango",
    "mangoes",
    "grape",
    "grapes",
    "strawberry",
    "strawberries",
    "cherry",
    "cherries",
    "cat",
    "cats",
    "dog",
    "dogs",
    "bird",
    "birds",
    "fish",
    "rabbit",
    "rabbits",
    "duck",
    "ducks",
    "chicken",
    "chickens",
    "ball",
    "balls",
    "doll",
    "dolls",
    "car",
    "cars",
    "block",
    "blocks",
    "puzzle",
    "puzzles",
    "crayon",
    "crayons",
    "marble",
    "marbles",
    "cookie",
    "cookies",
    "candy",
    "candies",
    "chocolate",
    "chocolates",
    "pizza",
    "pizzas",
    "cupcake",
    "cupcakes",
    "sandwich",
    "sandwiches",
    "pencil",
    "pencils",
    "book",
    "books",
    "eraser",
    "erasers",
    "notebook",
    "notebooks",
    "pen",
    "pens",
    "ruler",
    "rulers",
    "flower",
    "flowers",
    "leaf",
    "leaves",
    "stone",
    "stones",
    "shell",
    "shells",
    "butterfly",
    "butterflies",
    "star",
    "stars",
]

object_words = set(OBJECTS)

expected_counts = {
    "Statement 1: Counting": 60000,
    "Statement 2: Before/After": 80000,
    "Statement 3: Word Problems": 120000,
    "Statement 4: Comparisons": 100000,
    "Statement 5: Direct Math": 150000,
    "Statement 6: Word-Based Math": 90000,
}

categories = defaultdict(int)

for query in all_prompts:
    query_lower = query.lower()

    # Check in order from most specific to least specific

    # S3: Word Problems - contains object words (check first as it's very specific)
    if any(obj in query_lower for obj in object_words):
        categories["Statement 3: Word Problems"] += 1
        if len(examples["S3"]) < 5:
            examples["S3"].append(f"  {query}")
    # S1: Counting - "count", "numbers from 1 to", etc.
    elif any(
        p in query_lower
        for p in [
            "count till",
            "count from",
            "count to",
            "numbers from 1",
            "numbers up to",
            "counting till",
            "counting to",
            "count the numbers",
            "sequence from 1",
            "recite numbers",
            "list the numbers",
            "list numbers",
            "give me the count",
            "what numbers come",
            "show me counting",
        ]
    ):
        categories["Statement 1: Counting"] += 1
        if len(examples["S1"]) < 5:
            examples["S1"].append(f"  {query}")
    # S6: Word-Based Math - specific phrases (before S2 to avoid "less"/"more" confusion)
    elif any(
        p in query_lower
        for p in [
            "more than",
            "less than",
            "fewer than",
            "double of",
            "triple of",
            "twice",
            "thrice",
            "half of",
            "quarter of",
            "increased by",
            "decreased by",
            "added to",
            "subtracted from",
            "taken from",
            "with",
            "times",
            "multiplied by",
            "divided by",
            "split into",
            "shared among",
        ]
    ):
        # Exclude if it has operators (would be S5)
        if not any(op in query for op in ["+", "-", "×", "÷", "*", "/"]):
            categories["Statement 6: Word-Based Math"] += 1
            if len(examples["S6"]) < 5:
                examples["S6"].append(f"  {query}")
            continue
        else:
            categories["Statement 5: Direct Math"] += 1
            if len(examples["S5"]) < 5:
                examples["S5"].append(f"  {query}")
    # S2: Before/After - very specific patterns
    elif any(
        p in query_lower for p in ["after", "before", "follows", "precedes", "succeeds"]
    ):
        # Must NOT have operators or comparison words
        if not any(op in query for op in ["+", "-", "×", "÷", "*", "/"]) and not any(
            w in query_lower for w in ["greater", "bigger", "smaller", "which is"]
        ):
            categories["Statement 2: Before/After"] += 1
            if len(examples["S2"]) < 5:
                examples["S2"].append(f"  {query}")
        else:
            categories["Statement 5: Direct Math"] += 1
            if len(examples["S5"]) < 5:
                examples["S5"].append(f"  {query}")
    # S4: Comparisons - "greater", "bigger", "smaller", etc.
    elif any(
        p in query_lower
        for p in [
            "which number is",
            "which is",
            "what is",
            "greater",
            "bigger",
            "larger",
            "smaller",
            "lesser",
            "compare",
            "between",
            "pick the",
        ]
    ):
        # Must have comparison words, not operators
        if any(
            w in query_lower
            for w in [
                "greater",
                "bigger",
                "larger",
                "smaller",
                "less",
                "lower",
                "higher",
                "more",
            ]
        ) and not any(op in query for op in ["+", "-", "×", "÷", "*", "/"]):
            categories["Statement 4: Comparisons"] += 1
            if len(examples["S4"]) < 5:
                examples["S4"].append(f"  {query}")
        else:
            categories["Statement 5: Direct Math"] += 1
            if len(examples["S5"]) < 5:
                examples["S5"].append(f"  {query}")
    # S5: Direct Math - contains operators or specific math words
    elif any(op in query for op in ["+", "-", "×", "÷", "*", "/"]) or any(
        p in query_lower
        for p in [
            "calculate",
            "compute",
            "solve",
            "evaluate",
            "what is",
            "find the value",
            "equals",
            "answer to",
        ]
    ):
        categories["Statement 5: Direct Math"] += 1
        if len(examples["S5"]) < 5:
            examples["S5"].append(f"  {query}")
    else:
        categories["Uncategorized"] += 1
        if len(examples["Uncat"]) < 10:
            examples["Uncat"].append(f"  {query}")

print("=" * 80)
print("DISTRIBUTION VALIDATION")
print("=" * 80)
print(
    f"{'Category':<40} {'Actual':>10} {'Expected':>10} {'Difference':>12} {'Status':>10}"
)
print("-" * 80)

has_issues = False
for category in sorted(expected_counts.keys()):
    actual = categories.get(category, 0)
    expected = expected_counts[category]
    diff = actual - expected

    if abs(diff) <= 1000:
        status = "✓ OK"
    elif abs(diff) <= 5000:
        status = "⚠ WARNING"
        has_issues = True
    else:
        status = "✗ ERROR"
        has_issues = True

    print(f"{category:<40} {actual:>10,} {expected:>10,} {diff:>+12,} {status:>10}")

# Show uncategorized if any
uncategorized = categories.get("Uncategorized", 0)
if uncategorized > 0:
    print(
        f"{'Uncategorized':<40} {uncategorized:>10,} {'0':>10} {'+' + str(uncategorized):>12} {'✗ ERROR':>10}"
    )
    has_issues = True

print("-" * 80)
total_categorized = sum(v for k, v in categories.items() if k != "Uncategorized")
print(
    f"{'TOTAL (categorized)':<40} {total_categorized:>10,} {sum(expected_counts.values()):>10,}"
)

if has_issues:
    print("\n⚠ WARNING: Distribution has significant deviations from expected values!")
else:
    print("\n✓ Distribution looks good!")

# Show examples
print("\n" + "=" * 80)
print("SAMPLE EXAMPLES")
print("=" * 80)

for key in ["S1", "S2", "S3", "S4", "S5", "S6"]:
    if key in examples and examples[key]:
        print(f"\n{key} Examples:")
        for ex in examples[key][:3]:
            print(ex)

if "Uncat" in examples and examples["Uncat"]:
    print("\nUncategorized Examples (first 5):")
    for ex in examples["Uncat"][:5]:
        print(ex)
