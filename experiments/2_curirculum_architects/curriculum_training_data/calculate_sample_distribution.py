#!/usr/bin/env python3
"""
Analyze curriculum_training_data structure to understand all groups and examples,
then calculate distribution of 2 million samples across groups and examples.
"""

import json

# Parse the structure from curriculum_training_data
GROUPS = {
    "Group1": {
        "name": "Language and Literacy",
        "examples": [
            'What\'s is the spelling of "cat"?',
            'Can you say the "first" letter in "Cat"?',
            'Which word starts with the sound "B", "Ball" or "Cat".',
            'How many alphabets are there in "cat"?',
            'What word rhymes with "cat"?',
            'Is "cat" a person, animal or thing?',
            'At what location letter "a" comes in "cat"?',
            "What is the spelling of 11?",
            'What letter does "cat" ends with?',
            'Which word is longer "cat" or "cream"?',
        ],
    },
    "Group2": {
        "name": "Math and Numbers",
        "examples": [
            "Can you count till 5?",
            'What comes "after" 46?',
            'If you have one apple, and get one more, how many "apples" now?',
            "Which number is greater, 5 or -9?",
            "Which number is bigger 1000 or 999?",
            "What is 2 + 2?",
            "What is 5x5?",
            "What is 4 less than 10?",
        ],
    },
    "Group3": {
        "name": "Shapes, Colors & Patterns",
        "examples": [
            "What color is banana?",
            "What is the shape of a ball?",
            "Which shape has three sides?",
            "What comes next: blue, red, blue, red, …",
        ],
    },
    "Group4": {
        "name": "Everyday thinking & Concepts",
        "examples": [
            "List 10000 basic facts",
        ],
    },
    "Group5": {
        "name": "Examples",
        "examples": [
            'The spelling of " re" is  , r, e.',
            'The spelling of " be" is  , b, e.',
            'The spelling of " is" is  , i, s.',
            'The spelling of " on" is  , o, n.',
            'The spelling of " that" is  , t, h, a, t.',
            'The spelling of " for" is  , f, o, r.',
            'The spelling of "ver" is v, e, r.',
            'The spelling of " st" is  , s, t.',
        ],
    },
}


def calculate_distribution(total_samples=2_000_000):
    """Calculate sample distribution across groups and examples."""

    # Count total examples
    total_examples = sum(len(group["examples"]) for group in GROUPS.values())

    print("=" * 80)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal groups: {len(GROUPS)}")
    print(f"Total examples across all groups: {total_examples}\n")

    # Show structure
    for group_id, group_info in GROUPS.items():
        print(f"{group_id}: {group_info['name']}")
        print(f"  Examples: {len(group_info['examples'])}")
        for i, example in enumerate(group_info["examples"], 1):
            print(f"    {i}. {example[:70]}...")
        print()

    print("=" * 80)
    print("SAMPLE DISTRIBUTION STRATEGY")
    print("=" * 80)

    # Strategy 1: Equal distribution per example
    print("\nStrategy 1: Equal distribution per example")
    samples_per_example_equal = total_samples // total_examples
    remainder_equal = total_samples % total_examples

    print(f"Samples per example: {samples_per_example_equal:,}")
    print(f"Remainder: {remainder_equal:,}")
    print("\nDistribution:")
    for group_id, group_info in GROUPS.items():
        group_samples = len(group_info["examples"]) * samples_per_example_equal
        print(
            f"  {group_id}: {group_samples:,} samples ({len(group_info['examples'])} examples × {samples_per_example_equal:,})"
        )

    # Strategy 2: Weighted by group complexity/importance
    print("\n" + "=" * 80)
    print("Strategy 2: Weighted distribution (recommended)")
    print("=" * 80)

    # Assign weights based on complexity and educational value
    weights = {
        "Group1": 0.35,  # Language and Literacy - foundational, many variations
        "Group2": 0.30,  # Math and Numbers - important, many variations
        "Group3": 0.15,  # Shapes, Colors & Patterns - moderate
        "Group4": 0.10,  # Everyday thinking - fewer variations
        "Group5": 0.10,  # Examples - spelling variations
    }

    print("\nGroup weights:")
    for group_id, weight in weights.items():
        print(f"  {group_id}: {weight*100:.1f}%")

    # Calculate samples per group
    group_samples = {}
    for group_id, weight in weights.items():
        group_samples[group_id] = int(total_samples * weight)

    # Adjust for rounding
    total_allocated = sum(group_samples.values())
    difference = total_samples - total_allocated
    # Add difference to Group1 (largest group)
    group_samples["Group1"] += difference

    print(f"\nTotal allocated: {sum(group_samples.values()):,}")
    print("\nSamples per group:")
    for group_id, samples in group_samples.items():
        num_examples = len(GROUPS[group_id]["examples"])
        samples_per_example = samples // num_examples
        remainder = samples % num_examples
        print(f"  {group_id}: {samples:,} samples")
        print(f"    Examples: {num_examples}")
        print(f"    Per example: {samples_per_example:,} (remainder: {remainder:,})")

    # Strategy 3: Equal per group, then distribute within group
    print("\n" + "=" * 80)
    print("Strategy 3: Equal per group, then distribute within group")
    print("=" * 80)

    samples_per_group_equal = total_samples // len(GROUPS)
    remainder_group = total_samples % len(GROUPS)

    print(
        f"Samples per group: {samples_per_group_equal:,} (remainder: {remainder_group:,})"
    )
    print("\nDistribution:")
    for group_id, group_info in GROUPS.items():
        group_total = samples_per_group_equal + (1 if remainder_group > 0 else 0)
        remainder_group -= 1
        num_examples = len(group_info["examples"])
        samples_per_example = group_total // num_examples
        remainder_example = group_total % num_examples
        print(f"  {group_id}: {group_total:,} samples")
        print(
            f"    Per example: {samples_per_example:,} (remainder: {remainder_example:,})"
        )

    # Detailed breakdown for Strategy 2 (recommended)
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN - Strategy 2 (Recommended)")
    print("=" * 80)

    detailed_distribution = {}
    for group_id, group_info in GROUPS.items():
        total_group_samples = group_samples[group_id]
        num_examples = len(group_info["examples"])
        samples_per_example = total_group_samples // num_examples
        remainder = total_group_samples % num_examples

        detailed_distribution[group_id] = {
            "total_samples": total_group_samples,
            "examples": {},
        }

        print(f"\n{group_id}: {group_info['name']}")
        print(f"  Total samples: {total_group_samples:,}")
        print(f"  Number of examples: {num_examples}")
        print(f"  Base samples per example: {samples_per_example:,}")
        print(f"  Remainder: {remainder:,}")
        print("\n  Example distribution:")

        for i, example in enumerate(group_info["examples"], 1):
            # Distribute remainder to first examples
            example_samples = samples_per_example + (1 if i <= remainder else 0)
            detailed_distribution[group_id]["examples"][f"Example{i}"] = {
                "samples": example_samples,
                "template": example,
            }
            print(f"    Example {i}: {example_samples:,} samples")
            print(f"      Template: {example[:70]}...")

    # Save detailed distribution to JSON
    output_file = "sample_distribution_plan.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_samples": total_samples,
                "strategy": "Weighted distribution",
                "distribution": detailed_distribution,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n\nDetailed distribution saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(
        f"\n{'Group':<10} {'Group Name':<30} {'Examples':<10} {'Samples':<15} {'Per Example':<15}"
    )
    print("-" * 80)
    for group_id, group_info in GROUPS.items():
        total_group_samples = group_samples[group_id]
        num_examples = len(group_info["examples"])
        samples_per_example = total_group_samples // num_examples
        print(
            f"{group_id:<10} {group_info['name']:<30} {num_examples:<10} {total_group_samples:<15,} {samples_per_example:<15,}"
        )
    print("-" * 80)
    print(
        f"{'TOTAL':<10} {'':<30} {total_examples:<10} {sum(group_samples.values()):<15,} {'':<15}"
    )


if __name__ == "__main__":
    calculate_distribution(2_000_000)
