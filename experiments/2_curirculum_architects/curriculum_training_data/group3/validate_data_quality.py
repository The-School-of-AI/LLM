#!/usr/bin/env python3
# isort: skip_file
"""
Validation script to catch factual errors and unnatural phrasing in generated datasets.
Run this before committing any generated data.
"""

import json
import re
from typing import Dict, List

# ============================================================================
# FACTUAL WARNINGS (prompt-only) - Ambiguities to flag
# ============================================================================

AMBIGUOUS_OBJECTS = {
    # Multi-color / multi-part color objects
    "police car lights": "often more than one color",
    "traffic light": "has multiple colors (red, yellow, green)",
    "rainbow": "has many colors",
    # Commonly ambiguous / varies by type or context
    "mango": "can be yellow or green depending on ripeness",
    "watermelon": "green outside and red inside",
    "cheese": "varies by type (yellow/white/orange)",
}

# Objects that need article correction
NEEDS_ARTICLE_FIX = {
    "wool": "wool",  # Not "a wool"
    "butter": "butter",  # Not "a butter"
    "milk": "milk",  # Not "a milk"
    "water": "water",  # Not "a water"
}

# ============================================================================
# UNNATURAL PHRASING PATTERNS - Should be flagged
# ============================================================================

UNNATURAL_PATTERNS = [
    r"is what color\?",  # "A peach is what color?" → Should be "What color is a peach?"
    r"In what color would you find",  # Awkward phrasing
    r"predominant color",  # Too formal
    r"standard color",  # Too formal
    r"regular \w+",  # "regular seagull" - awkward
    r"typical \w+",  # "typical dust" - awkward
    r"average \w+",  # "average orange" - awkward
    r"normal \w+",  # "normal taxi" - awkward
    r"mature \w+",  # "mature butter" - doesn't make sense
    r"lovely \w+",  # "lovely chick" - awkward
    r"in nature,",  # Unnecessary context
    r"in a classroom",  # Unnecessary context
    r"in the mountains",  # Unnecessary context
    r"in a garden",  # Unnecessary context
    r"would be",  # "What color would a peach be?" - awkward
    r"a orange",  # Should be "an orange"
    r"a apple",  # Should be "an apple"
    r"a elephant",  # Should be "an elephant"
    r"a umbrella",  # Should be "an umbrella"
    r"a hour",  # Should be "an hour"
    r"\?\s*:",  # Colon after question mark
    r":\s*$",  # Colon at end of query
]

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def check_factual_warnings(query: str) -> List[str]:
    """
    Prompt-only factual warnings.
    Since we don't have answers, flag prompts that ask for a single color of
    inherently multi-color or ambiguous objects.
    """
    warnings: list[str] = []
    query_lower = query.lower()

    if "what color" in query_lower or "color" in query_lower:
        for obj, reason in AMBIGUOUS_OBJECTS.items():
            if obj in query_lower:
                warnings.append(f"Ambiguous object: '{obj}' ({reason})")
    return warnings


def check_unnatural_phrasing(query: str) -> List[str]:
    """Check for unnatural English phrasing."""
    issues = []

    for pattern in UNNATURAL_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            issues.append(f"Unnatural phrasing: matches pattern '{pattern}'")

    return issues


def check_grammar(query: str) -> List[str]:
    """Check for grammar errors."""
    errors = []

    # Article errors
    if re.search(r"\ba orange\b", query, re.IGNORECASE):
        errors.append("Grammar: 'a orange' should be 'an orange'")
    if re.search(r"\ba apple\b", query, re.IGNORECASE):
        errors.append("Grammar: 'a apple' should be 'an apple'")
    if re.search(r"\ba elephant\b", query, re.IGNORECASE):
        errors.append("Grammar: 'a elephant' should be 'an elephant'")

    # Non-countable nouns with articles
    for noun, correct in NEEDS_ARTICLE_FIX.items():
        if re.search(rf"\ba {noun}\b", query, re.IGNORECASE):
            errors.append(f"Grammar: 'a {noun}' should be '{correct}'")

    return errors


def check_formatting(query: str) -> List[str]:
    """Check for formatting issues."""
    issues = []

    # Colon after question mark
    if re.search(r"\?\s*:", query):
        issues.append("Formatting: Colon after question mark")

    # Colon at end
    if query.rstrip().endswith(":"):
        issues.append("Formatting: Colon at end of query")

    # Training-related text
    if "training" in query.lower():
        issues.append("Formatting: Contains 'training' text")

    return issues


def validate_sample(query: str) -> Dict[str, List[str]]:
    """Validate a single prompt string."""
    return {
        "factual_warnings": check_factual_warnings(query),
        "unnatural_phrasing": check_unnatural_phrasing(query),
        "grammar_errors": check_grammar(query),
        "formatting_issues": check_formatting(query),
    }


def validate_dataset(file_path: str, max_samples: int = None) -> Dict:
    """Validate an entire dataset file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backward compatibility: older datasets were dicts of {query: answer}
    prompts = list(data.keys()) if isinstance(data, dict) else list(data)

    total_samples = len(prompts)
    samples_to_check = prompts[:max_samples] if max_samples else prompts

    results = {
        "total_samples": total_samples,
        "checked_samples": len(samples_to_check),
        "factual_warnings": [],
        "unnatural_phrasing": [],
        "grammar_errors": [],
        "formatting_issues": [],
        "warning_count": 0,
        "error_count": 0,
    }

    for query in samples_to_check:
        validation = validate_sample(query)

        if validation["factual_warnings"]:
            results["factual_warnings"].append(
                {
                    "query": query,
                    "warnings": validation["factual_warnings"],
                }
            )
            results["warning_count"] += len(validation["factual_warnings"])

        if validation["unnatural_phrasing"]:
            results["unnatural_phrasing"].append(
                {"query": query, "issues": validation["unnatural_phrasing"]}
            )
            results["error_count"] += len(validation["unnatural_phrasing"])

        if validation["grammar_errors"]:
            results["grammar_errors"].append(
                {"query": query, "errors": validation["grammar_errors"]}
            )
            results["error_count"] += len(validation["grammar_errors"])

        if validation["formatting_issues"]:
            results["formatting_issues"].append(
                {"query": query, "issues": validation["formatting_issues"]}
            )
            results["error_count"] += len(validation["formatting_issues"])

    return results


def print_report(results: Dict, show_examples: int = 10):
    """Print validation report."""
    print("=" * 80)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 80)
    print(f"\nTotal samples: {results['total_samples']:,}")
    print(f"Samples checked: {results['checked_samples']:,}")
    print(f"Total errors found: {results['error_count']}")
    print(f"Total warnings found: {results.get('warning_count', 0)}")

    # Factual warnings
    if results["factual_warnings"]:
        print(f"\n⚠ FACTUAL WARNINGS: {len(results['factual_warnings'])} found")
        print("-" * 80)
        for i, item in enumerate(results["factual_warnings"][:show_examples], 1):
            print(f"\n{i}. Query: {item['query']}")
            for warning in item["warnings"]:
                print(f"   → {warning}")

    # Unnatural phrasing
    if results["unnatural_phrasing"]:
        print(f"\n⚠️  UNNATURAL PHRASING: {len(results['unnatural_phrasing'])} found")
        print("-" * 80)
        for i, item in enumerate(results["unnatural_phrasing"][:show_examples], 1):
            print(f"\n{i}. Query: {item['query']}")
            for issue in item["issues"]:
                print(f"   → {issue}")

    # Grammar errors
    if results["grammar_errors"]:
        print(f"\n❌ GRAMMAR ERRORS: {len(results['grammar_errors'])} found")
        print("-" * 80)
        for i, item in enumerate(results["grammar_errors"][:show_examples], 1):
            print(f"\n{i}. Query: {item['query']}")
            for error in item["errors"]:
                print(f"   → {error}")

    # Formatting issues
    if results["formatting_issues"]:
        print(f"\n⚠️  FORMATTING ISSUES: {len(results['formatting_issues'])} found")
        print("-" * 80)
        for i, item in enumerate(results["formatting_issues"][:show_examples], 1):
            print(f"\n{i}. Query: {item['query']}")
            for issue in item["issues"]:
                print(f"   → {issue}")

    # Summary
    print("\n" + "=" * 80)
    if results["error_count"] == 0:
        if results.get("warning_count", 0) == 0:
            print("✅ NO ERRORS OR WARNINGS FOUND - Dataset quality is good!")
        else:
            print(
                f"✅ NO ERRORS FOUND - Dataset has {results.get('warning_count', 0)} warning(s)"
            )
    else:
        print(f"❌ {results['error_count']} ERROR(S) FOUND - Dataset needs fixes")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python validate_data_quality.py <dataset.json> [max_samples]")
        print("\nExamples:")
        print("  python validate_data_quality.py ../group3.json 1000")
        print(
            "  python validate_data_quality.py ../../curriculum_training_data/group3.json 1000"
        )
        sys.exit(1)

    file_path = sys.argv[1]

    # Resolve relative paths
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Validating: {file_path}")
    if max_samples:
        print(f"Checking first {max_samples:,} samples...")

    results = validate_dataset(file_path, max_samples)
    print_report(results, show_examples=15)
