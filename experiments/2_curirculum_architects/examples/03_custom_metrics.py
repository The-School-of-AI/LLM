"""Example: Custom metric with rejection support.

This example demonstrates:
- Creating a custom MetricPlugin
- Implementing rejection logic
- Using read-only records
- Setting metric levels
"""

from pathlib import Path
from typing import Any, Dict

from curriculum_extractor import CurriculumExtractor
from curriculum_extractor.core.plugin import (
    ExtractionResult,
    MetricPlugin,
    ReadOnlyRecord,
)
from curriculum_extractor.utils.curriculum_loader import CurriculumConfig


class MinLengthFilter(MetricPlugin):
    """Filter out texts that are too short.

    This is a level 0 metric so it runs first, before expensive
    computations on content that would be rejected anyway.
    """

    name = "min_length"
    level = 0  # Run first
    min_chars = 50
    min_words = 10

    def compute(self, record: ReadOnlyRecord) -> Dict[str, Any]:
        """Compute length metrics."""
        text = record.get("text", "")
        words = text.split()

        return {
            "char_count": len(text),
            "word_count": len(words),
            "avg_word_length": len(text) / len(words) if words else 0,
        }

    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        """Extract with rejection for short texts."""
        text = record.get("text", "")
        words = text.split()

        # Check minimum length requirements
        if len(text) < self.min_chars:
            return ExtractionResult(
                metrics={},
                rejected=True,
                rejection_reason=f"Text too short: {len(text)} chars < {self.min_chars}",
            )

        if len(words) < self.min_words:
            return ExtractionResult(
                metrics={},
                rejected=True,
                rejection_reason=f"Too few words: {len(words)} < {self.min_words}",
            )

        return ExtractionResult(metrics=self.compute(record))


class LanguageFilter(MetricPlugin):
    """Filter out non-English content.

    Simple heuristic based on common English words.
    """

    name = "language"
    level = 0  # Run first

    COMMON_ENGLISH = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "or",
        "and",
    }

    def compute(self, record: ReadOnlyRecord) -> Dict[str, Any]:
        """Compute language metrics."""
        text = record.get("text", "")
        words = text.lower().split()

        if not words:
            return {"english_ratio": 0.0, "detected": "unknown"}

        english_count = sum(1 for w in words if w in self.COMMON_ENGLISH)
        ratio = english_count / len(words)

        return {
            "english_ratio": round(ratio, 3),
            "detected": "en" if ratio > 0.1 else "unknown",
        }

    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        """Extract with rejection for non-English."""
        metrics = self.compute(record)

        if metrics["english_ratio"] < 0.05:
            return ExtractionResult(
                metrics=metrics,
                rejected=True,
                rejection_reason=f"Low English ratio: {metrics['english_ratio']:.2f}",
            )

        return ExtractionResult(metrics=metrics)


class QualityScorer(MetricPlugin):
    """Compute overall quality score.

    This runs at level 1, after basic filters have removed
    obviously bad content.
    """

    name = "quality"
    level = 1  # Run after filters

    def compute(self, record: ReadOnlyRecord) -> Dict[str, Any]:
        """Compute quality metrics."""
        text = record.get("text", "")

        # Simple quality heuristics
        lines = text.strip().split("\n")
        non_empty_lines = [l for l in lines if l.strip()]  # noqa

        # Check for formatting issues
        has_proper_punctuation = any(text.rstrip().endswith(p) for p in ".!?:\"'")

        # Calculate diversity
        words = text.lower().split()
        unique_words = set(words)
        diversity = len(unique_words) / len(words) if words else 0

        # Composite score
        score = 0.0
        if len(text) > 100:
            score += 0.2
        if len(words) > 20:
            score += 0.2
        if diversity > 0.3:
            score += 0.2
        if has_proper_punctuation:
            score += 0.2
        if len(non_empty_lines) > 1:
            score += 0.2

        return {
            "score": round(score, 2),
            "diversity": round(diversity, 3),
            "line_count": len(non_empty_lines),
            "has_punctuation": has_proper_punctuation,
        }


def main():
    """Demonstrate custom metrics with rejection."""
    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"

    print("=" * 80)
    print("CURRICULUM EXTRACTOR - Custom Metrics")
    print("=" * 80)

    # Load config for custom metrics
    config = CurriculumConfig(curriculum_path)

    # Create custom metric instances
    custom_metrics = [
        MinLengthFilter(config),
        LanguageFilter(config),
        QualityScorer(config),
    ]

    # Initialize extractor with ONLY custom metrics
    extractor = CurriculumExtractor(
        curriculum_path,
        metrics=custom_metrics,  # Use only our custom metrics
        track_timing=True,
    )

    print("\n[OK] Loaded custom metrics:")
    for plugin in extractor.plugins:
        print(f"  - {plugin.name} (level {plugin.level})")

    # Test samples with varying quality
    samples = [
        {
            "id": "good_english",
            "text": """
The quick brown fox jumps over the lazy dog. This is a classic
pangram that contains every letter of the English alphabet. It has
been used for typing tests and font displays for many years. The
sentence is grammatically correct and easy to understand.
            """,
        },
        {
            "id": "too_short",
            "text": "Hello world!",
        },
        {
            "id": "non_english",
            "text": """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do 
eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim 
ad minim veniam, quis nostrud exercitation ullamco laboris.
            """,
        },
        {
            "id": "code_sample",
            "text": """
def process_data(items):
    '''Process a list of items and return the results.'''
    results = []
    for item in items:
        # Apply transformation
        transformed = item.upper()
        results.append(transformed)
    return results

# Example usage of the function
data = ['apple', 'banana', 'cherry']
output = process_data(data)
print(f"Processed: {output}")
            """,
        },
    ]

    print("\n" + "=" * 80)
    print("PROCESSING WITH CUSTOM METRICS")
    print("=" * 80)

    for sample in samples:
        print(f"\n--- {sample['id']} ---")
        print(f"Preview: {sample['text'][:60].strip()}...")

        metadata, rejection = extractor.extract_record(sample)

        if rejection:
            print(f"  ❌ REJECTED at '{rejection.rejected_at}'")
            print(f"     Reason: {rejection.rejected_reason}")
        else:
            print("  ✓ ACCEPTED")
            for key, value in metadata.items():
                if key != "curriculum_version":
                    print(f"     {key}: {value}")

    # Show timing
    timing = extractor.get_timing_stats()
    if timing:
        print("\n" + "-" * 40)
        print("TIMING:")
        print("-" * 40)
        for name, stats in timing.items():
            print(f"  {name}: {stats['mean_ms']:.3f}ms avg ({stats['count']} calls)")

    print("\n[OK] Custom metrics example complete.")


if __name__ == "__main__":
    main()
