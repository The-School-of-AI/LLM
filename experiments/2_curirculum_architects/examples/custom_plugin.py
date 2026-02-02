"""Example: Adding a custom metric without editing core code."""

from pathlib import Path

from curriculum_tags import CurriculumConfig, CurriculumTagger
from curriculum_tags.core.plugin import MetricPlugin


class SentimentMetric(MetricPlugin):
    """Custom sentiment metric (demo only)."""

    name = "sentiment"

    POSITIVE_WORDS = {"good", "great", "excellent", "amazing", "wonderful"}
    NEGATIVE_WORDS = {"bad", "terrible", "awful", "horrible", "poor"}

    def compute(self, sample):
        """Compute sentiment score."""
        text = sample.get("text", "").lower()
        words = text.split()

        if not words:
            return {"score": 0.0, "category": "neutral"}

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)

        score = (pos_count - neg_count) / len(words)

        if score > 0.05:
            category = "positive"
        elif score < -0.05:
            category = "negative"
        else:
            category = "neutral"

        return {"score": round(score, 3), "category": category}


def main():
    """Demo custom metric."""
    curriculum_path = Path(__file__).parent.parent / "curriculum.yaml"
    config = CurriculumConfig(curriculum_path)

    # Option 1: Add custom metric to default ones
    from curriculum_tags import DifficultyMetric

    metrics = [
        DifficultyMetric(config),
        SentimentMetric(config),  # Your custom metric
    ]

    tagger = CurriculumTagger(curriculum_path, metrics=metrics)

    # Test samples
    samples = [
        {"text": "This is great and wonderful!"},
        {"text": "This is terrible and awful."},
        {"text": "Just a normal sentence."},
    ]

    print("Custom Sentiment Metric Demo")
    print("=" * 60)

    for sample in samples:
        tagged = tagger.tag_sample(sample)
        sentiment = tagged["curriculum_tags"]["sentiment"]

        print(f"\nText: {sample['text']}")
        print(f"  Score: {sentiment['score']}")
        print(f"  Category: {sentiment['category']}")


if __name__ == "__main__":
    main()
