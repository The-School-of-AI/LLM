"""
Example usage of the Curriculum Band Classifier

This script demonstrates how to use the classifier programmatically.

Usage:
    python example_usage.py
    python example_usage.py --local-tokenizer C:\Balaji\erav4\capstone\tokenizer
"""

import argparse
import json

from classify_curriculum_bands import CurriculumBandClassifier

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Example usage of Curriculum Band Classifier"
)
parser.add_argument(
    "--local-tokenizer",
    type=str,
    default=None,
    help="Local path to tokenizer files (overrides default HuggingFace model)",
)
args = parser.parse_args()

# Initialize classifier with Llama 3.3 tokenizer
print("Initializing classifier...")
if args.local_tokenizer:
    print(f"Using local tokenizer from: {args.local_tokenizer}")
    classifier = CurriculumBandClassifier(local_tokenizer_path=args.local_tokenizer)
else:
    print("Using HuggingFace model: meta-llama/Llama-3.3-70B-Instruct")
    classifier = CurriculumBandClassifier(model_id="meta-llama/Llama-3.3-70B-Instruct")

# Example 1: Classify individual texts
print("\n" + "=" * 60)
print("Example 1: Classifying individual texts")
print("=" * 60)

examples = [
    "The cat sat on the mat. It was happy.",  # Should be B0
    "The weather is nice today. We went to the park.",  # Should be B0-B1
    "Photosynthesis is the process by which plants convert sunlight into energy.",  # Should be B2
    "The quicksort algorithm uses divide-and-conquer with O(n log n) complexity.",  # Should be B3
    "The transformer architecture employs self-attention mechanisms.",  # Should be B4
    "The antidisestablishmentarian framework utilizes zephyr-based optimization.",  # Should be B5
]

for i, text in enumerate(examples, 1):
    record = {"text": text, "example_id": i}
    classified = classifier.process_record(record)

    print(f"\nExample {i}:")
    print(f"  Text: {text[:60]}...")

    # Check if record was rejected due to insufficient tokens
    if classified.get("rejected", False):
        print("  Status: REJECTED")
        print(f"  Reason: {classified['rejection_reason']}")
        print(f"  Token Count: {classified['token_count']}")
        print(f"  Avg Token ID: {classified['token_stats']['avg']:.1f}")
        print(f"  Max Token ID: {classified['token_stats']['max']}")
    else:
        print(f"  Band: {classified['curriculum_band']}")
        print(f"  Avg Token ID: {classified['token_stats']['avg']:.1f}")
        print(f"  Max Token ID: {classified['token_stats']['max']}")
        print(f"  Description: {classified['classification_metadata']['description']}")

# Example 2: Create a sample dataset and classify it
print("\n" + "=" * 60)
print("Example 2: Creating and classifying a sample dataset")
print("=" * 60)

sample_dataset = [
    {"text": "The cat sat on the mat.", "id": 1, "source": "web"},
    {"text": "Hello world! How are you today?", "id": 2, "source": "web"},
    {"text": "Python is a programming language.", "id": 3, "source": "code"},
    {
        "text": "The algorithm uses dynamic programming to solve the problem.",
        "id": 4,
        "source": "code",
    },
    {
        "text": "Machine learning models use neural networks for pattern recognition.",
        "id": 5,
        "source": "research",
    },
    {
        "text": "The antidisestablishmentarian methodology employs sophisticated optimization techniques.",
        "id": 6,
        "source": "research",
    },
]

# Save sample dataset
with open("sample_dataset.jsonl", "w", encoding="utf-8") as f:
    for record in sample_dataset:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Created sample_dataset.jsonl")

# Classify the dataset
band_counts = classifier.process_dataset(
    input_file="sample_dataset.jsonl",
    output_file="sample_dataset_classified.jsonl",
    text_field="text",
    input_format="jsonl",
)

print("\nClassification complete!")
print("Check sample_dataset_classified.jsonl for results.")

# Example 3: Visualize band distribution
print("\n" + "=" * 60)
print("Example 3: Visualizing band distribution")
print("=" * 60)

try:
    from visualize_band_distribution import visualize_dataset

    visualize_dataset(
        input_file="sample_dataset_classified.jsonl",
        output_file="band_distribution.png",
        input_format="jsonl",
        title="Sample Dataset - Band Distribution",
    )
    print("\n✓ Band distribution visualization saved to: band_distribution.png")
except ImportError:
    print("Note: Install matplotlib to enable visualization:")
    print("  pip install matplotlib")
except Exception as e:
    print(f"Note: Visualization error: {e}")
