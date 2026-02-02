"""
Calibrate classification thresholds based on your actual tokenizer

This script helps you find the right thresholds by testing sample texts
from each difficulty band.
"""

import numpy as np
from classify_curriculum_bands import CurriculumBandClassifier

# Sample texts for each band (you can modify these)
SAMPLE_TEXTS = {
    "B0": [
        "The cat sat on the mat. It was happy.",
        "Hello world! How are you today?",
        "I like to play outside. The sun is bright.",
    ],
    "B1": [
        "The weather is nice today. We went to the park and had a picnic.",
        "My favorite subject in school is science. I enjoy learning about animals.",
    ],
    "B2": [
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The water cycle describes how water moves through the environment.",
    ],
    "B3": [
        "The quicksort algorithm uses divide-and-conquer with O(n log n) average complexity.",
        "Python's list comprehension provides a concise way to create lists.",
    ],
    "B4": [
        "The transformer architecture employs self-attention mechanisms to model long-range dependencies.",
        "Machine learning models use gradient descent to optimize their parameters.",
    ],
    "B5": [
        "The antidisestablishmentarian framework utilizes zephyr-based optimization for quantum computing applications.",
        "Sophisticated neural architectures leverage attention mechanisms for cross-modal understanding.",
    ],
}


def calibrate_thresholds(local_tokenizer_path=None):
    """
    Calibrate thresholds by analyzing sample texts from each band.

    Args:
        local_tokenizer_path: Optional local path to tokenizer
    """
    print("=" * 60)
    print("CALIBRATING THRESHOLDS")
    print("=" * 60)

    # Initialize classifier
    try:
        if local_tokenizer_path:
            classifier = CurriculumBandClassifier(
                local_tokenizer_path=local_tokenizer_path
            )
        else:
            classifier = CurriculumBandClassifier()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("\nTry specifying a local tokenizer path:")
        print("  python calibrate_thresholds.py --local-tokenizer <path>")
        return

    print("\n" + "=" * 60)
    print("ANALYZING SAMPLE TEXTS")
    print("=" * 60)

    band_stats = {}

    for band, texts in SAMPLE_TEXTS.items():
        print(f"\n{band} Samples:")
        print("-" * 60)

        all_avgs = []
        all_maxs = []
        all_p95s = []

        for i, text in enumerate(texts, 1):
            token_ids = classifier.tokenize_text(text)
            stats = classifier.calculate_token_stats(token_ids)

            all_avgs.append(stats["avg"])
            all_maxs.append(stats["max"])
            all_p95s.append(stats["p95"])

            print(f"  Sample {i}:")
            print(f"    Text: {text[:50]}...")
            print(
                f"    Avg: {stats['avg']:.1f}, Max: {stats['max']}, P95: {stats['p95']:.1f}"
            )

        # Calculate statistics across all samples in this band
        band_stats[band] = {
            "avg_mean": np.mean(all_avgs),
            "avg_max": np.max(all_avgs),
            "max_mean": np.mean(all_maxs),
            "max_max": np.max(all_maxs),
            "p95_mean": np.mean(all_p95s),
            "p95_max": np.max(all_p95s),
        }

        print(f"\n  {band} Summary:")
        print(
            f"    Avg ID: mean={band_stats[band]['avg_mean']:.1f}, max={band_stats[band]['avg_max']:.1f}"
        )
        print(
            f"    Max ID: mean={band_stats[band]['max_mean']:.1f}, max={band_stats[band]['max_max']}"
        )
        print(
            f"    P95 ID: mean={band_stats[band]['p95_mean']:.1f}, max={band_stats[band]['p95_max']:.1f}"
        )

    # Suggest thresholds
    print("\n" + "=" * 60)
    print("SUGGESTED THRESHOLDS")
    print("=" * 60)
    print("\nBased on your tokenizer, here are suggested thresholds:")
    print("\nTOKEN_ID_THRESHOLDS = {")

    bands = ["B0", "B1", "B2", "B3", "B4", "B5"]
    for i, band in enumerate(bands):
        if band in band_stats:
            stats = band_stats[band]
            # Use max values from this band as thresholds for next band
            if i < len(bands) - 1:
                # next_band = bands[i + 1]
                # Set threshold slightly above current band's max
                avg_thresh = int(stats["avg_max"] * 1.2)
                max_thresh = int(stats["max_max"] * 1.2)
                p95_thresh = int(stats["p95_max"] * 1.2)
            else:
                avg_thresh = "float('inf')"
                max_thresh = "float('inf')"
                p95_thresh = "float('inf')"

            print(f"    '{band}': {{")
            print(f"        'avg_max': {avg_thresh},")
            print(f"        'max_max': {max_thresh},")
            print(f"        'p95_max': {p95_thresh},")
            print(
                f"        'description': '{classifier.TOKEN_ID_THRESHOLDS[band]['description']}'"
            )
            print("    },")

    print("}")

    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("\n1. Review the suggested thresholds above")
    print("2. Update TOKEN_ID_THRESHOLDS in classify_curriculum_bands.py")
    print("3. Test with your actual dataset to fine-tune if needed")
    print("\nNote: Thresholds should ensure B0 samples are classified as B0,")
    print("      and each band's samples are correctly classified.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate classification thresholds")
    parser.add_argument(
        "--local-tokenizer",
        type=str,
        default=None,
        help="Local path to tokenizer files",
    )
    args = parser.parse_args()

    calibrate_thresholds(local_tokenizer_path=args.local_tokenizer)
