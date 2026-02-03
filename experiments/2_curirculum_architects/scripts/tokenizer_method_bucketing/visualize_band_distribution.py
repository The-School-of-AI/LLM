"""
Visualize Band Distribution from Classified Dataset

This script reads a classified dataset and creates visualizations showing
the distribution of records across B0-B5 curriculum bands.

Usage:
    python visualize_band_distribution.py --input classified_dataset.jsonl
    python visualize_band_distribution.py --input classified_dataset.jsonl --output band_distribution.png
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

# Band colors (matching curriculum difficulty progression)
BAND_COLORS = {
    "B0": "#d4edda",  # Light green - Nursery
    "B1": "#c3e6cb",  # Green - Primary
    "B2": "#fff4e1",  # Light orange - High School
    "B3": "#ffe4b5",  # Orange - Undergraduate
    "B4": "#f8d7da",  # Light red - Graduate
    "B5": "#f5c6cb",  # Red - PhD
}

BAND_NAMES = {
    "B0": "B0: Nursery",
    "B1": "B1: Primary",
    "B2": "B2: High School",
    "B3": "B3: Undergraduate",
    "B4": "B4: Graduate",
    "B5": "B5: PhD",
}


def load_classified_dataset(input_file: str, input_format: str = "jsonl") -> list:
    """
    Load classified dataset from file.

    Args:
        input_file: Path to input file
        input_format: Format of input file ('json' or 'jsonl')

    Returns:
        List of classified records
    """
    records = []

    if input_format == "jsonl":
        with open(input_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    else:  # json
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data
            else:
                records = [data]

    return records


def count_bands(records: list) -> dict:
    """
    Count records per band.

    Args:
        records: List of classified records

    Returns:
        Dictionary with band counts
    """
    band_counts = Counter()

    for record in records:
        if "curriculum_band" in record:
            band = record["curriculum_band"]
            band_counts[band] += 1
        else:
            print(
                f"Warning: Record missing 'curriculum_band' field: {record.get('id', 'unknown')}"
            )

    # Ensure all bands are present (even if count is 0)
    for band in ["B0", "B1", "B2", "B3", "B4", "B5"]:
        if band not in band_counts:
            band_counts[band] = 0

    return dict(band_counts)


def create_band_distribution_plot(
    band_counts: dict,
    output_file: str = None,
    title: str = "Curriculum Band Distribution",
):
    """
    Create visualization of band distribution.

    Args:
        band_counts: Dictionary with band counts
        output_file: Optional path to save the plot
        title: Plot title
    """
    bands = ["B0", "B1", "B2", "B3", "B4", "B5"]
    counts = [band_counts.get(band, 0) for band in bands]
    colors = [BAND_COLORS[band] for band in bands]
    # labels = [f"{BAND_NAMES[band]}\n({counts[i]:,})" for i, band in enumerate(bands)]

    total = sum(counts)
    percentages = [(count / total * 100) if total > 0 else 0 for count in counts]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    # Bar chart
    bars = ax1.bar(bands, counts, color=colors, edgecolor="black", linewidth=1.5)
    ax1.set_xlabel("Curriculum Band", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Records", fontsize=12, fontweight="bold")
    ax1.set_title("Band Distribution (Count)", fontsize=14, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Pie chart
    # Only show slices with non-zero values
    non_zero_data = [
        (bands[i], counts[i], colors[i]) for i in range(len(bands)) if counts[i] > 0
    ]
    if non_zero_data:
        pie_bands, pie_counts, pie_colors = zip(*non_zero_data)
        pie_labels = [
            f"{BAND_NAMES[band]}\n{count:,} ({count / total * 100:.1f}%)"
            for band, count in zip(pie_bands, pie_counts)
        ]

        wedges, texts, autotexts = ax2.pie(
            pie_counts,
            labels=pie_labels,
            colors=pie_colors,
            autopct="",
            startangle=90,
            textprops={"fontsize": 10, "fontweight": "bold"},
        )
        ax2.set_title("Band Distribution (Percentage)", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {output_file}")
    else:
        plt.show()

    return fig


def print_summary(band_counts: dict):
    """
    Print summary statistics.

    Args:
        band_counts: Dictionary with band counts
    """
    total = sum(band_counts.values())
    bands = ["B0", "B1", "B2", "B3", "B4", "B5"]

    print("\n" + "=" * 60)
    print("BAND DISTRIBUTION SUMMARY")
    print("=" * 60)
    print(f"{'Band':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 60)

    for band in bands:
        count = band_counts.get(band, 0)
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{BAND_NAMES[band]:<20} {count:<15,} {percentage:>6.2f}%")

    print("-" * 60)
    print(f"{'TOTAL':<20} {total:<15,} {'100.00%':>15}")
    print("=" * 60)


def visualize_dataset(
    input_file: str,
    output_file: str = None,
    input_format: str = "jsonl",
    title: str = None,
):
    """
    Main function to visualize band distribution from classified dataset.

    Args:
        input_file: Path to classified dataset file
        output_file: Optional path to save the plot
        input_format: Format of input file ('json' or 'jsonl')
        title: Optional custom title for the plot
    """
    print(f"Loading classified dataset from: {input_file}")

    # Load dataset
    records = load_classified_dataset(input_file, input_format)
    print(f"Loaded {len(records):,} records")

    if len(records) == 0:
        print("Error: No records found in dataset!")
        return

    # Count bands
    band_counts = count_bands(records)

    # Print summary
    print_summary(band_counts)

    # Create visualization
    if title is None:
        title = f"Curriculum Band Distribution\n({len(records):,} total records)"

    create_band_distribution_plot(band_counts, output_file, title)

    print("\n✓ Visualization complete!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize band distribution from classified dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to classified dataset file (JSON or JSONL)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot image (default: show interactively)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Input file format (default: jsonl)",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Custom title for the plot"
    )

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return

    # Visualize
    visualize_dataset(
        input_file=args.input,
        output_file=args.output,
        input_format=args.format,
        title=args.title,
    )


if __name__ == "__main__":
    main()
