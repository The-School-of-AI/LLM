"""
Curriculum Band Classifier for LLM Pretraining Datasets

This script classifies dataset records into B0-B5 difficulty bands based on
token ID frequencies from Meta's Llama 3.3 70B tokenizer (BPE).

Key Principle: Token ID is inversely related to frequency
- High frequency tokens → Low token IDs (0-1000) → B0 (Nursery)
- Low frequency tokens → High token IDs (10000+) → B5 (PhD)
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


class CurriculumBandClassifier:
    """
    Classifies text samples into B0-B5 curriculum bands based on token ID statistics.

    Bands:
    - B0 (Nursery): Very high frequency tokens, simple language
    - B1 (Primary): High frequency tokens, everyday language
    - B2 (High School): Medium frequency, structured knowledge
    - B3 (Undergraduate): Lower frequency, technical content
    - B4 (Graduate): Low frequency, complex reasoning
    - B5 (PhD): Very low frequency, advanced/rare terms
    """

    # Token ID thresholds for band classification
    # Based on inverse frequency relationship: low ID = high frequency
    # NOTE: Updated for Llama 3.3 tokenizer (vocab size ~128K)
    # Common words in Llama 3.3 typically have IDs in thousands, not hundreds
    TOKEN_ID_THRESHOLDS = {
        "B0": {
            "avg_max": 5000,  # Average token ID should be < 5000 (high frequency)
            "max_max": 10000,  # Maximum token ID should be < 10000
            "p95_max": 8000,  # 95th percentile should be < 8000
            "description": "Nursery: Very high frequency tokens, simple language",
        },
        "B1": {
            "avg_max": 10000,  # Average token ID should be < 10000
            "max_max": 20000,  # Maximum token ID should be < 20000
            "p95_max": 15000,  # 95th percentile should be < 15000
            "description": "Primary: High frequency tokens, everyday language",
        },
        "B2": {
            "avg_max": 20000,  # Average token ID should be < 20000
            "max_max": 40000,  # Maximum token ID should be < 40000
            "p95_max": 30000,  # 95th percentile should be < 30000
            "description": "High School: Medium frequency, structured knowledge",
        },
        "B3": {
            "avg_max": 40000,  # Average token ID should be < 40000
            "max_max": 70000,  # Maximum token ID should be < 70000
            "p95_max": 60000,  # 95th percentile should be < 60000
            "description": "Undergraduate: Lower frequency, technical content",
        },
        "B4": {
            "avg_max": 70000,  # Average token ID should be < 70000
            "max_max": 100000,  # Maximum token ID should be < 100000
            "p95_max": 90000,  # 95th percentile should be < 90000
            "description": "Graduate: Low frequency, complex reasoning",
        },
        "B5": {
            "avg_max": float("inf"),  # No upper limit
            "max_max": float("inf"),  # No upper limit
            "p95_max": float("inf"),  # No upper limit
            "description": "PhD: Very low frequency, advanced/rare terms",
        },
    }

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        local_tokenizer_path: str = None,
    ):
        """
        Initialize the classifier with a tokenizer.

        Args:
            model_id: HuggingFace model ID for Llama tokenizer (used if local_tokenizer_path is None)
            local_tokenizer_path: Optional local path to tokenizer files
        """
        print("Loading tokenizer...")
        try:
            if local_tokenizer_path:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_tokenizer_path, use_fast=True, local_files_only=True
                )
                print(f"Loaded tokenizer from local path: {local_tokenizer_path}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"Loaded tokenizer from HuggingFace: {model_id}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        print("Tokenizer loaded successfully!")

        # Verify it's a BPE tokenizer
        if hasattr(self.tokenizer, "backend_tokenizer"):
            print(f"Tokenizer type: {type(self.tokenizer.backend_tokenizer).__name__}")

        # Get vocabulary size for reference
        vocab_size = (
            len(self.tokenizer.get_vocab())
            if hasattr(self.tokenizer, "get_vocab")
            else "unknown"
        )
        print(
            f"Vocabulary size: {vocab_size:,}"
            if isinstance(vocab_size, int)
            else f"Vocabulary size: {vocab_size}"
        )

    def tokenize_text(self, text: str) -> List[int]:
        """
        Tokenize text and return token IDs.

        Args:
            text: Input text string

        Returns:
            List of token IDs
        """
        # Tokenize and convert to IDs
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens

    def calculate_token_stats(self, token_ids: List[int]) -> Dict[str, float]:
        """
        Calculate statistics about token IDs.

        Args:
            token_ids: List of token IDs

        Returns:
            Dictionary with statistics: avg, max, min, p50, p95, p99
        """
        if not token_ids:
            return {
                "avg": 0.0,
                "max": 0,
                "min": 0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "count": 0,
            }

        token_array = np.array(token_ids)

        return {
            "avg": float(np.mean(token_array)),
            "max": int(np.max(token_array)),
            "min": int(np.min(token_array)),
            "p50": float(np.percentile(token_array, 50)),
            "p95": float(np.percentile(token_array, 95)),
            "p99": float(np.percentile(token_array, 99)),
            "count": len(token_ids),
        }

    def classify_band(
        self, token_stats: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Classify a sample into B0-B5 band based on token statistics.

        Args:
            token_stats: Dictionary with token ID statistics

        Returns:
            Tuple of (band_name, classification_metadata)
        """
        avg_id = token_stats["avg"]
        max_id = token_stats["max"]
        p95_id = token_stats["p95"]

        # Check bands from easiest (B0) to hardest (B5)
        # We want the highest band that the sample qualifies for
        for band in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            thresholds = self.TOKEN_ID_THRESHOLDS[band]

            # Check if sample fits this band
            if (
                avg_id <= thresholds["avg_max"]
                and max_id <= thresholds["max_max"]
                and p95_id <= thresholds["p95_max"]
            ):
                metadata = {
                    "band": band,
                    "description": thresholds["description"],
                    "avg_token_id": avg_id,
                    "max_token_id": max_id,
                    "p95_token_id": p95_id,
                    "reason": f"avg={avg_id:.1f} <= {thresholds['avg_max']}, "
                    f"max={max_id} <= {thresholds['max_max']}, "
                    f"p95={p95_id:.1f} <= {thresholds['p95_max']}",
                }
                return band, metadata

        # Fallback to B5 if nothing matches (shouldn't happen with current thresholds)
        return "B5", {
            "band": "B5",
            "description": "PhD: Very low frequency, advanced/rare terms",
            "avg_token_id": avg_id,
            "max_token_id": max_id,
            "p95_token_id": p95_id,
            "reason": "Exceeded all thresholds, classified as B5",
        }

    def process_record(
        self, record: Dict[str, Any], text_field: str = "text"
    ) -> Dict[str, Any]:
        """
        Process a single dataset record and classify it.

        Args:
            record: Dictionary containing the dataset record
            text_field: Name of the field containing the text to classify

        Returns:
            Record with added classification metadata, or rejection metadata if token count < 4096
        """
        # Extract text from record
        if text_field not in record:
            raise ValueError(
                f"Field '{text_field}' not found in record. Available fields: {list(record.keys())}"
            )

        text = record[text_field]

        # Handle different text formats
        if isinstance(text, list):
            text = " ".join(text)
        elif not isinstance(text, str):
            text = str(text)

        # Tokenize
        token_ids = self.tokenize_text(text)

        # Count tokens
        token_count = len(token_ids)

        # Check if token count meets minimum requirement
        MIN_TOKEN_COUNT = 4096
        if token_count < MIN_TOKEN_COUNT:
            # Return rejected record with metadata
            result = record.copy()
            result["rejected"] = True
            result["rejection_reason"] = (
                f"Token count ({token_count}) below minimum threshold ({MIN_TOKEN_COUNT})"
            )
            result["token_count"] = token_count
            result["token_stats"] = self.calculate_token_stats(token_ids)
            return result

        # Calculate statistics
        token_stats = self.calculate_token_stats(token_ids)

        # Classify band
        band, metadata = self.classify_band(token_stats)

        # Add classification to record
        result = record.copy()
        result["curriculum_band"] = band
        result["token_stats"] = token_stats
        result["classification_metadata"] = metadata
        result["rejected"] = False

        return result

    def process_dataset(
        self,
        input_file: str,
        output_file: str,
        text_field: str = "text",
        input_format: str = "jsonl",
        rejected_output_file: str = None,
    ) -> Dict[str, int]:
        """
        Process entire dataset file and classify all records.

        Args:
            input_file: Path to input dataset file
            output_file: Path to output classified dataset file
            text_field: Name of the field containing text
            input_format: Format of input file ('json' or 'jsonl')
            rejected_output_file: Optional path to output rejected records file.
                                 If None, will be derived from output_file.

        Returns:
            Dictionary with band distribution counts and rejected count
        """
        print(f"\nProcessing dataset: {input_file}")
        print(f"Text field: '{text_field}'")
        print(f"Input format: {input_format}")

        # Determine rejected output file path
        if rejected_output_file is None:
            output_path = Path(output_file)
            rejected_output_file = str(
                output_path.parent / f"{output_path.stem}_rejected{output_path.suffix}"
            )

        # Read input file
        records = []
        if input_format == "jsonl":
            with open(input_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:  # json
            with open(input_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]

        print(f"Found {len(records)} records to process\n")

        # Process each record
        classified_records = []
        rejected_records = []
        band_counts = {"B0": 0, "B1": 0, "B2": 0, "B3": 0, "B4": 0, "B5": 0}
        rejected_count = 0

        for record in tqdm(records, desc="Classifying records"):
            try:
                processed = self.process_record(record, text_field)

                # Check if record was rejected
                if processed.get("rejected", False):
                    rejected_records.append(processed)
                    rejected_count += 1
                else:
                    classified_records.append(processed)
                    band = processed["curriculum_band"]
                    band_counts[band] += 1
            except Exception as e:
                print(f"\nError processing record: {e}")
                print(f"Record: {record}")
                continue

        # Write output file for classified records
        print(f"\nWriting classified dataset to: {output_file}")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_file.endswith(".jsonl"):
            with open(output_file, "w", encoding="utf-8") as f:
                for record in classified_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:  # json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(classified_records, f, ensure_ascii=False, indent=2)

        # Write rejected records file
        if rejected_records:
            print(f"Writing rejected records to: {rejected_output_file}")
            rejected_path = Path(rejected_output_file)
            rejected_path.parent.mkdir(parents=True, exist_ok=True)

            if rejected_output_file.endswith(".jsonl"):
                with open(rejected_output_file, "w", encoding="utf-8") as f:
                    for record in rejected_records:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:  # json
                with open(rejected_output_file, "w", encoding="utf-8") as f:
                    json.dump(rejected_records, f, ensure_ascii=False, indent=2)
            print(f"Saved {rejected_count} rejected records")

        print("Done!\n")

        # Print summary
        print("=" * 60)
        print("CLASSIFICATION SUMMARY")
        print("=" * 60)
        total = len(classified_records)
        for band in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            count = band_counts[band]
            percentage = (count / total * 100) if total > 0 else 0
            desc = self.TOKEN_ID_THRESHOLDS[band]["description"]
            print(f"{band:3s}: {count:6d} ({percentage:5.1f}%) - {desc}")
        print("=" * 60)
        print(f"Accepted: {total} records")
        print(
            f"Rejected: {rejected_count} records (token count below minimum threshold)"
        )
        print(f"Total processed: {len(records)} records")
        print("=" * 60)

        # Add rejected count to return dictionary
        band_counts["rejected"] = rejected_count
        return band_counts

    def visualize_band_distribution(
        self,
        classified_file: str,
        output_file: str = None,
        input_format: str = "jsonl",
        title: str = None,
    ):
        """
        Visualize band distribution from a classified dataset file.

        Args:
            classified_file: Path to classified dataset file
            output_file: Optional path to save the plot (default: show interactively)
            input_format: Format of input file ('json' or 'jsonl')
            title: Optional custom title for the plot

        Returns:
            matplotlib figure object
        """
        try:
            from collections import Counter

            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib is required for visualization.")
            print("Install it with: pip install matplotlib")
            return None

        # Load classified dataset
        records = []
        if input_format == "jsonl":
            with open(classified_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
        else:  # json
            with open(classified_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                records = data if isinstance(data, list) else [data]

        # Count bands
        band_counts = Counter()
        for record in records:
            if "curriculum_band" in record:
                band_counts[record["curriculum_band"]] += 1

        # Ensure all bands present
        for band in ["B0", "B1", "B2", "B3", "B4", "B5"]:
            if band not in band_counts:
                band_counts[band] = 0

        # Colors
        colors = {
            "B0": "#d4edda",
            "B1": "#c3e6cb",
            "B2": "#fff4e1",
            "B3": "#ffe4b5",
            "B4": "#f8d7da",
            "B5": "#f5c6cb",
        }
        band_names = {
            "B0": "B0: Nursery",
            "B1": "B1: Primary",
            "B2": "B2: High School",
            "B3": "B3: Undergraduate",
            "B4": "B4: Graduate",
            "B5": "B5: PhD",
        }

        bands = ["B0", "B1", "B2", "B3", "B4", "B5"]
        counts = [band_counts[band] for band in bands]
        total = sum(counts)
        percentages = [(c / total * 100) if total > 0 else 0 for c in counts]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        if title is None:
            title = f"Curriculum Band Distribution\n({total:,} total records)"
        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

        # Bar chart
        bars = ax1.bar(
            bands,
            counts,
            color=[colors[b] for b in bands],
            edgecolor="black",
            linewidth=1.5,
        )
        ax1.set_xlabel("Curriculum Band", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of Records", fontsize=12, fontweight="bold")
        ax1.set_title("Band Distribution (Count)", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        for bar, count, pct in zip(bars, counts, percentages):
            if bar.get_height() > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        # Pie chart
        non_zero = [(i, counts[i]) for i in range(len(bands)) if counts[i] > 0]
        if non_zero:
            indices, pie_counts = zip(*non_zero)
            pie_bands = [bands[i] for i in indices]
            pie_colors = [colors[b] for b in pie_bands]
            pie_labels = [
                f"{band_names[b]}\n{c:,} ({c / total * 100:.1f}%)"
                for b, c in zip(pie_bands, pie_counts)
            ]

            ax2.pie(
                pie_counts,
                labels=pie_labels,
                colors=pie_colors,
                autopct="",
                startangle=90,
                textprops={"fontsize": 10, "fontweight": "bold"},
            )
        ax2.set_title("Band Distribution (Percentage)", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"\n✓ Visualization saved to: {output_file}")
        else:
            plt.show()

        return fig


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Classify dataset records into B0-B5 curriculum bands based on token ID frequencies"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input dataset file (JSON or JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output classified dataset file",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help='Name of the field containing text to classify (default: "text")',
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="HuggingFace model ID for tokenizer (default: meta-llama/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--local-tokenizer",
        type=str,
        default=None,
        help="Local path to tokenizer files (overrides --model-id)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "jsonl"],
        default="jsonl",
        help="Input file format: json or jsonl (default: jsonl)",
    )

    args = parser.parse_args()

    # Initialize classifier
    classifier = CurriculumBandClassifier(
        model_id=args.model_id, local_tokenizer_path=args.local_tokenizer
    )

    # Process dataset
    classifier.process_dataset(
        input_file=args.input,
        output_file=args.output,
        text_field=args.text_field,
        input_format=args.format,
    )


if __name__ == "__main__":
    main()
