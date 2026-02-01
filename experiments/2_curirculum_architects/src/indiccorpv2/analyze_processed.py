#!/usr/bin/env python3
"""
Analyze Processed IndicCorpV2 Data

This script analyzes the output of the cleaning pipeline to:
- Verify difficulty distributions
- Check quality of classification
- Generate reports for each language
- Compare across languages
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd


class ProcessedDataAnalyzer:
    """Analyze processed IndicCorpV2 data"""

    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)

    def analyze_language(self, lang_dir: Path) -> Dict:
        """Analyze a single language"""

        lang_name = lang_dir.name
        print(f"\n{'='*70}")
        print(f"Analyzing: {lang_name}")
        print(f"{'='*70}")

        # Load processing stats
        stats_file = lang_dir / "processing_stats.json"
        if not stats_file.exists():
            print(f"⚠️  No stats file found for {lang_name}")
            return {}

        with open(stats_file) as f:
            stats = json.load(f)

        # Count documents per difficulty
        doc_counts = {}
        total_docs = 0

        for difficulty in ["B0", "B1", "B2", "B3", "B4"]:
            diff_dir = lang_dir / difficulty
            if not diff_dir.exists():
                doc_counts[difficulty] = 0
                continue

            # Count lines in all jsonl files
            count = 0
            for jsonl_file in diff_dir.glob("*.jsonl"):
                with open(jsonl_file) as f:
                    count += sum(1 for _ in f)

            doc_counts[difficulty] = count
            total_docs += count

        # Sample analysis: Read some documents for quality check
        samples = self._sample_documents(lang_dir, n_per_level=5)

        # Compile results
        result = {
            "language": lang_name,
            "stats": stats,
            "document_counts": doc_counts,
            "total_documents": total_docs,
            "samples": samples,
        }

        # Print summary
        self._print_language_summary(result)

        return result

    def _sample_documents(self, lang_dir: Path, n_per_level: int = 5) -> Dict:
        """Sample documents from each difficulty level"""

        samples = {}

        for difficulty in ["B0", "B1", "B2", "B3", "B4"]:
            diff_dir = lang_dir / difficulty
            if not diff_dir.exists():
                continue

            # Read first batch file
            batch_files = list(diff_dir.glob("*.jsonl"))
            if not batch_files:
                continue

            samples[difficulty] = []

            with open(batch_files[0]) as f:
                for i, line in enumerate(f):
                    if i >= n_per_level:
                        break

                    doc = json.loads(line)
                    samples[difficulty].append(
                        {
                            "text_preview": doc["text"][:200],
                            "word_count": doc["word_count"],
                            "category": doc["category"],
                            "confidence": doc["difficulty_confidence"],
                        }
                    )

        return samples

    def _print_language_summary(self, result: Dict):
        """Print summary for a language"""

        stats = result["stats"]
        counts = result["document_counts"]
        total = result["total_documents"]

        print("\nProcessing Statistics:")
        print(f"  Input documents: {stats.get('total_processed', 0):,}")
        print(f"  Output documents: {stats.get('kept', 0):,}")
        print(f"  Pass rate: {stats.get('pass_rate', 0)*100:.1f}%")
        print(f"  Duplicates removed: {stats.get('duplicates_removed', 0):,}")

        print("\nDifficulty Distribution:")
        for diff in ["B0", "B1", "B2", "B3", "B4"]:
            count = counts.get(diff, 0)
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {diff}: {count:8,} ({pct:5.1f}%)")

        print("\nCategory Distribution:")
        if "category_distribution" in stats:
            for cat, count in sorted(
                stats["category_distribution"].items(), key=lambda x: x[1], reverse=True
            ):
                pct = (count / total * 100) if total > 0 else 0
                print(f"  {cat:20s}: {count:6,} ({pct:5.1f}%)")

        print("\nTop Quality Failures:")
        if "quality_failures" in stats:
            for reason, count in sorted(
                stats["quality_failures"].items(), key=lambda x: x[1], reverse=True
            )[:5]:
                pct = (
                    (count / stats["total_processed"] * 100)
                    if stats["total_processed"] > 0
                    else 0
                )
                print(f"  {reason:25s}: {count:6,} ({pct:5.1f}%)")

    def analyze_all(self) -> List[Dict]:
        """Analyze all processed languages"""

        results = []

        # Find all language directories
        lang_dirs = [d for d in self.processed_dir.iterdir() if d.is_dir()]

        if not lang_dirs:
            print(f"⚠️  No language directories found in {self.processed_dir}")
            return results

        print(f"\nFound {len(lang_dirs)} language(s) to analyze\n")

        for lang_dir in sorted(lang_dirs):
            result = self.analyze_language(lang_dir)
            if result:
                results.append(result)

        # Generate comparison report
        if len(results) > 1:
            self._generate_comparison(results)

        return results

    def _generate_comparison(self, results: List[Dict]):
        """Generate comparison across languages"""

        print(f"\n{'='*70}")
        print("CROSS-LANGUAGE COMPARISON")
        print(f"{'='*70}\n")

        # Create comparison dataframe
        comparison_data = []

        for result in results:
            counts = result["document_counts"]
            stats = result["stats"]
            total = result["total_documents"]

            comparison_data.append(
                {
                    "Language": result["language"],
                    "Input Docs": stats.get("total_processed", 0),
                    "Output Docs": total,
                    "Pass Rate": f"{stats.get('pass_rate', 0)*100:.1f}%",
                    "B0": counts.get("B0", 0),
                    "B1": counts.get("B1", 0),
                    "B2": counts.get("B2", 0),
                    "B3": counts.get("B3", 0),
                    "B4": counts.get("B4", 0),
                }
            )

        df = pd.DataFrame(comparison_data)

        print(df.to_string(index=False))

        # Save to CSV
        output_file = self.processed_dir / "comparison_report.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Comparison report saved to: {output_file}")

        # Generate difficulty distribution percentages
        print("\nDifficulty Distribution (%):")
        print(f"{'Language':<15} {'B0':>6} {'B1':>6} {'B2':>6} {'B3':>6} {'B4':>6}")
        print("-" * 51)

        for _, row in df.iterrows():
            total = row["B0"] + row["B1"] + row["B2"] + row["B3"] + row["B4"]
            if total == 0:
                continue
            print(
                f"{row['Language']:<15} "
                f"{row['B0']/total*100:>5.1f}% "
                f"{row['B1']/total*100:>5.1f}% "
                f"{row['B2']/total*100:>5.1f}% "
                f"{row['B3']/total*100:>5.1f}% "
                f"{row['B4']/total*100:>5.1f}%"
            )

    def quality_check(self, lang_dir: Path, n_samples: int = 20):
        """Perform quality check on processed data"""

        print(f"\n{'='*70}")
        print(f"QUALITY CHECK: {lang_dir.name}")
        print(f"{'='*70}\n")

        for difficulty in ["B0", "B1", "B2", "B3", "B4"]:
            diff_dir = lang_dir / difficulty
            if not diff_dir.exists():
                continue

            print(f"\n--- {difficulty} Samples ---\n")

            batch_files = list(diff_dir.glob("*.jsonl"))
            if not batch_files:
                print("  No data found")
                continue

            # Read samples
            samples = []
            with open(batch_files[0]) as f:
                for i, line in enumerate(f):
                    if i >= n_samples:
                        break
                    samples.append(json.loads(line))

            # Analyze samples
            word_counts = [s["word_count"] for s in samples]
            confidences = [s["difficulty_confidence"] for s in samples]
            categories = Counter(s["category"] for s in samples)

            print(
                f"  Word count: min={min(word_counts)}, "
                f"avg={sum(word_counts)/len(word_counts):.0f}, "
                f"max={max(word_counts)}"
            )
            print(
                f"  Confidence: min={min(confidences):.2f}, "
                f"avg={sum(confidences)/len(confidences):.2f}, "
                f"max={max(confidences):.2f}"
            )
            print(f"  Categories: {dict(categories)}")

            # Print 2 sample texts
            print("\n  Sample texts:")
            for i, sample in enumerate(samples[:2]):
                print(
                    f"\n  [{i+1}] ({sample['word_count']} words, "
                    f"category={sample['category']}, "
                    f"conf={sample['difficulty_confidence']:.2f})"
                )
                print(f"  {sample['text'][:250]}...")


def main():
    parser = argparse.ArgumentParser(description="Analyze processed IndicCorpV2 data")
    parser.add_argument(
        "--processed-dir", required=True, help="Directory containing processed data"
    )
    parser.add_argument(
        "--language",
        help="Specific language to analyze (e.g., hi_Deva). If not provided, analyzes all.",
    )
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Perform detailed quality check with samples",
    )

    args = parser.parse_args()

    analyzer = ProcessedDataAnalyzer(args.processed_dir)

    if args.language:
        # Analyze specific language
        lang_dir = Path(args.processed_dir) / args.language
        if not lang_dir.exists():
            print(f"❌ Language directory not found: {lang_dir}")
            return

        analyzer.analyze_language(lang_dir)

        if args.quality_check:
            analyzer.quality_check(lang_dir)
    else:
        # Analyze all languages
        results = analyzer.analyze_all()

        if args.quality_check and results:
            # Quality check on first language
            lang_dir = Path(args.processed_dir) / results[0]["language"]
            analyzer.quality_check(lang_dir)

    print(f"\n{'='*70}")
    print("✅ Analysis complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
