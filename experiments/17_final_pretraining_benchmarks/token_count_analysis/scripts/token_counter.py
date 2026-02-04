"""
Token Counter Utility

This script provides utilities to count tokens in benchmark datasets.
Uses GPT-2 tokenizer as a baseline, but can be configured for other tokenizers.
"""

import argparse
import json
from typing import Any, Dict, List

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Missing dependencies. Install with: pip install transformers datasets")
    exit(1)


class TokenCounter:
    """Utility class for counting tokens in various data formats."""

    def __init__(self, tokenizer_name: str = "Xenova/gpt-4"):
        """
        Initialize token counter with specified tokenizer.

        Args:
            tokenizer_name: HuggingFace tokenizer name (default: Xenova/gpt-4 which uses cl100k_base)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

    def count_text(self, text: str) -> int:
        """Count tokens in a single text string."""
        return len(self.tokenizer.encode(text))

    def count_texts(self, texts: List[str]) -> Dict[str, Any]:
        """
        Count tokens across multiple texts.

        Returns:
            Dictionary with total, mean, min, max tokens
        """
        token_counts = [self.count_text(text) for text in texts]

        return {
            "total_tokens": sum(token_counts),
            "num_samples": len(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0,
            "token_counts": token_counts,
        }

    def count_dataset(
        self,
        dataset_name: str,
        split: str = "test",
        text_column: str = None,
        **load_kwargs,
    ) -> Dict[str, Any]:
        """
        Count tokens in a HuggingFace dataset.

        Args:
            dataset_name: Name of the dataset on HuggingFace
            split: Dataset split to analyze (default: test)
            text_column: Column containing text (auto-detect if None)
            **load_kwargs: Additional arguments for load_dataset

        Returns:
            Dictionary with token statistics
        """
        print(f"Loading dataset: {dataset_name} (split: {split})")

        try:
            dataset = load_dataset(dataset_name, split=split, **load_kwargs)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {"error": str(e)}

        # Auto-detect text column if not specified
        if text_column is None:
            text_column = self._detect_text_column(dataset)
            print(f"Auto-detected text column: {text_column}")

        # Extract all texts
        if text_column:
            texts = self._extract_texts(dataset, text_column)
        else:
            # For multi-column datasets, concatenate all text fields
            texts = self._extract_all_text(dataset)

        result = self.count_texts(texts)
        result["dataset_name"] = dataset_name
        result["split"] = split
        result["text_column"] = text_column
        result["tokenizer"] = self.tokenizer_name

        return result

    def _detect_text_column(self, dataset) -> str:
        """Auto-detect the main text column in a dataset."""
        # Common column names for text data
        common_names = [
            "text",
            "question",
            "prompt",
            "input",
            "content",
            "passage",
            "context",
            "sentence",
            "query",
        ]

        columns = dataset.column_names

        # Check for exact matches first
        for name in common_names:
            if name in columns:
                return name

        # Check for partial matches
        for name in common_names:
            for col in columns:
                if name in col.lower():
                    return col

        # Return first string column as fallback
        for col in columns:
            if isinstance(dataset[0][col], str):
                return col

        return None

    def _extract_texts(self, dataset, text_column: str) -> List[str]:
        """Extract texts from specified column."""
        texts = []
        for item in dataset:
            value = item[text_column]
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.extend([str(v) for v in value])
            else:
                texts.append(str(value))
        return texts

    def _extract_all_text(self, dataset) -> List[str]:
        """Extract all text content from a dataset (all columns)."""
        texts = []
        for item in dataset:
            combined_text = []
            for value in item.values():
                if isinstance(value, str):
                    combined_text.append(value)
                elif isinstance(value, (list, tuple)):
                    combined_text.extend([str(v) for v in value])
                else:
                    combined_text.append(str(value))
            texts.append(" ".join(combined_text))
        return texts

    def count_json_file(self, file_path: str, text_field: str = None) -> Dict[str, Any]:
        """
        Count tokens in a JSON file.

        Args:
            file_path: Path to JSON file
            text_field: Field containing text (auto-detect if None)

        Returns:
            Dictionary with token statistics
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if text_field:
                texts = [item[text_field] for item in data if text_field in item]
            else:
                # Concatenate all text fields
                texts = [json.dumps(item) for item in data]
        elif isinstance(data, dict):
            if text_field and text_field in data:
                texts = [data[text_field]]
            else:
                texts = [json.dumps(data)]
        else:
            texts = [str(data)]

        result = self.count_texts(texts)
        result["file_path"] = file_path
        result["tokenizer"] = self.tokenizer_name

        return result


def print_stats(stats: Dict[str, Any]):
    """Pretty print token statistics."""
    print("\n" + "=" * 60)
    print("TOKEN STATISTICS")
    print("=" * 60)

    if "error" in stats:
        print(f"Error: {stats['error']}")
        return

    if "dataset_name" in stats:
        print(f"Dataset: {stats['dataset_name']}")
        print(f"Split: {stats['split']}")
        if stats.get("text_column"):
            print(f"Text Column: {stats['text_column']}")

    if "file_path" in stats:
        print(f"File: {stats['file_path']}")

    print(f"Tokenizer: {stats['tokenizer']}")
    print(f"\nTotal Samples: {stats['num_samples']:,}")
    print(f"Total Tokens: {stats['total_tokens']:,}")
    print(f"Mean Tokens/Sample: {stats['mean_tokens']:.2f}")
    print(f"Min Tokens: {stats['min_tokens']:,}")
    print(f"Max Tokens: {stats['max_tokens']:,}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Count tokens in benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        help="HuggingFace dataset name (e.g., 'gsm8k', 'cais/mmlu')",
    )
    parser.add_argument("--file", type=str, help="Path to JSON file")
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--text-column",
        type=str,
        help="Column containing text (auto-detect if not specified)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default="gpt2", help="Tokenizer to use (default: gpt2)"
    )
    parser.add_argument("--output", type=str, help="Output JSON file path")
    parser.add_argument("--config", type=str, help="Additional dataset config name")

    args = parser.parse_args()

    counter = TokenCounter(tokenizer_name=args.tokenizer)

    if args.dataset:
        load_kwargs = {}
        if args.config:
            load_kwargs["name"] = args.config

        stats = counter.count_dataset(
            args.dataset, split=args.split, text_column=args.text_column, **load_kwargs
        )
    elif args.file:
        stats = counter.count_json_file(args.file, text_field=args.text_column)
    else:
        parser.print_help()
        return

    print_stats(stats)

    if args.output:
        # Remove token_counts list for cleaner output
        output_stats = {k: v for k, v in stats.items() if k != "token_counts"}
        with open(args.output, "w") as f:
            json.dump(output_stats, f, indent=2)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
