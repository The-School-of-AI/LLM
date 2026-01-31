"""
Frequency Analyzer - Compute token frequency statistics from large datasets.

Streams through HuggingFace datasets and builds frequency distributions
for frequency-aware token ID reindexing.

Usage:
    python frequency_analyzer.py \\
        --tokenizer ds_filtered \\
        --dataset indic \\
        --config ../config.yaml \\
        --output ../results/frequency_stats/ds_indic_freq.json
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from tqdm import tqdm
from loguru import logger

try:
    from datasets import load_dataset
except ImportError:
    logger.error("datasets library not installed. Run: pip install datasets")
    raise


@dataclass
class FrequencyStats:
    """Container for frequency statistics."""
    tokenizer_name: str
    dataset_name: str
    total_tokens: int
    unique_tokens: int
    vocab_size: int

    # Frequency distribution
    token_frequencies: Dict[int, int]  # token_id -> count

    # Statistics
    mean_frequency: float
    median_frequency: float
    std_frequency: float

    # Percentiles
    percentiles: Dict[str, float]  # e.g., "p50": freq_value

    # Classification
    head_tokens: List[int]  # top 10% by frequency
    torso_tokens: List[int]  # middle 40%
    tail_tokens: List[int]  # bottom 50%

    # Metadata
    bytes_processed: int
    samples_processed: int


class TokenizerWrapper:
    """Simple tokenizer wrapper for frequency computation."""

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        self.sorted_tokens = sorted(token_to_id.keys(), key=len, reverse=True)

    def encode(self, text: str) -> List[int]:
        """Greedy longest-match tokenization."""
        ids = []
        pos = 0
        while pos < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text[pos:pos+len(token)] == token:
                    ids.append(self.token_to_id[token])
                    pos += len(token)
                    matched = True
                    break
            if not matched:
                # Byte fallback
                char = text[pos]
                char_bytes = char.encode('utf-8')
                for byte in char_bytes:
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.token_to_id:
                        ids.append(self.token_to_id[byte_token])
                    else:
                        ids.append(0)  # Unknown token
                pos += 1
        return ids


class DatasetStreamer:
    """Streams data from HuggingFace datasets."""

    def __init__(self, config: Dict[str, Any], dataset_type: str):
        self.config = config
        self.dataset_type = dataset_type
        self.dataset_config = config['frequency_analysis']['datasets'][dataset_type]
        self.streaming_config = config['frequency_analysis']['streaming']

    def estimate_sample_count(self, target_gb: float) -> int:
        """Estimate number of samples to process for target GB."""
        # Rough estimate: 1KB per sample average
        bytes_target = target_gb * 1024 * 1024 * 1024
        return int(bytes_target / 1024)

    def stream_samples(self, max_samples: Optional[int] = None):
        """Stream samples from dataset."""
        dataset_name = self.dataset_config['name']
        subset = self.dataset_config.get('subset')
        split = self.dataset_config['split']
        text_column = self.dataset_config['text_column']

        logger.info(f"Loading dataset: {dataset_name} (streaming mode)")

        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                streaming=True,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_name}': {e}")
            raise

        sample_count = 0
        target_samples = max_samples or self.estimate_sample_count(
            self.dataset_config['sample_size_gb']
        )

        logger.info(f"Streaming up to {target_samples} samples...")

        for sample in dataset:
            if sample_count >= target_samples:
                break

            # Extract text
            if text_column in sample:
                text = sample[text_column]
                if isinstance(text, str) and text.strip():
                    yield text
                    sample_count += 1

            if sample_count % 1000 == 0:
                logger.debug(f"Processed {sample_count} samples")


class FrequencyAnalyzer:
    """Main frequency analyzer."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_path = Path(self.config['tokenizer_sources']['base_path'])

    def load_tokenizer(self, tokenizer_name: str) -> TokenizerWrapper:
        """Load tokenizer from JSON file."""
        json_path = self.base_path / f"{tokenizer_name}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)

        logger.info(f"Loaded tokenizer '{tokenizer_name}' with {len(token_to_id)} tokens")
        return TokenizerWrapper(token_to_id)

    def compute_frequency_distribution(
        self,
        tokenizer: TokenizerWrapper,
        dataset_type: str,
        max_samples: Optional[int] = None
    ) -> Dict[int, int]:
        """Compute token frequency distribution from dataset."""
        logger.info(f"Computing frequency distribution on '{dataset_type}' dataset")

        streamer = DatasetStreamer(self.config, dataset_type)
        frequency_counter = Counter()

        total_tokens = 0
        bytes_processed = 0
        samples_processed = 0

        for text in tqdm(streamer.stream_samples(max_samples), desc="Tokenizing"):
            # Tokenize
            token_ids = tokenizer.encode(text)

            # Update counters
            frequency_counter.update(token_ids)
            total_tokens += len(token_ids)
            bytes_processed += len(text.encode('utf-8'))
            samples_processed += 1

        logger.info(f"Processed {samples_processed} samples, {total_tokens} tokens, {bytes_processed/1e9:.2f} GB")

        return dict(frequency_counter), total_tokens, bytes_processed, samples_processed

    def apply_smoothing(self, frequencies: Dict[int, int], method: str, factor: float = 1.0) -> Dict[int, float]:
        """Apply smoothing to raw frequencies."""
        if method == "none":
            return {k: float(v) for k, v in frequencies.items()}
        elif method == "log":
            return {k: np.log(1 + v) * factor for k, v in frequencies.items()}
        elif method == "sqrt":
            return {k: np.sqrt(v) * factor for k, v in frequencies.items()}
        elif method == "power":
            return {k: (v ** factor) for k, v in frequencies.items()}
        else:
            logger.warning(f"Unknown smoothing method '{method}', using 'none'")
            return {k: float(v) for k, v in frequencies.items()}

    def compute_percentiles(self, frequencies: Dict[int, int], percentiles: List[int]) -> Dict[str, float]:
        """Compute frequency percentiles."""
        freq_values = list(frequencies.values())
        if not freq_values:
            return {}

        return {
            f"p{p}": float(np.percentile(freq_values, p))
            for p in percentiles
        }

    def classify_tokens(
        self,
        frequencies: Dict[int, int],
        head_percentile: float,
        torso_percentile: float
    ) -> Dict[str, List[int]]:
        """Classify tokens into head/torso/tail based on frequency."""
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

        total_tokens = len(sorted_tokens)
        head_cutoff = int(total_tokens * (1 - head_percentile / 100))
        torso_cutoff = int(total_tokens * (1 - torso_percentile / 100))

        head_tokens = [token_id for token_id, _ in sorted_tokens[:head_cutoff]]
        torso_tokens = [token_id for token_id, _ in sorted_tokens[head_cutoff:torso_cutoff]]
        tail_tokens = [token_id for token_id, _ in sorted_tokens[torso_cutoff:]]

        logger.info(f"Token classification: head={len(head_tokens)}, torso={len(torso_tokens)}, tail={len(tail_tokens)}")

        return {
            "head": head_tokens,
            "torso": torso_tokens,
            "tail": tail_tokens
        }

    def analyze(
        self,
        tokenizer_name: str,
        dataset_type: str,
        max_samples: Optional[int] = None,
        output_path: Optional[str] = None
    ) -> FrequencyStats:
        """Run full frequency analysis."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Frequency Analysis: {tokenizer_name} on {dataset_type}")
        logger.info(f"{'='*60}")

        # Load tokenizer
        tokenizer = self.load_tokenizer(tokenizer_name)

        # Compute frequency distribution
        frequencies, total_tokens, bytes_processed, samples_processed = \
            self.compute_frequency_distribution(tokenizer, dataset_type, max_samples)

        # Apply smoothing
        smoothing_config = self.config['frequency_analysis']['computation']
        smoothed_frequencies = self.apply_smoothing(
            frequencies,
            smoothing_config['smoothing'],
            smoothing_config['smoothing_factor']
        )

        # Compute statistics
        freq_values = list(frequencies.values())
        mean_freq = np.mean(freq_values) if freq_values else 0.0
        median_freq = np.median(freq_values) if freq_values else 0.0
        std_freq = np.std(freq_values) if freq_values else 0.0

        # Compute percentiles
        percentiles = self.compute_percentiles(
            frequencies,
            smoothing_config['percentiles']
        )

        # Classify tokens
        classification = self.classify_tokens(
            frequencies,
            smoothing_config['classification']['head_percentile'],
            smoothing_config['classification']['torso_percentile']
        )

        # Create stats object
        stats = FrequencyStats(
            tokenizer_name=tokenizer_name,
            dataset_name=dataset_type,
            total_tokens=total_tokens,
            unique_tokens=len(frequencies),
            vocab_size=len(tokenizer.token_to_id),
            token_frequencies=frequencies,
            mean_frequency=float(mean_freq),
            median_frequency=float(median_freq),
            std_frequency=float(std_freq),
            percentiles=percentiles,
            head_tokens=classification["head"],
            torso_tokens=classification["torso"],
            tail_tokens=classification["tail"],
            bytes_processed=bytes_processed,
            samples_processed=samples_processed
        )

        # Save results
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(asdict(stats), f, indent=2)

            logger.info(f"Frequency stats saved to: {output_file}")

        logger.info(f"\nStatistics:")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Unique tokens: {len(frequencies):,}")
        logger.info(f"  Mean frequency: {mean_freq:.2f}")
        logger.info(f"  Median frequency: {median_freq:.2f}")
        logger.info(f"  Head tokens: {len(classification['head']):,}")
        logger.info(f"  Torso tokens: {len(classification['torso']):,}")
        logger.info(f"  Tail tokens: {len(classification['tail']):,}")

        return stats

    def merge_frequency_stats(
        self,
        stats_list: List[FrequencyStats],
        output_path: Optional[str] = None
    ) -> FrequencyStats:
        """Merge multiple frequency statistics (e.g., from different datasets)."""
        if not stats_list:
            raise ValueError("No stats to merge")

        logger.info(f"Merging {len(stats_list)} frequency stats...")

        # Merge frequencies
        merged_frequencies = Counter()
        total_tokens = 0
        bytes_processed = 0
        samples_processed = 0

        for stats in stats_list:
            merged_frequencies.update(stats.token_frequencies)
            total_tokens += stats.total_tokens
            bytes_processed += stats.bytes_processed
            samples_processed += stats.samples_processed

        # Recompute statistics
        freq_values = list(merged_frequencies.values())
        mean_freq = np.mean(freq_values)
        median_freq = np.median(freq_values)
        std_freq = np.std(freq_values)

        # Recompute percentiles
        smoothing_config = self.config['frequency_analysis']['computation']
        percentiles = self.compute_percentiles(
            merged_frequencies,
            smoothing_config['percentiles']
        )

        # Reclassify tokens
        classification = self.classify_tokens(
            merged_frequencies,
            smoothing_config['classification']['head_percentile'],
            smoothing_config['classification']['torso_percentile']
        )

        # Create merged stats
        merged_stats = FrequencyStats(
            tokenizer_name=stats_list[0].tokenizer_name,
            dataset_name="merged",
            total_tokens=total_tokens,
            unique_tokens=len(merged_frequencies),
            vocab_size=stats_list[0].vocab_size,
            token_frequencies=dict(merged_frequencies),
            mean_frequency=float(mean_freq),
            median_frequency=float(median_freq),
            std_frequency=float(std_freq),
            percentiles=percentiles,
            head_tokens=classification["head"],
            torso_tokens=classification["torso"],
            tail_tokens=classification["tail"],
            bytes_processed=bytes_processed,
            samples_processed=samples_processed
        )

        # Save if output path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w') as f:
                json.dump(asdict(merged_stats), f, indent=2)

            logger.info(f"Merged frequency stats saved to: {output_file}")

        return merged_stats


def main():
    parser = argparse.ArgumentParser(description="Token Frequency Analyzer")
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Tokenizer name (e.g., ds_filtered)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['indic', 'code'],
                        help='Dataset type (indic or code)')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for frequency stats JSON')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--merge', action='store_true',
                        help='Merge stats from multiple datasets')

    args = parser.parse_args()

    analyzer = FrequencyAnalyzer(args.config)

    if args.merge:
        # Load and merge existing stats
        logger.info("Merge mode: loading existing stats...")
        # Implementation for merging would go here
        pass
    else:
        # Analyze single dataset
        stats = analyzer.analyze(
            tokenizer_name=args.tokenizer,
            dataset_type=args.dataset,
            max_samples=args.max_samples,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
