"""
Token ID Reindexer - Frequency-aware token ID remapping.

Takes a tokenizer and frequency statistics, then creates a new ID mapping
where IDs are ordered by frequency (with optional category blocks and smoothing).

Token strings and merge rules are preserved; only IDs change.

Usage:
    python id_reindexer.py \\
        --tokenizer ds_filtered \\
        --frequency-stats ../results/frequency_stats/ds_merged_freq.json \\
        --config ../config.yaml \\
        --output ../results/reindexed_tokenizers/ds_reindexed/
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger


@dataclass
class ReindexingResult:
    """Container for reindexing results."""
    tokenizer_name: str
    strategy: str
    vocab_size: int

    # Mappings
    old_id_to_new_id: Dict[int, int]
    new_id_to_token: Dict[int, str]
    token_to_new_id: Dict[str, int]

    # ID ranges (for category blocks strategy)
    id_ranges: Dict[str, Dict[str, int]]  # category -> {start, end}

    # Metadata
    special_tokens_count: int
    head_tokens_count: int
    torso_tokens_count: int
    tail_tokens_count: int


class SpecialTokenManager:
    """Manages special token allocation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.special_tokens = self._load_special_tokens()

    def _load_special_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Load special tokens from config."""
        all_special_tokens = {}

        for category, tokens in self.config['special_tokens'].items():
            for token_def in tokens:
                name = token_def['name']
                token_str = token_def['token']
                token_id = token_def['id']

                all_special_tokens[token_str] = {
                    'name': name,
                    'id': token_id,
                    'category': category
                }

        return all_special_tokens

    def get_reserved_ids(self) -> List[int]:
        """Get list of reserved special token IDs."""
        return [token['id'] for token in self.special_tokens.values()]

    def get_special_token_mapping(self) -> Dict[str, int]:
        """Get token_str -> ID mapping for special tokens."""
        return {token_str: token['id'] for token_str, token in self.special_tokens.items()}

    def allocate_special_tokens(
        self,
        existing_vocab: Dict[str, int]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Allocate special tokens, separating them from existing vocab.

        Returns:
            - special_token_mapping: special tokens with their reserved IDs
            - regular_vocab: existing vocab tokens (excluding special tokens)
        """
        special_mapping = self.get_special_token_mapping()
        regular_vocab = {k: v for k, v in existing_vocab.items() if k not in special_mapping}

        logger.info(f"Allocated {len(special_mapping)} special tokens")
        logger.info(f"Regular vocab size: {len(regular_vocab)}")

        return special_mapping, regular_vocab


class FrequencyBasedReindexer:
    """Reindexes token IDs based on frequency."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reindexing_config = config['reindexing']
        self.special_token_manager = SpecialTokenManager(config)

    def load_tokenizer(self, tokenizer_name: str) -> Dict[str, int]:
        """Load tokenizer vocabulary."""
        base_path = Path(self.config['tokenizer_sources']['base_path'])
        json_path = base_path / f"{tokenizer_name}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)

        logger.info(f"Loaded tokenizer '{tokenizer_name}' with {len(token_to_id)} tokens")
        return token_to_id

    def load_frequency_stats(self, stats_path: str) -> Dict[int, int]:
        """Load frequency statistics."""
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Convert string keys back to ints
        token_frequencies = {int(k): v for k, v in stats['token_frequencies'].items()}

        logger.info(f"Loaded frequency stats with {len(token_frequencies)} tokens")
        return token_frequencies

    def apply_moe_smoothing(self, frequencies: Dict[int, int]) -> Dict[int, float]:
        """Apply smoothing to reduce MoE routing skew."""
        if not self.reindexing_config['moe_smoothing']['enabled']:
            return {k: float(v) for k, v in frequencies.items()}

        method = self.reindexing_config['moe_smoothing']['method']
        temperature = self.reindexing_config['moe_smoothing']['temperature']

        if method == "log_smooth":
            return {k: np.log(1 + v) * temperature for k, v in frequencies.items()}
        else:
            logger.warning(f"Unknown smoothing method '{method}', using raw frequencies")
            return {k: float(v) for k, v in frequencies.items()}

    def strategy_pure_frequency(
        self,
        token_to_id: Dict[str, int],
        frequencies: Dict[int, int]
    ) -> ReindexingResult:
        """Pure frequency-based reindexing (most common = ID 0)."""
        logger.info("Using pure frequency strategy")

        # Separate special tokens
        special_mapping, regular_vocab = self.special_token_manager.allocate_special_tokens(token_to_id)

        # Get frequencies for regular tokens
        id_to_token = {v: k for k, v in regular_vocab.items()}
        token_freqs = [(old_id, frequencies.get(old_id, 0)) for old_id in id_to_token.keys()]

        # Sort by frequency (descending)
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        # Assign new IDs (special tokens first, then frequency-sorted)
        old_id_to_new_id = {}
        new_id_to_token = {}
        token_to_new_id = {}

        # Assign special tokens
        for token_str, new_id in special_mapping.items():
            new_id_to_token[new_id] = token_str
            token_to_new_id[token_str] = new_id
            # Note: special tokens may not have old IDs in the original vocab

        # Assign regular tokens
        next_id = 256  # Start after special token range
        for old_id, freq in token_freqs:
            token_str = id_to_token[old_id]
            old_id_to_new_id[old_id] = next_id
            new_id_to_token[next_id] = token_str
            token_to_new_id[token_str] = next_id
            next_id += 1

        return ReindexingResult(
            tokenizer_name="",
            strategy="pure_frequency",
            vocab_size=len(token_to_new_id),
            old_id_to_new_id=old_id_to_new_id,
            new_id_to_token=new_id_to_token,
            token_to_new_id=token_to_new_id,
            id_ranges={"special": {"start": 0, "end": 255}},
            special_tokens_count=len(special_mapping),
            head_tokens_count=0,
            torso_tokens_count=0,
            tail_tokens_count=0
        )

    def strategy_category_blocks(
        self,
        token_to_id: Dict[str, int],
        frequencies: Dict[int, int]
    ) -> ReindexingResult:
        """Category blocks strategy with frequency sorting within each block."""
        logger.info("Using category blocks strategy")

        # Separate special tokens
        special_mapping, regular_vocab = self.special_token_manager.allocate_special_tokens(token_to_id)

        # Get category blocks configuration
        blocks = self.reindexing_config['category_blocks']

        # Apply MoE smoothing to frequencies
        smoothed_frequencies = self.apply_moe_smoothing(frequencies)

        # Get token-frequency pairs for regular vocab
        id_to_token = {v: k for k, v in regular_vocab.items()}
        token_freqs = [
            (old_id, id_to_token[old_id], smoothed_frequencies.get(old_id, 0))
            for old_id in id_to_token.keys()
        ]

        # Sort by smoothed frequency
        token_freqs.sort(key=lambda x: x[2], reverse=True)

        # Determine block sizes
        total_regular_tokens = len(token_freqs)
        special_block = blocks['special_tokens']
        high_block = blocks['high_frequency']
        medium_block = blocks['medium_frequency']
        low_block = blocks['low_frequency']

        # Calculate actual block sizes based on available tokens
        high_size = min(high_block['end_id'] - high_block['start_id'], total_regular_tokens)
        medium_size = min(medium_block['end_id'] - medium_block['start_id'], total_regular_tokens - high_size)
        low_size = min(low_block['end_id'] - low_block['start_id'], total_regular_tokens - high_size - medium_size)

        # Split tokens into blocks
        high_tokens = token_freqs[:high_size]
        medium_tokens = token_freqs[high_size:high_size + medium_size]
        low_tokens = token_freqs[high_size + medium_size:high_size + medium_size + low_size]

        logger.info(f"Block allocation: high={len(high_tokens)}, medium={len(medium_tokens)}, low={len(low_tokens)}")

        # Assign new IDs
        old_id_to_new_id = {}
        new_id_to_token = {}
        token_to_new_id = {}

        # Assign special tokens
        for token_str, new_id in special_mapping.items():
            new_id_to_token[new_id] = token_str
            token_to_new_id[token_str] = new_id

        # Assign high frequency tokens
        next_id = high_block['start_id']
        for old_id, token_str, freq in high_tokens:
            old_id_to_new_id[old_id] = next_id
            new_id_to_token[next_id] = token_str
            token_to_new_id[token_str] = next_id
            next_id += 1

        # Assign medium frequency tokens
        next_id = medium_block['start_id']
        for old_id, token_str, freq in medium_tokens:
            old_id_to_new_id[old_id] = next_id
            new_id_to_token[next_id] = token_str
            token_to_new_id[token_str] = next_id
            next_id += 1

        # Assign low frequency tokens
        next_id = low_block['start_id']
        for old_id, token_str, freq in low_tokens:
            old_id_to_new_id[old_id] = next_id
            new_id_to_token[next_id] = token_str
            token_to_new_id[token_str] = next_id
            next_id += 1

        return ReindexingResult(
            tokenizer_name="",
            strategy="category_blocks",
            vocab_size=len(token_to_new_id),
            old_id_to_new_id=old_id_to_new_id,
            new_id_to_token=new_id_to_token,
            token_to_new_id=token_to_new_id,
            id_ranges={
                "special": {"start": special_block['start_id'], "end": special_block['end_id']},
                "high_frequency": {"start": high_block['start_id'], "end": high_block['start_id'] + len(high_tokens)},
                "medium_frequency": {"start": medium_block['start_id'], "end": medium_block['start_id'] + len(medium_tokens)},
                "low_frequency": {"start": low_block['start_id'], "end": low_block['start_id'] + len(low_tokens)}
            },
            special_tokens_count=len(special_mapping),
            head_tokens_count=len(high_tokens),
            torso_tokens_count=len(medium_tokens),
            tail_tokens_count=len(low_tokens)
        )

    def reindex(
        self,
        tokenizer_name: str,
        frequency_stats_path: str
    ) -> ReindexingResult:
        """Main reindexing function."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Reindexing tokenizer: {tokenizer_name}")
        logger.info(f"{'='*60}")

        # Load tokenizer and frequency stats
        token_to_id = self.load_tokenizer(tokenizer_name)
        frequencies = self.load_frequency_stats(frequency_stats_path)

        # Select strategy
        strategy = self.reindexing_config['strategy']

        if strategy == "pure_frequency":
            result = self.strategy_pure_frequency(token_to_id, frequencies)
        elif strategy == "category_blocks":
            result = self.strategy_category_blocks(token_to_id, frequencies)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        result.tokenizer_name = tokenizer_name

        logger.info(f"\nReindexing complete:")
        logger.info(f"  Strategy: {result.strategy}")
        logger.info(f"  Vocab size: {result.vocab_size}")
        logger.info(f"  Special tokens: {result.special_tokens_count}")
        logger.info(f"  Head tokens: {result.head_tokens_count}")
        logger.info(f"  Torso tokens: {result.torso_tokens_count}")
        logger.info(f"  Tail tokens: {result.tail_tokens_count}")

        return result

    def save_reindexed_tokenizer(
        self,
        result: ReindexingResult,
        output_dir: str
    ):
        """Save reindexed tokenizer files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save token -> new ID mapping
        tokenizer_file = output_path / "tokenizer_reindexed.json"
        with open(tokenizer_file, 'w', encoding='utf-8') as f:
            json.dump(result.token_to_new_id, f, indent=2, ensure_ascii=False)

        # Save old ID -> new ID mapping
        id_mapping_file = output_path / "id_mapping.json"
        with open(id_mapping_file, 'w') as f:
            # Convert int keys to strings for JSON
            id_mapping_str = {str(k): v for k, v in result.old_id_to_new_id.items()}
            json.dump(id_mapping_str, f, indent=2)

        # Save new ID -> token mapping
        inverse_mapping_file = output_path / "id_to_token.json"
        with open(inverse_mapping_file, 'w', encoding='utf-8') as f:
            # Convert int keys to strings for JSON
            inverse_mapping_str = {str(k): v for k, v in result.new_id_to_token.items()}
            json.dump(inverse_mapping_str, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            "tokenizer_name": result.tokenizer_name,
            "strategy": result.strategy,
            "vocab_size": result.vocab_size,
            "id_ranges": result.id_ranges,
            "special_tokens_count": result.special_tokens_count,
            "head_tokens_count": result.head_tokens_count,
            "torso_tokens_count": result.torso_tokens_count,
            "tail_tokens_count": result.tail_tokens_count
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\nReindexed tokenizer saved to: {output_path}")
        logger.info(f"  - tokenizer_reindexed.json")
        logger.info(f"  - id_mapping.json")
        logger.info(f"  - id_to_token.json")
        logger.info(f"  - metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Token ID Reindexer")
    parser.add_argument('--tokenizer', type=str, required=True,
                        help='Tokenizer name (e.g., ds_filtered)')
    parser.add_argument('--frequency-stats', type=str, required=True,
                        help='Path to frequency statistics JSON')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for reindexed tokenizer')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create reindexer
    reindexer = FrequencyBasedReindexer(config)

    # Reindex
    result = reindexer.reindex(
        tokenizer_name=args.tokenizer,
        frequency_stats_path=args.frequency_stats
    )

    # Save results
    reindexer.save_reindexed_tokenizer(result, args.output)


if __name__ == "__main__":
    main()
