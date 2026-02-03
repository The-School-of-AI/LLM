#!/usr/bin/env python3
"""
Tokenizer Reindexing - Complete Token Selection & ID Assignment

This script handles BOTH token selection and ID assignment using RRF algorithm.
It can work directly from filtered tokenizers or from a pre-merged vocabulary.

Key Features:
- RRF-based token SELECTION (picks best tokens from large vocab) - ALWAYS ENABLED
- RRF-based ID ASSIGNMENT (orders by coverage + quality + category) - ALWAYS ENABLED
- Strict token length enforcement (max 32 chars by default)
- Category-aware allocation (100k English, 10k Indic, 9k code)
- Special token preservation (IDs 0-99)
- Reserved ID ranges at END (IDs 127800-127999 for future governance)
- Percentile-based ID bands for MoE routing
- Validation and safety checks
- Documentation generation for downstream teams
- NOTE: Frequency analysis done separately by generate_tokenizer_report.py

Usage:
    # Complete pipeline: from filtered tokenizers to final vocab (RRF always used)
    python tokenizer_reindexing.py \\
        --input-dir filtered_tokenizer \\
        --output merged_tokens/reindexed_tokenizer_128k.json \\
        --target-size 128000 \\
        --reserved-count 200
    
    # With long tokens CSV for reference
    python tokenizer_reindexing.py \\
        --input-dir filtered_tokenizer \\
        --output merged_tokens/reindexed_tokenizer_128k.json \\
        --long-tokens-csv tokenizer_results/long_tokens_32plus.csv \\
        --target-size 128000
    
    # From pre-merged vocab (if you already ran merge)
    python tokenizer_reindexing.py \\
        --input merged_tokens/merged_tokenizer_128k.json \\
        --output reindexed_tokenizer_128k.json
    
    # With custom special tokens
    python tokenizer_reindexing.py \\
        --input merged_tokens/merged_tokenizer_128k.json \\
        --output reindexed_tokenizer_128k.json \\
        --dataset-dir ../datasets \\
        --special-tokens-config special_tokens_config.json \\
        --reserved-ids-start 100 \\
        --reserved-ids-end 299
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Progress bar (optional)
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(x, **kwargs):
        return x


# Import tokenizer utilities
try:
    from utils import VocabularyWrapper, categorize_token, check_language

    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    if __name__ == "__main__":
        print("⚠️  Warning: utils.py not found in current directory")
        print("   Make sure to run from tokenizer_filter/ directory")


class SpecialTokensManager:
    """Manages special tokens with fixed IDs."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize special tokens from config or defaults."""
        self.special_tokens = {}
        self.reserved_range = (100, 299)  # Default reserved range
        self.token_descriptions = {}

        if config_path and config_path.exists():
            self._load_config(config_path)
        else:
            self._load_default_special_tokens()

    def _load_config(self, config_path: Path):
        """Load special tokens from configuration file."""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        scheme = config.get("special_tokens_scheme", {})

        # Load base tokens
        base_tokens = scheme.get("base_tokens", {})
        if "tokens" in base_tokens:
            for token_def in base_tokens["tokens"]:
                token = token_def["token"]
                token_id = token_def["id"]
                desc = token_def.get("description", "")
                self.special_tokens[token] = token_id
                self.token_descriptions[token] = desc

        # Load extended tokens
        for category in [
            "instruction_tokens",
            "document_tokens",
            "code_tokens",
            "json_tool_tokens",
            "metadata_tokens",
        ]:
            if category in scheme:
                tokens = scheme[category].get("tokens", [])
                for token_def in tokens:
                    token = token_def["token"]
                    token_id = token_def["id"]
                    desc = token_def.get("description", "")
                    self.special_tokens[token] = token_id
                    self.token_descriptions[token] = desc

        # Load reserved range
        if "future_governance_reserved" in scheme:
            id_range = scheme["future_governance_reserved"].get("id_range", [100, 299])
            self.reserved_range = tuple(id_range)

    def _load_default_special_tokens(self):
        """Load minimal default special tokens (base tokens at start)."""
        # Core control tokens (0-9)
        self.special_tokens.update(
            {
                "</s>": 0,
                "<pad>": 1,
                "<s>": 2,
                "<unk>": 3,
                "<|endoftext|>": 4,
                "<|im_end|>": 5,
                "<|im_start|>": 6,
                "[/INST]": 7,
                "[INST]": 8,
                "[TOOL_CALLS]": 9,
            }
        )

        # Extended special tokens (10-99)
        extended_tokens = {
            # Instruction tokens (10-19)
            "<|system|>": 10,
            "<|user|>": 11,
            "<|assistant|>": 12,
            "[/TOOL_CALLS]": 13,
            "<|function|>": 14,
            # Document structure (20-29)
            "<|doc_start|>": 20,
            "<|doc_end|>": 21,
            "<|chunk_start|>": 22,
            "<|chunk_end|>": 23,
            "<|section|>": 24,
            # Code blocks (30-39)
            "<|code_start|>": 30,
            "<|code_end|>": 31,
            "<|python|>": 32,
            "<|javascript|>": 33,
            "<|typescript|>": 34,
            # JSON / Tool calling (40-49)
            "<|json_start|>": 40,
            "<|json_end|>": 41,
            "<|tool_call|>": 42,
            "<|tool_response|>": 43,
            "<|parameters|>": 44,
        }

        self.special_tokens.update(extended_tokens)

        # Reserved range at the END of vocabulary (default: last 200 IDs)
        self.reserved_range = (127800, 127999)  # For 128k vocab

        # Add descriptions
        self.token_descriptions = {
            "</s>": "End of sequence",
            "<pad>": "Padding token",
            "<s>": "Start of sequence",
            "<unk>": "Unknown token",
            "<|endoftext|>": "End of text (GPT-style)",
            "<|im_end|>": "Instruction mode end",
            "<|im_start|>": "Instruction mode start",
            "[/INST]": "End instruction (Llama-style)",
            "[INST]": "Start instruction (Llama-style)",
            "[TOOL_CALLS]": "Tool calls start",
            "<|system|>": "System role",
            "<|user|>": "User role",
            "<|assistant|>": "Assistant role",
            "[/TOOL_CALLS]": "Tool calls end",
            "<|function|>": "Function call",
            "<|doc_start|>": "Document start",
            "<|doc_end|>": "Document end",
            "<|chunk_start|>": "Chunk start",
            "<|chunk_end|>": "Chunk end",
            "<|section|>": "Section marker",
            "<|code_start|>": "Code block start",
            "<|code_end|>": "Code block end",
            "<|python|>": "Python code",
            "<|javascript|>": "JavaScript code",
            "<|typescript|>": "TypeScript code",
            "<|json_start|>": "JSON start",
            "<|json_end|>": "JSON end",
            "<|tool_call|>": "Tool call",
            "<|tool_response|>": "Tool response",
            "<|parameters|>": "Parameters",
        }

    def get_special_tokens(self) -> Dict[str, int]:
        """Return all special tokens with their IDs."""
        return self.special_tokens.copy()

    def get_reserved_range(self) -> Tuple[int, int]:
        """Return reserved ID range for future governance."""
        return self.reserved_range

    def get_regular_vocab_start(self) -> int:
        """Get starting ID for regular vocabulary (right after special tokens)."""
        max_special = max(self.special_tokens.values()) if self.special_tokens else 0
        # Regular vocab starts right after special tokens
        # Reserved IDs are at the END, so they don't affect this
        return max_special + 1

    def is_special_token(self, token: str) -> bool:
        """Check if token is a special token."""
        return token in self.special_tokens

    def get_token_description(self, token: str) -> str:
        """Get description for a special token."""
        return self.token_descriptions.get(token, "")


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining multiple rankings."""

    def __init__(self, k: int = 60):
        """
        Initialize RRF with a constant k.

        Args:
            k: Parameter that controls impact of outlier rankings (default: 60)
        """
        self.k = k

    def fuse(self, ranked_lists: List[List[str]]) -> List[Tuple[str, float]]:
        """
        Implements Reciprocal Rank Fusion (RRF).

        Args:
            ranked_lists: List of lists, where each inner list contains tokens in ranked order.

        Returns:
            List of (token, score) tuples sorted by descending RRF score.
        """
        fused_scores = defaultdict(float)

        for rank_list in ranked_lists:
            for rank, token in enumerate(rank_list):
                # Rank is 0-indexed, so we add 1 for proper RRF calculation
                fused_scores[token] += 1.0 / (self.k + (rank + 1))

        # Sort tokens based on their scores in descending order
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


class TokenSelector:
    """Select tokens using RRF when vocabulary exceeds target size."""

    def __init__(self, verbose: bool = True):
        """Initialize token selector."""
        self.verbose = verbose
        self.category_allocations = {
            "english": 107970,  # Fill to exactly 128k (127,970 regular + 30 special)
            "indic": 10000,
            "code": 10000,  # Increased from 9k
            "numeric": 5000,  # Increased from 4k
            "symbols": 3500,  # Increased from 3k
            "json_structural": 200,
            "special": 200,
            "other": 100,
            "whitespace": 0,  # Total: ~127,970 regular tokens
        }

    def select_tokens_with_rrf(
        self, all_tokens: Dict[str, Dict], target_size: int, rrf_k: int = 60
    ) -> Dict[str, Dict]:
        """
        Select best tokens using RRF when vocabulary exceeds target size.

        Args:
            all_tokens: Dict of token -> metadata (source, category, original_id, etc.)
            target_size: Target vocabulary size (e.g., 128000)
            rrf_k: RRF constant

        Returns:
            Selected tokens with metadata
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TOKEN SELECTION PHASE (RRF)")
            print("=" * 80)
            print(f"  Input tokens: {len(all_tokens):,}")
            print(f"  Target size: {target_size:,}")

        # Group by category
        by_category = defaultdict(list)
        for token, metadata in all_tokens.items():
            category = metadata.get("category", "other")
            by_category[category].append((token, metadata))

        if self.verbose:
            print("\n  📊 Tokens by category:")
            for cat in sorted(by_category.keys()):
                print(f"     {cat:20s}: {len(by_category[cat]):7,} tokens")

        # Select tokens per category using RRF
        selected_tokens = {}
        rrf = ReciprocalRankFusion(k=rrf_k)

        for category, tokens_list in by_category.items():
            max_alloc = self.category_allocations.get(category, 100)

            if len(tokens_list) <= max_alloc:
                # All tokens fit, no selection needed
                for token, metadata in tokens_list:
                    selected_tokens[token] = metadata
                if self.verbose:
                    print(
                        f"\n  ✓ {category}: All {len(tokens_list):,} tokens included (under limit)"
                    )
            else:
                # Need to select best tokens using RRF
                if self.verbose:
                    print(
                        f"\n  🔀 {category}: Selecting {max_alloc:,} from {len(tokens_list):,} using RRF..."
                    )

                selected = self._select_category_tokens_rrf(
                    tokens_list, max_alloc, rrf, rrf_k
                )

                for token, metadata in selected:
                    selected_tokens[token] = metadata

        if self.verbose:
            print(f"\n  ✅ Total tokens selected: {len(selected_tokens):,}")

        return selected_tokens

    def _select_category_tokens_rrf(
        self,
        tokens_list: List[Tuple],
        max_alloc: int,
        rrf: "ReciprocalRankFusion",
        rrf_k: int,
    ) -> List[Tuple]:
        """Select best tokens for a category using RRF."""

        # Create metadata lookup for faster access
        token_to_metadata = {token: metadata for token, metadata in tokens_list}
        token_strings = list(token_to_metadata.keys())

        # Build ranking signals

        # Signal 1: Coverage (how many tokenizers have this token)
        coverage_ranked = sorted(
            token_strings, key=lambda t: -token_to_metadata[t].get("source_count", 1)
        )

        # Signal 2: Source quality (which tokenizer it came from)
        tokenizer_priority = {
            "gptoss": 1,
            "mistral": 2,
            "byted": 3,
            "ds": 4,
            "olmo": 5,
            "gemma": 6,
            "qwen": 7,
            "qwencode": 7,
            "olmocode": 7,
            "dscode": 8,
        }

        quality_ranked = sorted(
            token_strings,
            key=lambda t: tokenizer_priority.get(
                token_to_metadata[t].get("source", "unknown"), 99
            ),
        )

        # Signal 3: Original ID (lower ID in source = more important)
        id_ranked = sorted(
            token_strings, key=lambda t: token_to_metadata[t].get("original_id", 999999)
        )

        # Signal 4: Token length (shorter often more useful)
        length_ranked = sorted(token_strings, key=lambda t: len(t))

        # Apply RRF
        ranked_lists = [coverage_ranked, quality_ranked, id_ranked, length_ranked]
        rrf_results = rrf.fuse(ranked_lists)

        if self.verbose:
            print("     Top 5 by RRF:")
            for i, (token, score) in enumerate(rrf_results[:5], 1):
                meta = token_to_metadata[token]
                print(
                    f"       {i}. {token!r} (score={score:.4f}, source={meta.get('source')}, "
                    f"coverage={meta.get('source_count')}/10)"
                )

        # Take top max_alloc tokens
        selected_tokens = [token for token, score in rrf_results[:max_alloc]]

        # Return with metadata
        result = [(token, token_to_metadata[token]) for token in selected_tokens]

        return result


class FrequencyBasedReindexer:
    """Reindex tokenizer based on frequency analysis."""

    def __init__(self, special_tokens_mgr: SpecialTokensManager, verbose: bool = True):
        """
        Initialize reindexer.

        Args:
            special_tokens_mgr: Special tokens manager
            verbose: Enable verbose output
        """
        self.special_tokens_mgr = special_tokens_mgr
        self.verbose = verbose
        self.percentile_bands = []
        self.id_range_documentation = {}

        # Tokenizer priority for RRF ranking
        self.tokenizer_priority = {
            "gptoss": 1,
            "mistral": 2,
            "byted": 3,
            "ds": 4,
            "olmo": 5,
            "gemma": 6,
            "qwen": 7,
            "qwencode": 7,
            "olmocode": 7,
            "dscode": 8,
        }

    def reindex(
        self,
        vocab: Dict[str, int],
        frequencies: Dict[str, int],
        token_metadata: Optional[Dict[str, Dict]] = None,
        smoothing_factor: float = 0.1,
        target_size: int = 128000,
        use_rrf: bool = False,
        rrf_k: int = 60,
    ) -> Dict[str, int]:
        """
        Reindex vocabulary based on frequency or RRF.

        Args:
            vocab: Original vocabulary (token -> old_id)
            frequencies: Token frequencies (token -> count)
            token_metadata: Optional metadata about tokens (source, category, etc.)
            smoothing_factor: Smoothing for frequency ordering (0.0 = exact, 1.0 = random)
            target_size: Target vocabulary size (default: 128000)
            use_rrf: Use Reciprocal Rank Fusion instead of simple frequency (default: False)
            rrf_k: RRF constant k (default: 60)

        Returns:
            New vocabulary (token -> new_id)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("FREQUENCY-BASED REINDEXING")
            print("=" * 80)

        # Step 1: Preserve special tokens at the beginning
        special_tokens = self.special_tokens_mgr.get_special_tokens()
        reindexed_vocab = special_tokens.copy()

        if self.verbose:
            max_special_id = max(special_tokens.values())
            print(
                f"\n  🔐 Special tokens: {len(special_tokens)} preserved at IDs 0-{max_special_id}"
            )

        # Step 2: Calculate reserved IDs at the END
        reserved_start, reserved_end = self.special_tokens_mgr.get_reserved_range()
        reserved_count = reserved_end - reserved_start + 1

        # Adjust reserved range if it extends beyond target size
        if reserved_end >= target_size:
            reserved_end = target_size - 1
            reserved_start = reserved_end - reserved_count + 1
            if self.verbose:
                print("  ⚠️  Adjusted reserved range to fit target size")

        if self.verbose:
            print(
                f"  🔒 Reserved IDs: {reserved_count} IDs ({reserved_start}-{reserved_end}) at END (future governance)"
            )

        # Step 3: Prepare regular vocabulary (between special tokens and reserved IDs)
        regular_vocab_start = self.special_tokens_mgr.get_regular_vocab_start()
        regular_vocab_end = reserved_start - 1  # End before reserved IDs
        max_regular_tokens = regular_vocab_end - regular_vocab_start + 1

        if self.verbose:
            print("\n  📚 Regular vocabulary allocation:")
            print(f"     Start ID: {regular_vocab_start}")
            print(f"     End ID:   {regular_vocab_end}")
            print(f"     Capacity: {max_regular_tokens:,} tokens")

        # Get tokens excluding special tokens
        regular_tokens = [
            token for token in vocab.keys() if token not in special_tokens
        ]

        if self.verbose:
            print(f"  📊 Regular tokens to reindex: {len(regular_tokens):,}")

        # Check if we have enough space
        if len(regular_tokens) > max_regular_tokens:
            if self.verbose:
                print(
                    f"  ⚠️  WARNING: {len(regular_tokens):,} tokens exceed capacity of {max_regular_tokens:,}"
                )
                print(
                    "     Will truncate to fit. Consider reducing reserved IDs or increasing target size."
                )

        # Step 4: Rank tokens using either simple frequency or RRF
        if use_rrf and token_metadata:
            token_freq_pairs = self._rank_with_rrf(
                regular_tokens, frequencies, token_metadata, rrf_k
            )
            if self.verbose:
                print("\n  🔄 Ranking using Reciprocal Rank Fusion (RRF)")
                print(f"     RRF k={rrf_k}, combining multiple ranking signals")
                print("     High RRF score → Low IDs (better for MoE routing)")
        else:
            # Simple frequency-based ranking with smoothing
            token_freq_pairs = []
            for token in regular_tokens:
                freq = frequencies.get(token, 0)

                # Apply smoothing: add random noise proportional to smoothing factor
                if smoothing_factor > 0:
                    noise = random.gauss(
                        0, smoothing_factor * math.log(max(freq, 1) + 1)
                    )
                    smoothed_freq = freq + noise
                else:
                    smoothed_freq = freq

                token_freq_pairs.append((token, freq, smoothed_freq))

            # Sort by smoothed frequency (descending) then by token (ascending)
            token_freq_pairs.sort(key=lambda x: (-x[2], x[0]))

            if self.verbose:
                print(
                    f"\n  🔄 Sorting by frequency (smoothing factor: {smoothing_factor})"
                )
                print("     High frequency → Low IDs (better for MoE routing)")

        # Step 5: Assign new IDs (only up to regular_vocab_end)
        current_id = regular_vocab_start
        tokens_assigned = 0

        for token, original_freq, smoothed_freq in token_freq_pairs:
            if current_id > regular_vocab_end:
                if self.verbose:
                    print(
                        f"  ⚠️  Reached end of regular vocab space at ID {regular_vocab_end}"
                    )
                break

            reindexed_vocab[token] = current_id
            current_id += 1
            tokens_assigned += 1

        if self.verbose:
            print("\n  ✓ Reindexing complete")
            print(f"     Regular tokens assigned: {tokens_assigned:,}")
            print(f"     Special tokens: {len(special_tokens)}")
            print(f"     Reserved IDs: {reserved_count} (at end)")
            print(f"     Total vocabulary size: {len(reindexed_vocab):,}")
            print(
                f"     Regular vocab ID range: {regular_vocab_start} - {current_id - 1}"
            )
            print(f"     Reserved ID range: {reserved_start} - {reserved_end}")

        # Step 6: Generate percentile bands
        self._generate_percentile_bands(
            token_freq_pairs[:tokens_assigned], regular_vocab_start, regular_vocab_end
        )

        # Step 7: Generate ID range documentation
        self._generate_id_documentation(
            reindexed_vocab,
            frequencies,
            regular_vocab_start,
            regular_vocab_end,
            reserved_start,
            reserved_end,
        )

        return reindexed_vocab

    def _rank_with_rrf(
        self,
        tokens: List[str],
        frequencies: Dict[str, int],
        token_metadata: Dict[str, Dict],
        rrf_k: int = 60,
    ) -> List[Tuple]:
        """
        Rank tokens using Reciprocal Rank Fusion (RRF).

        Combines multiple ranking signals:
        1. Frequency rank (from dataset analysis)
        2. Source quality rank (tokenizer priority)
        3. Source count rank (tokens in more tokenizers = better)
        4. Category rank (code/indic prioritized)

        Args:
            tokens: List of tokens to rank
            frequencies: Token frequencies
            token_metadata: Metadata with 'source', 'category', 'source_count'
            rrf_k: RRF constant

        Returns:
            List of (token, original_freq, rrf_score) tuples sorted by RRF score
        """
        if self.verbose:
            print("\n  🔀 Generating multiple ranking signals for RRF...")

        # Signal 1: Frequency rank (high freq = low rank number = better)
        freq_sorted = sorted(tokens, key=lambda t: -frequencies.get(t, 0))

        # Signal 2: Source quality rank (gptoss=best, etc.)
        def get_source_priority(token):
            if token in token_metadata:
                source = token_metadata[token].get("source", "unknown")
                return self.tokenizer_priority.get(source, 99)
            return 99

        source_sorted = sorted(tokens, key=lambda t: (get_source_priority(t), t))

        # Signal 3: Source count rank (more tokenizers = better)
        def get_source_count(token):
            if token in token_metadata:
                return token_metadata[token].get("source_count", 1)
            return 1

        source_count_sorted = sorted(tokens, key=lambda t: -get_source_count(t))

        # Signal 4: Category rank (prioritize code, indic)
        category_priority = {
            "code": 1,
            "indic": 2,
            "english": 3,
            "numeric": 4,
            "symbols": 5,
            "special": 6,
            "other": 7,
        }

        def get_category_priority(token):
            if token in token_metadata:
                category = token_metadata[token].get("category", "other")
                return category_priority.get(category, 99)
            return 99

        category_sorted = sorted(tokens, key=lambda t: (get_category_priority(t), t))

        if self.verbose:
            print("     ✓ Generated 4 ranking signals:")
            print("       1. Frequency rank")
            print("       2. Source quality rank (tokenizer priority)")
            print("       3. Source count rank (token coverage)")
            print("       4. Category rank (code/indic prioritized)")

        # Apply RRF to combine rankings
        rrf = ReciprocalRankFusion(k=rrf_k)
        ranked_lists = [
            freq_sorted,
            source_sorted,
            source_count_sorted,
            category_sorted,
        ]

        rrf_results = rrf.fuse(ranked_lists)

        if self.verbose:
            print(f"     ✓ Fused rankings using RRF (k={rrf_k})")
            print("       Top 5 tokens by RRF score:")
            for i, (token, score) in enumerate(rrf_results[:5], 1):
                freq = frequencies.get(token, 0)
                print(f"         {i}. {token!r} (freq={freq:,}, RRF={score:.4f})")

        # Convert to format expected by rest of code
        # (token, original_freq, rrf_score)
        token_freq_pairs = []
        for token, rrf_score in rrf_results:
            original_freq = frequencies.get(token, 0)
            token_freq_pairs.append((token, original_freq, rrf_score))

        return token_freq_pairs

    def _generate_percentile_bands(
        self, token_freq_pairs: List[Tuple], start_id: int, end_id: int
    ):
        """Generate percentile-based ID bands for MoE routing."""
        if self.verbose:
            print("\n  📊 Generating percentile bands for MoE routing...")

        total_tokens = len(token_freq_pairs)

        # Define percentile bands
        percentiles = [
            (0, 1, "Top 1% (ultra-frequent)"),
            (1, 5, "Top 5% (very frequent)"),
            (5, 10, "Top 10% (frequent)"),
            (10, 25, "Top 25% (common)"),
            (25, 50, "Top 50% (moderate)"),
            (50, 75, "Top 75% (less common)"),
            (75, 90, "Top 90% (rare)"),
            (90, 95, "Top 95% (very rare)"),
            (95, 99, "Top 99% (ultra-rare)"),
            (99, 100, "Bottom 1% (tail)"),
        ]

        bands = []
        for p_start, p_end, description in percentiles:
            # Calculate token indices for this percentile
            idx_start = int(total_tokens * p_start / 100)
            idx_end = int(total_tokens * p_end / 100)

            if idx_start >= idx_end:
                continue

            # Get ID range
            id_start = start_id + idx_start
            id_end = start_id + idx_end - 1

            # Get frequency range
            if idx_end < len(token_freq_pairs):
                freq_high = token_freq_pairs[idx_start][1]  # original freq
                freq_low = token_freq_pairs[idx_end - 1][1]
            else:
                freq_high = token_freq_pairs[idx_start][1]
                freq_low = token_freq_pairs[-1][1]

            band = {
                "percentile_range": (p_start, p_end),
                "description": description,
                "id_range": (id_start, id_end),
                "token_count": idx_end - idx_start,
                "frequency_range": (freq_low, freq_high),
            }

            bands.append(band)

            if self.verbose:
                print(
                    f"     {description:25s} IDs {id_start:6,}-{id_end:6,} "
                    f"({band['token_count']:6,} tokens, freq: {freq_low:8,}-{freq_high:8,})"
                )

        self.percentile_bands = bands

    def _generate_id_documentation(
        self,
        reindexed_vocab: Dict[str, int],
        frequencies: Dict[str, int],
        regular_start: int,
        regular_end: int,
        reserved_start: int,
        reserved_end: int,
    ):
        """Generate ID range documentation for downstream teams."""
        special_tokens = self.special_tokens_mgr.get_special_tokens()

        # Special tokens range
        special_ids = sorted(special_tokens.values())

        # Get actual max ID used for regular vocab
        regular_vocab_ids = [
            v for k, v in reindexed_vocab.items() if k not in special_tokens
        ]
        max_regular_id = max(regular_vocab_ids) if regular_vocab_ids else regular_start

        self.id_range_documentation = {
            "special_tokens": {
                "id_range": (min(special_ids), max(special_ids)),
                "count": len(special_tokens),
                "description": "Control tokens for formatting and instructions",
                "examples": list(special_tokens.keys())[:5],
            },
            "regular_vocabulary": {
                "id_range": (regular_start, max_regular_id),
                "count": len(reindexed_vocab) - len(special_tokens),
                "description": "Regular vocabulary tokens ordered by frequency",
                "ordering": "High frequency = Low ID (MoE-safe)",
                "note": "Frequency-ordered for optimal MoE routing",
            },
            "reserved_ids": {
                "id_range": (reserved_start, reserved_end),
                "count": reserved_end - reserved_start + 1,
                "description": "Reserved at END for future governance and special tokens",
                "location": "End of vocabulary (does not disrupt regular tokens)",
                "note": "Do not use these IDs in current models",
            },
            "percentile_bands": self.percentile_bands,
        }

    def get_percentile_bands(self) -> List[Dict]:
        """Get percentile bands for MoE routing."""
        return self.percentile_bands

    def get_id_documentation(self) -> Dict:
        """Get ID range documentation."""
        return self.id_range_documentation


class VocabularyValidator:
    """Validates vocabulary meets all requirements."""

    def __init__(self, verbose: bool = True):
        """Initialize validator."""
        self.verbose = verbose
        self.validation_errors = []
        self.validation_warnings = []

    def validate_vocabulary(
        self, vocab: Dict[str, int], max_token_length: int = 32
    ) -> bool:
        """
        Validate vocabulary meets requirements.

        Args:
            vocab: Vocabulary to validate
            max_token_length: Maximum allowed token length

        Returns:
            True if validation passes
        """
        if self.verbose:
            print("\n🔍 Validating vocabulary...")

        # Check 1: No tokens exceed max length
        long_tokens = []
        for token, token_id in vocab.items():
            if len(token) > max_token_length:
                long_tokens.append((token, len(token)))

        if long_tokens:
            self.validation_errors.append(
                f"Found {len(long_tokens)} tokens exceeding max length {max_token_length}"
            )
            if self.verbose:
                print(
                    f"  ❌ FAIL: {len(long_tokens)} tokens > {max_token_length} chars"
                )
                print(f"     Examples: {long_tokens[:5]}")
                print(
                    "     ⚠️  These tokens should be handled by the merger/filter first!"
                )
        else:
            if self.verbose:
                print(f"  ✓ PASS: All tokens ≤ {max_token_length} chars")

        # Check 2: No duplicate IDs
        id_counts = defaultdict(int)
        for token, token_id in vocab.items():
            id_counts[token_id] += 1

        duplicate_ids = {tid: count for tid, count in id_counts.items() if count > 1}
        if duplicate_ids:
            self.validation_errors.append(
                f"Found {len(duplicate_ids)} duplicate token IDs"
            )
            if self.verbose:
                print(f"  ❌ FAIL: {len(duplicate_ids)} duplicate IDs")
        else:
            if self.verbose:
                print("  ✓ PASS: All token IDs unique")

        # Check 3: Vocab size is reasonable
        vocab_size = len(vocab)
        if vocab_size < 1000:
            self.validation_warnings.append(
                f"Vocabulary very small: {vocab_size} tokens"
            )
            if self.verbose:
                print(f"  ⚠️  WARNING: Small vocabulary ({vocab_size} tokens)")
        elif vocab_size > 200000:
            self.validation_warnings.append(
                f"Vocabulary very large: {vocab_size} tokens"
            )
            if self.verbose:
                print(f"  ⚠️  WARNING: Large vocabulary ({vocab_size} tokens)")
        else:
            if self.verbose:
                print(f"  ✓ PASS: Reasonable vocabulary size ({vocab_size:,} tokens)")

        # Check 4: Has some common tokens
        common_tokens = ["the", "a", "is", " ", "\n", ".", ","]
        found_common = sum(1 for t in common_tokens if t in vocab)
        if found_common < 3:
            self.validation_warnings.append(
                f"Few common tokens found ({found_common}/{len(common_tokens)})"
            )
            if self.verbose:
                print(
                    f"  ⚠️  WARNING: Only {found_common}/{len(common_tokens)} common tokens found"
                )
        else:
            if self.verbose:
                print(
                    f"  ✓ PASS: Found {found_common}/{len(common_tokens)} common tokens"
                )

        # Summary
        has_errors = len(self.validation_errors) > 0

        if self.verbose:
            if has_errors:
                print(
                    f"\n  ❌ Validation FAILED with {len(self.validation_errors)} errors"
                )
                for error in self.validation_errors:
                    print(f"     - {error}")
            else:
                print("\n  ✅ Validation PASSED")

            if self.validation_warnings:
                print(f"  ⚠️  {len(self.validation_warnings)} warnings:")
                for warning in self.validation_warnings:
                    print(f"     - {warning}")

        return not has_errors

    def get_validation_report(self) -> Dict:
        """Get validation report."""
        return {
            "passed": len(self.validation_errors) == 0,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings,
        }


class TokenizerReindexer:
    """Main tokenizer reindexing pipeline with token selection."""

    def __init__(
        self,
        special_tokens_config: Optional[Path] = None,
        verbose: bool = True,
        max_token_length: int = 32,
        config_path: str = "tokenizer_config.json",
    ):
        """
        Initialize tokenizer reindexer.

        Args:
            special_tokens_config: Path to special tokens config
            verbose: Enable verbose output
            max_token_length: Maximum token length to validate (default: 32)
            config_path: Path to tokenizer config (for category allocation)
        """
        self.verbose = verbose
        self.max_token_length = max_token_length
        self.config_path = config_path
        self.special_tokens_mgr = SpecialTokensManager(special_tokens_config)
        self.validator = VocabularyValidator(verbose=verbose)
        self.token_selector = TokenSelector(verbose=verbose)

        # Load category allocations from config
        self._load_category_allocations()

        self.original_vocabs = {}  # Can load multiple tokenizers
        self.original_vocab = {}  # Single vocabulary (for --input mode)
        self.token_metadata = {}  # Per-token metadata
        self.reindexed_vocab = {}
        self.mapping = {}  # old_id -> new_id
        self.reverse_mapping = {}  # new_id -> old_id
        self.stats = defaultdict(int)  # Statistics tracking

    def _load_category_allocations(self):
        """Load category allocations from config."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # Try to extract category priorities
            if (
                "category_priorities" in config
                and "keep" in config["category_priorities"]
            ):
                # Use defaults but can be overridden
                pass

            if self.verbose:
                print(f"✓ Loaded config: {self.config_path}")
        except Exception:
            if self.verbose:
                print("ℹ️  Using default category allocations")

    def load_directory(self, input_dir: Path, validate: bool = True) -> int:
        """
        Load multiple tokenizers from directory (for token selection).

        Args:
            input_dir: Directory with tokenizer JSON files
            validate: Whether to validate vocabularies

        Returns:
            Number of tokenizers loaded
        """
        if self.verbose:
            print(f"\n📂 Loading tokenizers from directory: {input_dir}")

        if not input_dir.exists():
            print(f"  ✗ Directory not found: {input_dir}")
            return 0

        # Find all JSON files
        json_files = sorted(input_dir.glob("*.json"))

        if not json_files:
            print("  ✗ No JSON files found")
            return 0

        loaded = 0
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract vocab
                if isinstance(data, dict):
                    if (
                        "model" in data
                        and isinstance(data["model"], dict)
                        and "vocab" in data["model"]
                    ):
                        vocab = data["model"]["vocab"]
                    elif "vocab" in data and isinstance(data["vocab"], dict):
                        vocab = data["vocab"]
                    else:
                        vocab = data

                    # Extract name
                    name = json_file.stem.replace("_filtered", "").replace(
                        "_tokenizer", ""
                    )
                    self.original_vocabs[name] = vocab
                    loaded += 1

                    if self.verbose:
                        print(f"  ✓ {name}: {len(vocab):,} tokens")

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ Error loading {json_file.name}: {e}")

        if self.verbose:
            print(f"\n  ✓ Loaded {loaded} tokenizer(s)")

        return loaded

    def load_vocabulary(self, vocab_path: Path, validate: bool = True) -> bool:
        """
        Load vocabulary from JSON file with validation.

        Args:
            vocab_path: Path to vocabulary JSON
            validate: Whether to validate vocabulary (default: True)

        Returns:
            True if loaded successfully
        """
        if self.verbose:
            print(f"\n📖 Loading vocabulary from: {vocab_path}")

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)

            # Handle different JSON formats
            if isinstance(vocab, dict):
                if "model" in vocab and "vocab" in vocab["model"]:
                    self.original_vocab = vocab["model"]["vocab"]
                elif "vocab" in vocab:
                    self.original_vocab = vocab["vocab"]
                else:
                    # Assume flat format
                    self.original_vocab = vocab

            if self.verbose:
                print(f"  ✓ Loaded {len(self.original_vocab):,} tokens")

            # Validate vocabulary
            if validate:
                validation_passed = self.validator.validate_vocabulary(
                    self.original_vocab, max_token_length=self.max_token_length
                )
                if not validation_passed:
                    print("\n⚠️  Vocabulary validation failed!")
                    print("   Consider running filter/merge pipeline first to:")
                    print(f"   1. Remove tokens > {self.max_token_length} chars")
                    print("   2. Chunk long repeating patterns")
                    print("   3. Remove duplicate IDs")
                    return False

            return True

        except Exception as e:
            print(f"  ✗ Error loading vocabulary: {e}")
            return False

    def collect_and_categorize_tokens(self) -> Dict[str, Dict]:
        """
        Collect tokens from all loaded tokenizers and categorize them.

        Returns:
            Dict of token -> metadata (source, category, original_id, source_count)
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print("TOKEN COLLECTION & CATEGORIZATION")
            print("=" * 80)

        if not HAS_UTILS:
            print("  ✗ Error: utils.py not found. Cannot categorize tokens.")
            print("    Make sure to run from tokenizer_filter/ directory")
            return {}

        # Registry to track all instances of each token
        token_registry = defaultdict(list)

        # Tokenizer priority for deduplication
        tokenizer_priority = {
            "gptoss": 1,
            "mistral": 2,
            "byted": 3,
            "ds": 4,
            "olmo": 5,
            "gemma": 6,
            "qwen": 7,
            "qwencode": 7,
            "olmocode": 7,
            "dscode": 8,
        }

        # Collect tokens from all tokenizers
        for tokenizer_name, vocab in self.original_vocabs.items():
            if self.verbose:
                print(f"\n  Processing {tokenizer_name}...")

            wrapper = VocabularyWrapper(tokenizer_name, vocab)

            iterator = tqdm(
                vocab.items(),
                desc="    Collecting",
                disable=not HAS_TQDM or not self.verbose,
            )

            for token_value, token_id in iterator:
                # Decode token
                try:
                    decoded = wrapper.decode([token_id])
                except Exception:
                    decoded = token_value

                # ✅ STRICT LENGTH CHECK: Skip tokens > max_token_length (32)
                if len(token_value) > self.max_token_length:
                    self.stats["tokens_filtered_long"] += 1
                    continue

                # Categorize
                category = categorize_token(decoded, check_language)

                # Register instance
                token_registry[token_value].append(
                    {
                        "source": tokenizer_name,
                        "original_id": token_id,
                        "category": category,
                        "decoded": decoded,
                    }
                )

        # Build metadata for each token (select best instance for deduplication)
        all_tokens_with_metadata = {}

        if self.verbose:
            print(f"\n  🔍 Deduplicating {len(token_registry):,} unique tokens...")

        for token, instances in token_registry.items():
            # Select best instance based on priority
            best = min(
                instances,
                key=lambda x: (
                    tokenizer_priority.get(x["source"], 99),
                    x["original_id"],
                ),
            )

            all_tokens_with_metadata[token] = {
                "source": best["source"],
                "original_id": best["original_id"],
                "category": best["category"],
                "decoded": best["decoded"],
                "source_count": len(instances),  # How many tokenizers had it
            }

        if self.verbose:
            print(f"\n  ✅ Collected {len(all_tokens_with_metadata):,} unique tokens")
            print(f"     From {len(self.original_vocabs)} tokenizer(s)")

            # Show filtering stats
            if self.stats["tokens_filtered_long"] > 0:
                print(
                    f"\n  🔧 Length Filtering (max_token_length={self.max_token_length}):"
                )
                print(
                    f"     Filtered out: {self.stats['tokens_filtered_long']:,} tokens (length > {self.max_token_length})"
                )

            # Show category breakdown
            by_cat = defaultdict(int)
            for meta in all_tokens_with_metadata.values():
                by_cat[meta["category"]] += 1

            print("\n  📊 Category breakdown (after length filtering):")
            for cat in sorted(by_cat.keys(), key=lambda c: -by_cat[c]):
                print(f"     {cat:20s}: {by_cat[cat]:7,} tokens")

        return all_tokens_with_metadata

    def select_and_reindex(
        self, smoothing_factor: float = 0.1, target_size: int = 128000, rrf_k: int = 60
    ) -> Dict[str, int]:
        """
        Complete pipeline: token selection + reindexing using RRF.

        Args:
            smoothing_factor: Smoothing for frequency ordering
            target_size: Target vocabulary size (default: 128000)
            rrf_k: RRF constant k (default: 60)

        Returns:
            Final reindexed vocabulary
        """
        # Step 1: Collect and categorize tokens (if from directory)
        if self.original_vocabs:
            all_tokens_metadata = self.collect_and_categorize_tokens()
            self.token_metadata = all_tokens_metadata

            # Step 2: Select best tokens using RRF (ALWAYS)
            # Calculate available space: target_size - special_tokens - reserved_count
            special_tokens_count = len(self.special_tokens_mgr.get_special_tokens())
            reserved_start, reserved_end = self.special_tokens_mgr.get_reserved_range()
            reserved_count = reserved_end - reserved_start + 1
            available_for_regular = target_size - special_tokens_count - reserved_count

            if self.verbose:
                print("\n  📊 Token allocation calculation:")
                print(f"     Target size: {target_size:,}")
                print(f"     Special tokens: {special_tokens_count}")
                print(f"     Reserved IDs: {reserved_count}")
                print(f"     Available for regular: {available_for_regular:,}")

            if len(all_tokens_metadata) > available_for_regular:
                selected_tokens = self.token_selector.select_tokens_with_rrf(
                    all_tokens_metadata, available_for_regular, rrf_k=rrf_k
                )

                # Build vocab from selected tokens
                working_vocab = {
                    token: meta["original_id"]
                    for token, meta in selected_tokens.items()
                }
                self.token_metadata = selected_tokens
            else:
                # No selection needed, use all tokens
                working_vocab = {}
                for tokens_dict in self.original_vocabs.values():
                    working_vocab.update(tokens_dict)
        else:
            # Single vocabulary file loaded (already merged/selected)
            working_vocab = self.original_vocab
            if self.verbose:
                print(
                    f"\n  ℹ️  Using pre-merged vocabulary ({len(working_vocab):,} tokens)"
                )

        # Step 3: Reindex with RRF ordering (ALWAYS, no frequency needed)
        reindexer = FrequencyBasedReindexer(
            self.special_tokens_mgr, verbose=self.verbose
        )

        self.reindexed_vocab = reindexer.reindex(
            working_vocab,
            {},  # No frequency data - RRF uses other signals
            token_metadata=(
                self.token_metadata if hasattr(self, "token_metadata") else None
            ),
            smoothing_factor=smoothing_factor,
            target_size=target_size,
            use_rrf=True,  # Always use RRF
            rrf_k=rrf_k,
        )

        # Generate ID mappings
        self._generate_mappings(working_vocab)

        # Store documentation
        self.percentile_bands = reindexer.get_percentile_bands()
        self.id_documentation = reindexer.get_id_documentation()

        return self.reindexed_vocab

    def _generate_mappings(self, working_vocab: Dict[str, int]):
        """Generate old_id -> new_id mappings."""
        if self.verbose:
            print("\n  🔄 Generating ID mappings...")

        # Build mappings
        for token, new_id in self.reindexed_vocab.items():
            if token in working_vocab:
                old_id = working_vocab[token]
                self.mapping[old_id] = new_id
                self.reverse_mapping[new_id] = old_id

        if self.verbose:
            print(f"     ✓ Created mappings for {len(self.mapping):,} tokens")

    def save_vocabulary(self, output_path: Path):
        """Save reindexed vocabulary to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort by new ID
        sorted_vocab = dict(sorted(self.reindexed_vocab.items(), key=lambda x: x[1]))

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_vocab, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"\n💾 Reindexed vocabulary saved to: {output_path}")

    def save_mappings(self, output_path: Path):
        """Save ID mappings for reference (optional)."""
        mapping_path = output_path.parent / f"{output_path.stem}_id_mapping.json"

        mapping_data = {
            "old_to_new": {str(k): v for k, v in sorted(self.mapping.items())},
            "new_to_old": {str(k): v for k, v in sorted(self.reverse_mapping.items())},
            "metadata": {
                "total_tokens": len(self.reindexed_vocab),
                "timestamp": datetime.now().isoformat(),
            },
        }

        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping_data, f, indent=2)

        if self.verbose:
            print(f"💾 ID mappings saved to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reindex tokenizer based on RRF algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline: Token selection + reindexing from filtered tokenizers (RRF always used)
  python tokenizer_reindexing.py \\
      --input-dir filtered_tokenizer \\
      --output reindexed_tokenizer_128k.json \\
      --target-size 128000 \\
      --reserved-count 200
  
  # With long tokens CSV for reference
  python tokenizer_reindexing.py \\
      --input-dir filtered_tokenizer \\
      --output reindexed_tokenizer_128k.json \\
      --long-tokens-csv tokenizer_results/long_tokens_32plus_*.csv \\
      --target-size 128000
        """,
    )

    # Input/output (can be file OR directory)
    parser.add_argument(
        "--input", "-i", help="Input merged tokenizer JSON file (single vocab)"
    )
    parser.add_argument(
        "--input-dir",
        help="Input directory with filtered tokenizers (for selection + reindexing)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output reindexed tokenizer JSON file"
    )

    parser.add_argument(
        "--config",
        "-c",
        default="tokenizer_config.json",
        help="Tokenizer config file (default: tokenizer_config.json)",
    )

    # Configuration
    parser.add_argument(
        "--special-tokens-config", help="Special tokens configuration JSON"
    )
    parser.add_argument(
        "--smoothing",
        "-s",
        type=float,
        default=0.1,
        help="Smoothing factor for frequency ordering (0.0-1.0, default: 0.1)",
    )

    # RRF options (RRF is always used - it's the only method)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF constant k (default: 60, higher = less impact of low ranks)",
    )

    # Reserved IDs (at the end of vocabulary)
    parser.add_argument(
        "--reserved-count",
        type=int,
        default=200,
        help="Number of IDs to reserve at end (default: 200)",
    )
    parser.add_argument(
        "--target-size",
        "-t",
        type=int,
        default=128000,
        help="Target vocabulary size (default: 128000)",
    )

    # Output options
    parser.add_argument(
        "--save-mappings", action="store_true", help="Save old_id -> new_id mappings"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (minimal output)"
    )

    # Validation options
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip vocabulary validation (not recommended)",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=32,
        help="Maximum token length for strict filtering (default: 32)",
    )
    parser.add_argument(
        "--long-tokens-csv",
        help="CSV file with tokens > max_token_length for reference/analysis (e.g., long_tokens_32plus.csv)",
    )

    args = parser.parse_args()

    # Validate input arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir must be specified")

    if args.input and args.input_dir:
        parser.error("Cannot specify both --input and --input-dir (use one)")

    verbose = not args.quiet

    if verbose:
        print("=" * 80)
        print("TOKENIZER REINDEXING - FREQUENCY-AWARE ID ASSIGNMENT")
        print("=" * 80)

    # Initialize reindexer
    special_tokens_config = (
        Path(args.special_tokens_config) if args.special_tokens_config else None
    )
    reindexer = TokenizerReindexer(
        special_tokens_config,
        verbose=verbose,
        max_token_length=args.max_token_length,
        config_path=args.config,
    )

    # Set reserved range at the end
    if args.reserved_count:
        reserved_start = args.target_size - args.reserved_count
        reserved_end = args.target_size - 1
        reindexer.special_tokens_mgr.reserved_range = (reserved_start, reserved_end)
        if verbose:
            print("\n⚙️  Configuration:")
            print(f"   Target size: {args.target_size:,}")
            print(
                f"   Max token length: {args.max_token_length} chars (strict enforcement)"
            )
            print(
                f"   Reserved IDs: {args.reserved_count} (at end: {reserved_start}-{reserved_end})"
            )
            if args.long_tokens_csv:
                print(f"   Long tokens CSV: {args.long_tokens_csv}")

    # Load input (directory or single file)
    validate = not args.skip_validation

    if args.input_dir:
        # Load from directory (will do selection + reindexing)
        input_dir = Path(args.input_dir)
        loaded = reindexer.load_directory(input_dir, validate=validate)

        if loaded == 0:
            print("\n❌ No tokenizers loaded from directory")
            return 1

        if verbose:
            print("\n✓ Ready for token selection and reindexing")
            print(f"  Mode: SELECTION + REINDEXING (from {loaded} tokenizers)")

    else:
        # Load single vocabulary file (will do reindexing only)
        input_path = Path(args.input)
        if not reindexer.load_vocabulary(input_path, validate=validate):
            if validate:
                print("\n💡 TIP: Run the complete pipeline first:")
                print(
                    "   1. python tokenizer_filter.py --input-dir ../data/ --output-dir filtered_tokenizer"
                )
                print(
                    "   2. python tokenizer_reindexing.py --input-dir filtered_tokenizer --output reindexed.json"
                )
            return 1

        if verbose:
            print("\n✓ Ready for reindexing")
            print("  Mode: REINDEXING ONLY (pre-merged vocab)")

    # Load long tokens CSV if provided (for reference/analysis)
    if args.long_tokens_csv:
        long_tokens_csv = Path(args.long_tokens_csv)
        if long_tokens_csv.exists():
            if verbose:
                print(f"\n  ℹ️  Long tokens CSV provided: {long_tokens_csv.name}")
                print("     (These tokens are already filtered during collection)")
        else:
            print(f"\n  ⚠️  Warning: Long tokens CSV not found: {long_tokens_csv}")

    # Execute selection and reindexing (RRF is always used)
    # Note: Frequency analysis is done separately by generate_tokenizer_report.py
    reindexer.select_and_reindex(
        smoothing_factor=args.smoothing, target_size=args.target_size, rrf_k=args.rrf_k
    )

    # Save output (ONLY the JSON file)
    output_path = Path(args.output)
    reindexer.save_vocabulary(output_path)

    if args.save_mappings:
        reindexer.save_mappings(output_path)

    if verbose:
        print("\n" + "=" * 80)
        print("✅ REINDEXING COMPLETE")
        print("=" * 80)
        print(f"\nOutput file: {output_path}")
        print(f"  Total tokens: {len(reindexer.reindexed_vocab):,}")

        # Count tokens by ID range
        special_count = sum(1 for v in reindexer.reindexed_vocab.values() if v < 45)
        regular_count = sum(
            1 for v in reindexer.reindexed_vocab.values() if 45 <= v < 127800
        )
        reserved_count = sum(
            1 for v in reindexer.reindexed_vocab.values() if v >= 127800
        )

        print(f"  Special tokens (0-44): {special_count}")
        print(f"  Regular tokens (45-127799): {regular_count}")
        print(f"  Reserved IDs used (127800+): {reserved_count}")

        # Check for long tokens
        long_tokens = {
            k: v
            for k, v in reindexer.reindexed_vocab.items()
            if len(k) > args.max_token_length
        }
        if long_tokens:
            print(f"  ⚠️  Tokens > {args.max_token_length} chars: {len(long_tokens)}")
        else:
            print(f"  ✅ All tokens ≤ {args.max_token_length} characters")

        if args.save_mappings:
            print("\n  Optional: ID mappings saved")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
