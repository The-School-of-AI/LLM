"""
MoE Routing Validation - Validates frequency-ID correlation for MoE safety.

Checks that token ID ordering doesn't create perfect correlation with frequency,
which could lead to routing skew in Mixture-of-Experts models.

Usage:
    python moe_validation.py \\
        --frequency-stats ../results/frequency_stats/ds_merged_freq.json \\
        --reindexed ../results/reindexed_tokenizers/ds_reindexed/ \\
        --config ../config.yaml
"""

import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from scipy.stats import spearmanr
from loguru import logger


@dataclass
class MoEValidationResult:
    """Container for MoE routing validation results."""
    passed: bool
    spearman_correlation: float
    max_allowed_correlation: float
    vocab_skew_score: float
    entropy: float
    id_frequency_pairs: int
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class MoEValidator:
    """Validates tokenizer for MoE routing safety."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.moe_config = self.config['reindexing']['validation']['moe_routing']

    def load_frequency_stats(self, stats_path: str) -> Dict[int, int]:
        """Load frequency statistics (old_id -> frequency)."""
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Convert string keys back to ints
        token_frequencies = {int(k): v for k, v in stats['token_frequencies'].items()}

        logger.info(f"Loaded frequency stats with {len(token_frequencies)} tokens")
        return token_frequencies

    def load_id_mapping(self, reindexed_dir: str) -> Dict[int, int]:
        """Load old_id -> new_id mapping."""
        reindexed_path = Path(reindexed_dir)
        id_mapping_file = reindexed_path / "id_mapping.json"

        with open(id_mapping_file, 'r') as f:
            id_mapping_str = json.load(f)
            id_mapping = {int(k): v for k, v in id_mapping_str.items()}

        logger.info(f"Loaded ID mapping with {len(id_mapping)} entries")
        return id_mapping

    def compute_spearman_correlation(
        self,
        frequencies: Dict[int, int],
        id_mapping: Dict[int, int]
    ) -> Tuple[float, int]:
        """
        Compute Spearman correlation between new token IDs and frequencies.

        Returns:
            - correlation coefficient
            - number of ID-frequency pairs
        """
        # Build pairs: (new_id, frequency)
        pairs = []
        for old_id, new_id in id_mapping.items():
            if old_id in frequencies:
                pairs.append((new_id, frequencies[old_id]))

        if not pairs:
            logger.warning("No ID-frequency pairs found!")
            return 0.0, 0

        # Extract new IDs and frequencies
        new_ids = [p[0] for p in pairs]
        freqs = [p[1] for p in pairs]

        # Compute Spearman correlation
        correlation, p_value = spearmanr(new_ids, freqs)

        logger.info(f"Spearman correlation: {correlation:.4f} (p-value: {p_value:.4e})")

        return float(correlation), len(pairs)

    def compute_vocab_skew(self, frequencies: Dict[int, int]) -> float:
        """
        Compute vocabulary skew using Gini coefficient.

        Measures inequality in token frequency distribution.
        Higher skew = more unequal distribution.
        """
        freq_values = sorted(frequencies.values())
        n = len(freq_values)

        if n == 0:
            return 0.0

        # Gini coefficient
        cumsum = np.cumsum(freq_values)
        gini = (2 * np.sum((np.arange(1, n + 1)) * freq_values)) / (n * np.sum(freq_values)) - (n + 1) / n

        logger.info(f"Vocabulary skew (Gini): {gini:.4f}")

        return float(gini)

    def compute_entropy(self, frequencies: Dict[int, int]) -> float:
        """
        Compute Shannon entropy of token frequency distribution.

        Higher entropy = more uniform distribution.
        """
        freq_values = list(frequencies.values())
        total = sum(freq_values)

        if total == 0:
            return 0.0

        # Normalize to probabilities
        probs = [f / total for f in freq_values]

        # Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        logger.info(f"Frequency entropy: {entropy:.4f} bits")

        return float(entropy)

    def validate(
        self,
        frequency_stats_path: str,
        reindexed_dir: str
    ) -> MoEValidationResult:
        """
        Run MoE routing validation.

        Checks:
        1. Spearman correlation between new IDs and frequencies (should not be perfect)
        2. Vocabulary skew (for context)
        3. Entropy (for context)
        """
        logger.info(f"\n{'='*60}")
        logger.info("MoE Routing Validation")
        logger.info(f"{'='*60}")

        # Load data
        frequencies = self.load_frequency_stats(frequency_stats_path)
        id_mapping = self.load_id_mapping(reindexed_dir)

        # Compute metrics
        correlation, pair_count = self.compute_spearman_correlation(frequencies, id_mapping)
        vocab_skew = self.compute_vocab_skew(frequencies)
        entropy = self.compute_entropy(frequencies)

        # Check threshold
        max_correlation = self.moe_config['max_spearman_correlation']
        passed = abs(correlation) < max_correlation

        # Build result
        if passed:
            message = f"✓ MoE validation passed: correlation {correlation:.4f} < {max_correlation}"
        else:
            message = f"✗ MoE validation failed: correlation {correlation:.4f} >= {max_correlation}"

        result = MoEValidationResult(
            passed=passed,
            spearman_correlation=correlation,
            max_allowed_correlation=max_correlation,
            vocab_skew_score=vocab_skew,
            entropy=entropy,
            id_frequency_pairs=pair_count,
            message=message,
            details={
                "interpretation": {
                    "correlation": "Measures how strongly new IDs correlate with frequency. Perfect correlation (±1.0) is bad for MoE.",
                    "vocab_skew": "Gini coefficient. Higher = more unequal token distribution. Range: [0, 1]",
                    "entropy": "Shannon entropy of frequency distribution. Higher = more uniform."
                },
                "recommendations": self._get_recommendations(correlation, max_correlation, vocab_skew)
            }
        )

        logger.info(f"\n{message}")
        logger.info(f"  Correlation: {correlation:.4f}")
        logger.info(f"  Vocab skew: {vocab_skew:.4f}")
        logger.info(f"  Entropy: {entropy:.4f} bits")

        return result

    def _get_recommendations(
        self,
        correlation: float,
        max_correlation: float,
        vocab_skew: float
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if abs(correlation) >= max_correlation:
            recommendations.append(
                "⚠ Correlation too high! Consider increasing smoothing temperature in config.yaml"
            )
            recommendations.append(
                f"  Current: {self.config['reindexing']['moe_smoothing']['temperature']}"
            )
            recommendations.append(
                "  Suggestion: Try temperature=0.2 or temperature=0.5 for more aggressive smoothing"
            )

        if abs(correlation) > 0.9:
            recommendations.append(
                "⚠ Very high correlation detected. This may cause MoE routing to be frequency-biased."
            )

        if vocab_skew > 0.8:
            recommendations.append(
                "ℹ High vocabulary skew detected. This is normal for natural language (Zipf's law)."
            )

        if not recommendations:
            recommendations.append("✓ No issues detected. Token ID ordering looks good for MoE usage.")

        return recommendations


def main():
    parser = argparse.ArgumentParser(description="MoE Routing Validator")
    parser.add_argument('--frequency-stats', type=str, required=True,
                        help='Path to frequency statistics JSON')
    parser.add_argument('--reindexed', type=str, required=True,
                        help='Path to reindexed tokenizer directory')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for validation report JSON')

    args = parser.parse_args()

    # Create validator
    validator = MoEValidator(args.config)

    # Run validation
    result = validator.validate(
        frequency_stats_path=args.frequency_stats,
        reindexed_dir=args.reindexed
    )

    # Print recommendations
    if result.details and 'recommendations' in result.details:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        for rec in result.details['recommendations']:
            print(rec)

    # Save report if output specified
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        logger.info(f"\nMoE validation report saved to: {output_file}")

    # Exit with appropriate code
    exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
