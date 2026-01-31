"""
Validation Suite - Comprehensive tests for reindexed tokenizers.

Validates that reindexing preserves:
- Token strings (unchanged)
- Encoding/decoding equivalence
- Special token handling
- Merge rules (implicitly through encoding)

Usage:
    python validation_suite.py \\
        --original ds_filtered \\
        --reindexed ../results/reindexed_tokenizers/ds_reindexed/ \\
        --config ../config.yaml
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
from special_tokens import SpecialTokenRegistry


@dataclass
class ValidationResult:
    """Container for validation results."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class TokenizerValidator:
    """Validates reindexed tokenizers."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_path = Path(self.config['tokenizer_sources']['base_path'])
        self.special_token_registry = SpecialTokenRegistry(config_path)

    def load_original_tokenizer(self, tokenizer_name: str) -> Dict[str, int]:
        """Load original tokenizer."""
        json_path = self.base_path / f"{tokenizer_name}.json"

        if not json_path.exists():
            raise FileNotFoundError(f"Original tokenizer not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)

        logger.info(f"Loaded original tokenizer with {len(token_to_id)} tokens")
        return token_to_id

    def load_reindexed_tokenizer(self, reindexed_dir: str) -> Tuple[Dict[str, int], Dict[int, int], Dict[str, Any]]:
        """Load reindexed tokenizer and metadata."""
        reindexed_path = Path(reindexed_dir)

        # Load token -> ID mapping
        tokenizer_file = reindexed_path / "tokenizer_reindexed.json"
        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)

        # Load old ID -> new ID mapping
        id_mapping_file = reindexed_path / "id_mapping.json"
        with open(id_mapping_file, 'r') as f:
            id_mapping_str = json.load(f)
            id_mapping = {int(k): v for k, v in id_mapping_str.items()}

        # Load metadata
        metadata_file = reindexed_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        logger.info(f"Loaded reindexed tokenizer with {len(token_to_id)} tokens")
        return token_to_id, id_mapping, metadata

    def test_vocab_size(self, original: Dict[str, int], reindexed: Dict[str, int]) -> ValidationResult:
        """Test that vocab size is preserved (or correctly adjusted for special tokens)."""
        original_size = len(original)
        reindexed_size = len(reindexed)
        special_tokens_count = len(self.special_token_registry.get_all_special_tokens())

        # Reindexed vocab should be >= original (may include special tokens)
        expected_min_size = original_size

        passed = reindexed_size >= expected_min_size

        return ValidationResult(
            test_name="vocab_size",
            passed=passed,
            message=f"Vocab size: original={original_size}, reindexed={reindexed_size}, special={special_tokens_count}",
            details={
                "original_size": original_size,
                "reindexed_size": reindexed_size,
                "special_tokens_count": special_tokens_count
            }
        )

    def test_token_strings_preserved(self, original: Dict[str, int], reindexed: Dict[str, int]) -> ValidationResult:
        """Test that all original token strings are preserved in reindexed vocab."""
        original_tokens = set(original.keys())
        reindexed_tokens = set(reindexed.keys())

        # All original tokens should be in reindexed (or replaced by special tokens)
        missing_tokens = original_tokens - reindexed_tokens
        extra_tokens = reindexed_tokens - original_tokens

        # Extra tokens should only be special tokens
        special_token_strings = {t.token for t in self.special_token_registry.get_all_special_tokens()}
        unexpected_extra = extra_tokens - special_token_strings

        passed = len(missing_tokens) == 0 and len(unexpected_extra) == 0

        return ValidationResult(
            test_name="token_strings_preserved",
            passed=passed,
            message=f"Token strings: {len(original_tokens)} original, {len(missing_tokens)} missing, {len(extra_tokens)} extra",
            details={
                "original_count": len(original_tokens),
                "missing_count": len(missing_tokens),
                "extra_count": len(extra_tokens),
                "unexpected_extra_count": len(unexpected_extra),
                "missing_tokens": list(missing_tokens)[:10],  # First 10 for debugging
                "unexpected_extra_tokens": list(unexpected_extra)[:10]
            }
        )

    def test_id_mapping_consistency(
        self,
        original: Dict[str, int],
        reindexed: Dict[str, int],
        id_mapping: Dict[int, int]
    ) -> ValidationResult:
        """Test that ID mapping is consistent."""
        inconsistencies = []

        for token, original_id in original.items():
            if token in reindexed:
                reindexed_id = reindexed[token]
                expected_new_id = id_mapping.get(original_id)

                if expected_new_id is not None and expected_new_id != reindexed_id:
                    inconsistencies.append({
                        'token': token,
                        'original_id': original_id,
                        'expected_new_id': expected_new_id,
                        'actual_new_id': reindexed_id
                    })

        passed = len(inconsistencies) == 0

        return ValidationResult(
            test_name="id_mapping_consistency",
            passed=passed,
            message=f"ID mapping: {len(inconsistencies)} inconsistencies found",
            details={
                "inconsistency_count": len(inconsistencies),
                "inconsistencies": inconsistencies[:10]  # First 10 for debugging
            }
        )

    def test_special_tokens(self, reindexed: Dict[str, int]) -> ValidationResult:
        """Test that special tokens are correctly included."""
        validation_report = self.special_token_registry.validate_tokenizer(reindexed)

        passed = validation_report['valid']

        return ValidationResult(
            test_name="special_tokens",
            passed=passed,
            message=f"Special tokens: {validation_report['present_tokens']}/{validation_report['total_special_tokens']} present",
            details=validation_report
        )

    def test_encode_decode_equivalence(
        self,
        original: Dict[str, int],
        reindexed: Dict[str, int],
        test_samples: List[str]
    ) -> ValidationResult:
        """Test that encoding and decoding produces same results."""
        from tokenizer_evaluator import SimpleTokenizer

        original_tokenizer = SimpleTokenizer(original)
        reindexed_tokenizer = SimpleTokenizer(reindexed)

        mismatches = []

        for i, text in enumerate(test_samples):
            # Encode with both tokenizers
            original_ids = original_tokenizer.encode(text)
            reindexed_ids = reindexed_tokenizer.encode(text)

            # Decode with both tokenizers
            original_decoded = original_tokenizer.decode(original_ids)
            reindexed_decoded = reindexed_tokenizer.decode(reindexed_ids)

            # Check if decoded text matches
            if original_decoded != reindexed_decoded:
                mismatches.append({
                    'sample_index': i,
                    'original_text': text[:50],
                    'original_decoded': original_decoded[:50],
                    'reindexed_decoded': reindexed_decoded[:50],
                    'original_token_count': len(original_ids),
                    'reindexed_token_count': len(reindexed_ids)
                })

        passed = len(mismatches) == 0

        return ValidationResult(
            test_name="encode_decode_equivalence",
            passed=passed,
            message=f"Encode/Decode: {len(test_samples) - len(mismatches)}/{len(test_samples)} samples passed",
            details={
                "total_samples": len(test_samples),
                "passed_samples": len(test_samples) - len(mismatches),
                "failed_samples": len(mismatches),
                "mismatches": mismatches[:5]  # First 5 for debugging
            }
        )

    def test_frequency_ordering(
        self,
        reindexed: Dict[str, int],
        metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Test that tokens are ordered according to frequency strategy."""
        strategy = metadata.get('strategy', 'unknown')

        if strategy == "category_blocks":
            # Check that ID ranges are as expected
            id_ranges = metadata.get('id_ranges', {})

            # Verify ranges don't overlap and are in correct order
            range_issues = []

            expected_order = ['special', 'high_frequency', 'medium_frequency', 'low_frequency']
            prev_end = -1

            for category in expected_order:
                if category not in id_ranges:
                    continue

                range_info = id_ranges[category]
                start = range_info['start']
                end = range_info.get('end', start)

                if start <= prev_end:
                    range_issues.append(f"{category} range starts at {start}, overlaps with previous range ending at {prev_end}")

                prev_end = end

            passed = len(range_issues) == 0

            return ValidationResult(
                test_name="frequency_ordering",
                passed=passed,
                message=f"Frequency ordering ({strategy}): {'valid' if passed else 'issues found'}",
                details={
                    "strategy": strategy,
                    "id_ranges": id_ranges,
                    "range_issues": range_issues
                }
            )
        else:
            # For other strategies, just check that IDs are assigned
            passed = len(reindexed) > 0

            return ValidationResult(
                test_name="frequency_ordering",
                passed=passed,
                message=f"Frequency ordering ({strategy}): basic check passed",
                details={"strategy": strategy}
            )

    def run_all_tests(
        self,
        original_tokenizer_name: str,
        reindexed_dir: str,
        test_samples: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run all validation tests."""
        logger.info(f"\n{'='*80}")
        logger.info("VALIDATION SUITE")
        logger.info(f"{'='*80}")

        # Load tokenizers
        original = self.load_original_tokenizer(original_tokenizer_name)
        reindexed, id_mapping, metadata = self.load_reindexed_tokenizer(reindexed_dir)

        # Default test samples if not provided
        if test_samples is None:
            test_samples = [
                "Hello, world!",
                "भारत एक विशाल देश है।",
                "def calculate_sum(a, b):\n    return a + b",
                '{"name": "Alice", "age": 30}',
                "The quick brown fox jumps over the lazy dog.",
                "संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति।",
                "[1, 2, 3, 4, 5]",
                "print('Hello, World!')"
            ]

        # Run tests
        results = []

        logger.info("\n1. Testing vocab size...")
        results.append(self.test_vocab_size(original, reindexed))

        logger.info("2. Testing token strings preservation...")
        results.append(self.test_token_strings_preserved(original, reindexed))

        logger.info("3. Testing ID mapping consistency...")
        results.append(self.test_id_mapping_consistency(original, reindexed, id_mapping))

        logger.info("4. Testing special tokens...")
        results.append(self.test_special_tokens(reindexed))

        logger.info("5. Testing encode/decode equivalence...")
        results.append(self.test_encode_decode_equivalence(original, reindexed, test_samples))

        logger.info("6. Testing frequency ordering...")
        results.append(self.test_frequency_ordering(reindexed, metadata))

        # Summarize results
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)

        logger.info(f"\n{'='*80}")
        logger.info("VALIDATION RESULTS")
        logger.info(f"{'='*80}")

        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            logger.info(f"{status}: {result.test_name}")
            logger.info(f"       {result.message}")

        logger.info(f"\n{'='*80}")
        logger.info(f"Summary: {passed_count}/{total_count} tests passed")
        logger.info(f"{'='*80}\n")

        all_passed = passed_count == total_count

        return {
            "all_passed": all_passed,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": [asdict(r) for r in results]
        }


def main():
    parser = argparse.ArgumentParser(description="Tokenizer Validation Suite")
    parser.add_argument('--original', type=str, required=True,
                        help='Original tokenizer name (e.g., ds_filtered)')
    parser.add_argument('--reindexed', type=str, required=True,
                        help='Path to reindexed tokenizer directory')
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for validation report JSON')

    args = parser.parse_args()

    # Create validator
    validator = TokenizerValidator(args.config)

    # Run tests
    report = validator.run_all_tests(
        original_tokenizer_name=args.original,
        reindexed_dir=args.reindexed
    )

    # Save report if output specified
    if args.output:
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Validation report saved to: {output_file}")

    # Exit with appropriate code
    exit(0 if report['all_passed'] else 1)


if __name__ == "__main__":
    main()
