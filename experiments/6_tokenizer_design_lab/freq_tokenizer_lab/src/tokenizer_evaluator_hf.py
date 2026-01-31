"""
Tokenizer Evaluator (HuggingFace Dataset Version)
Evaluates tokenizers on real datasets instead of hardcoded samples.

Usage:
    python tokenizer_evaluator_hf.py --config ../config.yaml
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import unicodedata
import re
from loguru import logger
from datasets import load_dataset
from tqdm import tqdm
import sys


@dataclass
class TokenizerScore:
    """Score container for a single tokenizer."""
    name: str
    indic_score: float = 0.0
    code_score: float = 0.0
    json_score: float = 0.0
    overall_score: float = 0.0
    indic_metrics: Dict[str, Any] = None
    code_metrics: Dict[str, Any] = None
    json_metrics: Dict[str, Any] = None
    passed_hard_filters: bool = True
    rejection_reasons: List[str] = None

    def __post_init__(self):
        if self.indic_metrics is None:
            self.indic_metrics = {}
        if self.code_metrics is None:
            self.code_metrics = {}
        if self.json_metrics is None:
            self.json_metrics = {}
        if self.rejection_reasons is None:
            self.rejection_reasons = []


class TokenizerLoader:
    """Loads tokenizers from filtered JSON files."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def load_tokenizer(self, name: str) -> Dict[str, int]:
        """Load token-to-ID mapping from JSON."""
        json_path = self.base_path / f"{name}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            token_to_id = json.load(f)

        logger.info(f"Loaded tokenizer '{name}' with {len(token_to_id)} tokens")
        return token_to_id

    def load_all_tokenizers(self, tokenizer_names: List[str]) -> Dict[str, Dict[str, int]]:
        """Load all specified tokenizers."""
        tokenizers = {}
        for name in tokenizer_names:
            try:
                tokenizers[name] = self.load_tokenizer(name)
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{name}': {e}")
        return tokenizers


class SimpleTokenizer:
    """Simple tokenizer wrapper for greedy longest-match tokenization."""

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}
        # Sort tokens by length (longest first) for greedy matching
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
                # Byte fallback for unmatched character
                char = text[pos]
                char_bytes = char.encode('utf-8')
                for byte in char_bytes:
                    # Use byte token if available, otherwise mark as unknown
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.token_to_id:
                        ids.append(self.token_to_id[byte_token])
                    else:
                        # Fallback: use token ID 0 or create synthetic ID
                        ids.append(0)
                pos += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(id, "") for id in ids]
        return "".join(tokens)


class IndicBenchmarkHF:
    """Benchmark suite for Indic language quality using HuggingFace IndicCorpV2."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config['evaluation_hf']['indic']
        self.thresholds = config['evaluation']['indic']['failure_thresholds']

    def is_devanagari(self, char: str) -> bool:
        """Check if character is Devanagari script."""
        try:
            return 0x0900 <= ord(char) <= 0x097F
        except:
            return False

    def is_byte_fallback(self, token: str) -> bool:
        """Check if token is a byte fallback token."""
        return token.startswith("<0x") and token.endswith(">")

    def compute_tokens_per_char(self, text: str, ids: List[int]) -> float:
        """Compute token efficiency (lower is better)."""
        return len(ids) / len(text) if len(text) > 0 else float('inf')

    def compute_byte_fallback_rate(self, text: str, ids: List[int], tokenizer: SimpleTokenizer) -> float:
        """Compute percentage of tokens that are byte fallbacks."""
        byte_fallback_count = sum(1 for id in ids if self.is_byte_fallback(tokenizer.id_to_token.get(id, "")))
        return byte_fallback_count / len(ids) if len(ids) > 0 else 0.0

    def compute_fragmentation_score(self, text: str, ids: List[int]) -> float:
        """Measure text fragmentation (higher = more fragmented)."""
        # Ideal: 1 token per word; fragmented: many tokens per word
        words = text.split()
        if len(words) == 0:
            return 0.0
        return len(ids) / len(words)

    def compute_devanagari_quality(self, text: str, ids: List[int], tokenizer: SimpleTokenizer) -> float:
        """Specific quality metric for Devanagari script."""
        devanagari_chars = [c for c in text if self.is_devanagari(c)]
        if len(devanagari_chars) == 0:
            return 1.0  # No Devanagari, perfect score by default

        # Decode and check if Devanagari is preserved
        decoded = tokenizer.decode(ids)
        preserved_count = sum(1 for c in devanagari_chars if c in decoded)
        return preserved_count / len(devanagari_chars)

    def evaluate_tokenizer(self, tokenizer: SimpleTokenizer, tokenizer_name: str) -> Dict[str, Any]:
        """Run full Indic benchmark on tokenizer using HuggingFace dataset."""
        logger.info(f"Running Indic benchmark on '{tokenizer_name}' using IndicCorpV2")

        dataset_name = self.eval_config['dataset']
        sample_size = self.eval_config['sample_size']
        text_column = self.eval_config['text_column']
        split = self.eval_config.get('split', 'hin_Deva')  # Default to Hindi Devanagari

        # Load dataset in streaming mode
        logger.info(f"Loading dataset: {dataset_name} split={split} (streaming)")
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=True)
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            return {
                "by_language": {},
                "aggregate": {},
                "passed_filters": False,
                "rejection_reasons": [f"Dataset loading failed: {e}"]
            }

        # Collect metrics across samples
        all_metrics = defaultdict(list)
        processed = 0

        logger.info(f"Processing {sample_size} samples from IndicCorpV2...")
        for sample in tqdm(dataset, total=sample_size, desc=f"Evaluating {tokenizer_name}"):
            if processed >= sample_size:
                break

            text = sample.get(text_column, "")
            if not text or len(text) < 10:  # Skip very short texts
                continue

            # Tokenize
            ids = tokenizer.encode(text)

            # Compute metrics
            tokens_per_char = self.compute_tokens_per_char(text, ids)
            byte_fallback_rate = self.compute_byte_fallback_rate(text, ids, tokenizer)
            fragmentation_score = self.compute_fragmentation_score(text, ids)
            devanagari_quality = self.compute_devanagari_quality(text, ids, tokenizer)

            all_metrics["tokens_per_char"].append(tokens_per_char)
            all_metrics["byte_fallback_rate"].append(byte_fallback_rate)
            all_metrics["fragmentation_score"].append(fragmentation_score)
            all_metrics["devanagari_quality"].append(devanagari_quality)
            all_metrics["num_tokens"].append(len(ids))
            all_metrics["num_chars"].append(len(text))

            processed += 1

        # Compute aggregate statistics
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in all_metrics.items()
        }

        # Check hard filters
        passed = True
        reasons = []

        if avg_metrics["byte_fallback_rate"] > self.thresholds["byte_fallback_rate"]:
            passed = False
            reasons.append(f"High byte fallback rate: {avg_metrics['byte_fallback_rate']:.2%}")

        if avg_metrics["tokens_per_char"] > self.thresholds["tokens_per_char"]:
            passed = False
            reasons.append(f"Excessive tokenization: {avg_metrics['tokens_per_char']:.2f} tokens/char")

        logger.info(f"Indic evaluation complete: {processed} samples processed")
        logger.info(f"  Avg tokens/char: {avg_metrics['tokens_per_char']:.3f}")
        logger.info(f"  Avg byte fallback: {avg_metrics['byte_fallback_rate']:.2%}")
        logger.info(f"  Avg Devanagari quality: {avg_metrics['devanagari_quality']:.3f}")

        return {
            "samples_processed": processed,
            "aggregate": avg_metrics,
            "passed_filters": passed,
            "rejection_reasons": reasons
        }


class CodeBenchmarkHF:
    """Benchmark suite for code tokenization using HuggingFace Dolma datasets."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config['evaluation_hf']['code']
        self.thresholds = config['evaluation']['code']['failure_thresholds']

    def is_code_content(self, text: str) -> bool:
        """Heuristic to check if text contains code."""
        code_indicators = ['{', '}', '(', ')', 'function', 'def ', 'class ', 'import ', '#include', 'const ', 'let ', 'var ']
        return any(indicator in text for indicator in code_indicators)

    def compute_tokens_per_line(self, code: str, ids: List[int]) -> float:
        """Compute average tokens per line of code."""
        lines = [line for line in code.split('\n') if line.strip()]
        return len(ids) / len(lines) if len(lines) > 0 else float('inf')

    def compute_symbol_preservation(self, code: str, ids: List[int], tokenizer: SimpleTokenizer) -> float:
        """Check if important symbols are preserved (braces, operators, etc.)."""
        important_symbols = ['{', '}', '(', ')', '[', ']', '=', ':', ';', ',', '.']
        decoded = tokenizer.decode(ids)

        preserved_count = sum(1 for sym in important_symbols if code.count(sym) == decoded.count(sym))
        return preserved_count / len(important_symbols)

    def compute_structure_quality(self, code: str, ids: List[int]) -> float:
        """Measure how well code structure is tokenized (chars per token, higher is better)."""
        return len(code) / len(ids) if len(ids) > 0 else 0.0

    def evaluate_tokenizer(self, tokenizer: SimpleTokenizer, tokenizer_name: str) -> Dict[str, Any]:
        """Run full code benchmark on tokenizer using HuggingFace Dolma."""
        logger.info(f"Running code benchmark on '{tokenizer_name}' using Dolma")

        all_metrics = defaultdict(list)
        processed = 0

        # Process each dataset
        for dataset_config in self.eval_config['datasets']:
            dataset_name = dataset_config['name']
            sample_size = dataset_config['sample_size']
            text_column = dataset_config['text_column']
            sample_fraction = dataset_config.get('sample_fraction', 1.0)
            split = dataset_config.get('split', 'train')

            logger.info(f"Loading dataset: {dataset_name} split={split} (streaming)")
            try:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
            except Exception as e:
                logger.error(f"Failed to load dataset {dataset_name}: {e}")
                continue

            dataset_processed = 0
            logger.info(f"Processing up to {sample_size} samples from {dataset_name}...")

            for sample in tqdm(dataset, total=sample_size, desc=f"{tokenizer_name} - {dataset_name}"):
                if dataset_processed >= sample_size:
                    break

                # Sample only a fraction if specified
                if sample_fraction < 1.0:
                    import random
                    if random.random() > sample_fraction:
                        continue

                text = sample.get(text_column, "")
                if not text or len(text) < 50:  # Skip very short texts
                    continue

                # Filter for code-like content (for general Dolma)
                if "dolmino_mix" not in dataset_name.lower() and not self.is_code_content(text):
                    continue

                # Tokenize
                ids = tokenizer.encode(text)

                # Compute metrics
                tokens_per_line = self.compute_tokens_per_line(text, ids)
                symbol_preservation = self.compute_symbol_preservation(text, ids, tokenizer)
                structure_quality = self.compute_structure_quality(text, ids)

                all_metrics["tokens_per_line"].append(tokens_per_line)
                all_metrics["symbol_preservation"].append(symbol_preservation)
                all_metrics["structure_quality"].append(structure_quality)
                all_metrics["num_tokens"].append(len(ids))
                all_metrics["num_chars"].append(len(text))

                dataset_processed += 1
                processed += 1

            logger.info(f"Processed {dataset_processed} samples from {dataset_name}")

        # Compute aggregate statistics
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in all_metrics.items()
        }

        # Check hard filters
        passed = True
        reasons = []

        if processed == 0:
            passed = False
            reasons.append("No samples processed from datasets")
        elif avg_metrics.get("tokens_per_line", 0) > self.thresholds["tokens_per_line"]:
            passed = False
            reasons.append(f"Excessive tokens per line: {avg_metrics['tokens_per_line']:.2f}")

        logger.info(f"Code evaluation complete: {processed} samples processed")
        if processed > 0:
            logger.info(f"  Avg tokens/line: {avg_metrics.get('tokens_per_line', 0.0):.3f}")
            logger.info(f"  Avg symbol preservation: {avg_metrics.get('symbol_preservation', 0.0):.3f}")
            logger.info(f"  Avg chars/token: {avg_metrics.get('structure_quality', 0.0):.3f}")

        return {
            "samples_processed": processed,
            "aggregate": avg_metrics,
            "passed_filters": passed,
            "rejection_reasons": reasons
        }


class TokenizerEvaluatorHF:
    """Main evaluator orchestrating all HuggingFace-based benchmarks."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.loader = TokenizerLoader(self.config['tokenizer_sources']['base_path'])
        self.indic_benchmark = IndicBenchmarkHF(self.config)
        self.code_benchmark = CodeBenchmarkHF(self.config)

    def evaluate_single_tokenizer(self, name: str, token_to_id: Dict[str, int]) -> TokenizerScore:
        """Evaluate a single tokenizer across all benchmarks."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating tokenizer: {name}")
        logger.info(f"{'='*60}")

        tokenizer = SimpleTokenizer(token_to_id)
        score = TokenizerScore(name=name)

        # Run benchmarks
        indic_results = self.indic_benchmark.evaluate_tokenizer(tokenizer, name)
        code_results = self.code_benchmark.evaluate_tokenizer(tokenizer, name)

        # Store metrics
        score.indic_metrics = indic_results
        score.code_metrics = code_results

        # Compute scores (0-100 scale)
        # Indic: lower tokens/char, lower byte fallback, higher devanagari quality
        indic_agg = indic_results['aggregate']
        score.indic_score = (
            (1.0 / max(indic_agg.get('tokens_per_char', 1.0), 0.1)) * 30 +
            (1.0 - indic_agg.get('byte_fallback_rate', 0.0)) * 40 +
            indic_agg.get('devanagari_quality', 0.0) * 30
        )

        # Code: lower tokens/line, higher symbol preservation, higher chars/token
        code_agg = code_results['aggregate']
        score.code_score = (
            (1.0 / max(code_agg.get('tokens_per_line', 1.0), 0.1)) * 2 +
            code_agg.get('symbol_preservation', 0.0) * 30 +
            code_agg.get('structure_quality', 0.0) * 20
        )

        # Overall score (weighted average: Indic 50%, Code 50%)
        score.overall_score = (
            score.indic_score * 0.5 +
            score.code_score * 0.5
        )

        # Check filters
        score.passed_hard_filters = (
            indic_results['passed_filters'] and
            code_results['passed_filters']
        )

        score.rejection_reasons = (
            indic_results['rejection_reasons'] +
            code_results['rejection_reasons']
        )

        logger.info(f"Scores for '{name}':")
        logger.info(f"  Indic: {score.indic_score:.2f}")
        logger.info(f"  Code: {score.code_score:.2f}")
        logger.info(f"  Overall: {score.overall_score:.2f}")
        logger.info(f"  Passed filters: {score.passed_hard_filters}")

        return score

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks on all tokenizers and generate rankings."""
        logger.info("Starting HuggingFace-based tokenizer evaluation...")

        tokenizer_names = self.config['tokenizer_sources']['tokenizers']
        tokenizers = self.loader.load_all_tokenizers(tokenizer_names)

        scores = []
        for name, token_to_id in tokenizers.items():
            try:
                score = self.evaluate_single_tokenizer(name, token_to_id)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to evaluate tokenizer '{name}': {e}")
                import traceback
                traceback.print_exc()

        # Filter and rank
        passed_scores = [s for s in scores if s.passed_hard_filters]
        failed_scores = [s for s in scores if not s.passed_hard_filters]

        passed_scores.sort(key=lambda x: x.overall_score, reverse=True)
        failed_scores.sort(key=lambda x: x.overall_score, reverse=True)

        # Generate results
        results = {
            "evaluation_type": "huggingface_datasets",
            "top_ranked": [asdict(s) for s in passed_scores[:3]],
            "all_passed": [asdict(s) for s in passed_scores],
            "failed": [asdict(s) for s in failed_scores],
            "summary": {
                "total_evaluated": len(scores),
                "passed_filters": len(passed_scores),
                "failed_filters": len(failed_scores),
                "recommended": passed_scores[0].name if passed_scores else None
            }
        }

        # Save results
        output_path = Path(self.config['output']['evaluation_results_hf'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {output_path}")

        if passed_scores:
            logger.info(f"\nTop 3 Recommendations:")
            for i, score in enumerate(passed_scores[:3], 1):
                logger.info(f"{i}. {score.name} (score: {score.overall_score:.2f})")
        else:
            logger.warning("No tokenizers passed hard filters!")

        return results


def main():
    parser = argparse.ArgumentParser(description="Tokenizer Evaluation Suite (HuggingFace Datasets)")
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Setup logging
    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

    evaluator = TokenizerEvaluatorHF(args.config)
    results = evaluator.run_all_benchmarks()


if __name__ == "__main__":
    main()
