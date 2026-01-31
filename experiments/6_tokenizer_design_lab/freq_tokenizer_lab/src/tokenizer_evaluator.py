"""
Tokenizer Evaluator - Comprehensive benchmark suite for tokenizer selection.

Evaluates tokenizers on:
- Indic language quality (Devanagari, fragmentation, byte-fallback)
- Code efficiency (Python, JS, C++, symbol handling)
- JSON and structured data handling

Usage:
    python tokenizer_evaluator.py --config ../config.yaml
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


class IndicBenchmark:
    """Benchmark suite for Indic language quality."""

    # Sample texts for different Indic languages (Devanagari-heavy)
    SAMPLE_TEXTS = {
        "hindi": "भारत एक विशाल देश है। यहाँ की संस्कृति और परंपराएँ बहुत समृद्ध हैं। हिंदी भारत की राजभाषा है।",
        "sanskrit": "संस्कृतं भारतस्य प्राचीनतमा भाषा अस्ति। वेदाः पुराणानि च संस्कृतभाषायां लिखिताः सन्ति।",
        "marathi": "महाराष्ट्र हा भारताचा एक महत्त्वाचा राज्य आहे। मराठी ही महाराष्ट्राची राजभाषा आहे।",
        "nepali": "नेपाल हिमालयको काखमा बसेको एक सुन्दर देश हो। नेपाली भाषा देवनागरी लिपिमा लेखिन्छ।",
        "bengali": "বাংলা ভারত এবং বাংলাদেশের একটি প্রধান ভাষা। বাংলা সাহিত্য অত্যন্ত সমৃদ্ধ।",
        "tamil": "தமிழ் உலகின் பழமையான மொழிகளில் ஒன்றாகும். தமிழ் இலக்கியம் மிகவும் வளமானது.",
        "telugu": "తెలుగు భారతదేశంలో అత్యంత ముఖ్యమైన భాషలలో ఒకటి। తెలుగు సాహిత్యం చాలా గొప్పది.",
        "gujarati": "ગુજરાત ભારતનું એક સમૃદ્ધ રાજ્ય છે। ગુજરાતી ભાષા અને સંસ્કૃતિ ખૂબ જ સુંદર છે।"
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
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
        """Run full Indic benchmark on tokenizer."""
        logger.info(f"Running Indic benchmark on '{tokenizer_name}'")

        metrics_by_lang = {}
        aggregate_metrics = defaultdict(list)

        for lang, text in self.SAMPLE_TEXTS.items():
            ids = tokenizer.encode(text)

            # Compute metrics
            tokens_per_char = self.compute_tokens_per_char(text, ids)
            byte_fallback_rate = self.compute_byte_fallback_rate(text, ids, tokenizer)
            fragmentation_score = self.compute_fragmentation_score(text, ids)
            devanagari_quality = self.compute_devanagari_quality(text, ids, tokenizer)

            metrics_by_lang[lang] = {
                "tokens_per_char": tokens_per_char,
                "byte_fallback_rate": byte_fallback_rate,
                "fragmentation_score": fragmentation_score,
                "devanagari_quality": devanagari_quality,
                "num_tokens": len(ids),
                "num_chars": len(text)
            }

            # Aggregate
            aggregate_metrics["tokens_per_char"].append(tokens_per_char)
            aggregate_metrics["byte_fallback_rate"].append(byte_fallback_rate)
            aggregate_metrics["fragmentation_score"].append(fragmentation_score)
            aggregate_metrics["devanagari_quality"].append(devanagari_quality)

        # Average across languages
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in aggregate_metrics.items()
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

        return {
            "by_language": metrics_by_lang,
            "aggregate": avg_metrics,
            "passed_filters": passed,
            "rejection_reasons": reasons
        }


class CodeBenchmark:
    """Benchmark suite for code tokenization quality."""

    # Sample code in different languages
    SAMPLE_CODE = {
        "python": '''
def calculate_fibonacci(n: int) -> list[int]:
    """Calculate Fibonacci sequence up to n terms."""
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

result = calculate_fibonacci(10)
print(f"Fibonacci: {result}")
''',
        "javascript": '''
function calculateFactorial(n) {
  if (n <= 1) return 1;
  return n * calculateFactorial(n - 1);
}

const numbers = [1, 2, 3, 4, 5];
const factorials = numbers.map(n => calculateFactorial(n));
console.log(`Factorials: ${factorials}`);
''',
        "typescript": '''
interface User {
  id: number;
  name: string;
  email?: string;
}

class UserManager {
  private users: Map<number, User> = new Map();

  addUser(user: User): void {
    this.users.set(user.id, user);
  }

  getUser(id: number): User | undefined {
    return this.users.get(id);
  }
}
''',
        "c": '''
#include <stdio.h>
#include <stdlib.h>

int* create_array(int size) {
    int* arr = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        arr[i] = i * i;
    }
    return arr;
}

int main() {
    int* numbers = create_array(10);
    free(numbers);
    return 0;
}
''',
        "cpp": '''
#include <vector>
#include <algorithm>
#include <iostream>

template<typename T>
class Container {
private:
    std::vector<T> data;

public:
    void add(const T& item) {
        data.push_back(item);
    }

    void sort() {
        std::sort(data.begin(), data.end());
    }
};
'''
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config['evaluation']['code']['failure_thresholds']

    def compute_tokens_per_line(self, code: str, ids: List[int]) -> float:
        """Compute average tokens per line of code."""
        lines = [line for line in code.split('\n') if line.strip()]
        return len(ids) / len(lines) if len(lines) > 0 else float('inf')

    def compute_symbol_preservation(self, code: str, ids: List[int], tokenizer: SimpleTokenizer) -> float:
        """Check if important symbols are preserved (braces, operators, etc.)."""
        important_symbols = ['{', '}', '(', ')', '[', ']', '=', ':', ';', ',', '.', '->', '::']
        decoded = tokenizer.decode(ids)

        preserved_count = sum(1 for sym in important_symbols if code.count(sym) == decoded.count(sym))
        return preserved_count / len(important_symbols)

    def compute_structure_quality(self, code: str, ids: List[int]) -> float:
        """Measure how well code structure is tokenized (indentation, spacing)."""
        # Simple heuristic: ratio of original length to token count
        return len(code) / len(ids) if len(ids) > 0 else 0.0

    def evaluate_tokenizer(self, tokenizer: SimpleTokenizer, tokenizer_name: str) -> Dict[str, Any]:
        """Run full code benchmark on tokenizer."""
        logger.info(f"Running code benchmark on '{tokenizer_name}'")

        metrics_by_lang = {}
        aggregate_metrics = defaultdict(list)

        for lang, code in self.SAMPLE_CODE.items():
            ids = tokenizer.encode(code)

            # Compute metrics
            tokens_per_line = self.compute_tokens_per_line(code, ids)
            symbol_preservation = self.compute_symbol_preservation(code, ids, tokenizer)
            structure_quality = self.compute_structure_quality(code, ids)

            metrics_by_lang[lang] = {
                "tokens_per_line": tokens_per_line,
                "symbol_preservation": symbol_preservation,
                "structure_quality": structure_quality,
                "num_tokens": len(ids),
                "num_chars": len(code)
            }

            # Aggregate
            aggregate_metrics["tokens_per_line"].append(tokens_per_line)
            aggregate_metrics["symbol_preservation"].append(symbol_preservation)
            aggregate_metrics["structure_quality"].append(structure_quality)

        # Average across languages
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in aggregate_metrics.items()
        }

        # Check hard filters
        passed = True
        reasons = []

        if avg_metrics["tokens_per_line"] > self.thresholds["tokens_per_line"]:
            passed = False
            reasons.append(f"Excessive tokens per line: {avg_metrics['tokens_per_line']:.2f}")

        return {
            "by_language": metrics_by_lang,
            "aggregate": avg_metrics,
            "passed_filters": passed,
            "rejection_reasons": reasons
        }


class JSONBenchmark:
    """Benchmark suite for JSON and tool-calling tokenization."""

    SAMPLE_JSON = {
        "simple": '{"name": "Alice", "age": 30, "city": "Mumbai"}',
        "nested": '{"user": {"id": 123, "profile": {"name": "Bob", "skills": ["Python", "JavaScript"]}}}',
        "array": '[1, 2, 3, 4, 5, 10, 100, 1000]',
        "unicode": '{"message": "हेलो वर्ल्ड", "emoji": "🚀"}',
        "tool_schema": '''{
  "name": "calculate_sum",
  "description": "Calculate sum of numbers",
  "parameters": {
    "type": "object",
    "properties": {
      "numbers": {"type": "array", "items": {"type": "number"}}
    }
  }
}'''
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def compute_tokens_per_byte(self, json_str: str, ids: List[int]) -> float:
        """Compute token efficiency for JSON."""
        return len(ids) / len(json_str.encode('utf-8')) if json_str else float('inf')

    def compute_structure_preservation(self, json_str: str, ids: List[int], tokenizer: SimpleTokenizer) -> float:
        """Check if JSON structure is preserved in tokenization."""
        decoded = tokenizer.decode(ids)
        # Check if key structural elements match
        structural_chars = ['{', '}', '[', ']', ':', ',', '"']
        preserved = sum(1 for char in structural_chars if json_str.count(char) == decoded.count(char))
        return preserved / len(structural_chars)

    def evaluate_tokenizer(self, tokenizer: SimpleTokenizer, tokenizer_name: str) -> Dict[str, Any]:
        """Run full JSON benchmark on tokenizer."""
        logger.info(f"Running JSON benchmark on '{tokenizer_name}'")

        metrics_by_case = {}
        aggregate_metrics = defaultdict(list)

        for case_name, json_str in self.SAMPLE_JSON.items():
            ids = tokenizer.encode(json_str)

            # Compute metrics
            tokens_per_byte = self.compute_tokens_per_byte(json_str, ids)
            structure_preservation = self.compute_structure_preservation(json_str, ids, tokenizer)

            metrics_by_case[case_name] = {
                "tokens_per_byte": tokens_per_byte,
                "structure_preservation": structure_preservation,
                "num_tokens": len(ids),
                "num_bytes": len(json_str.encode('utf-8'))
            }

            # Aggregate
            aggregate_metrics["tokens_per_byte"].append(tokens_per_byte)
            aggregate_metrics["structure_preservation"].append(structure_preservation)

        # Average across cases
        avg_metrics = {
            key: sum(values) / len(values) if values else 0.0
            for key, values in aggregate_metrics.items()
        }

        return {
            "by_case": metrics_by_case,
            "aggregate": avg_metrics,
            "passed_filters": True,
            "rejection_reasons": []
        }


class TokenizerEvaluator:
    """Main evaluator orchestrating all benchmarks."""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.loader = TokenizerLoader(self.config['tokenizer_sources']['base_path'])
        self.indic_benchmark = IndicBenchmark(self.config)
        self.code_benchmark = CodeBenchmark(self.config)
        self.json_benchmark = JSONBenchmark(self.config)

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
        json_results = self.json_benchmark.evaluate_tokenizer(tokenizer, name)

        # Store metrics
        score.indic_metrics = indic_results
        score.code_metrics = code_results
        score.json_metrics = json_results

        # Compute scores (0-100 scale)
        # Indic: lower tokens/char, lower byte fallback, higher devanagari quality
        indic_agg = indic_results['aggregate']
        score.indic_score = (
            (1.0 / max(indic_agg.get('tokens_per_char', 1.0), 0.1)) * 30 +
            (1.0 - indic_agg.get('byte_fallback_rate', 0.0)) * 40 +
            indic_agg.get('devanagari_quality', 0.0) * 30
        )

        # Code: lower tokens/line, higher symbol preservation
        code_agg = code_results['aggregate']
        score.code_score = (
            (1.0 / max(code_agg.get('tokens_per_line', 1.0), 0.1)) * 2 +
            code_agg.get('symbol_preservation', 0.0) * 50
        )

        # JSON: lower tokens/byte, higher structure preservation
        json_agg = json_results['aggregate']
        score.json_score = (
            (1.0 / max(json_agg.get('tokens_per_byte', 1.0), 0.1)) * 30 +
            json_agg.get('structure_preservation', 0.0) * 50
        )

        # Overall score (weighted average)
        score.overall_score = (
            score.indic_score * 0.4 +
            score.code_score * 0.4 +
            score.json_score * 0.2
        )

        # Check filters
        score.passed_hard_filters = (
            indic_results['passed_filters'] and
            code_results['passed_filters'] and
            json_results['passed_filters']
        )

        score.rejection_reasons = (
            indic_results['rejection_reasons'] +
            code_results['rejection_reasons'] +
            json_results['rejection_reasons']
        )

        logger.info(f"Scores for '{name}':")
        logger.info(f"  Indic: {score.indic_score:.2f}")
        logger.info(f"  Code: {score.code_score:.2f}")
        logger.info(f"  JSON: {score.json_score:.2f}")
        logger.info(f"  Overall: {score.overall_score:.2f}")
        logger.info(f"  Passed filters: {score.passed_hard_filters}")

        return score

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run benchmarks on all tokenizers and generate rankings."""
        logger.info("Starting tokenizer evaluation...")

        tokenizer_names = self.config['tokenizer_sources']['tokenizers']
        tokenizers = self.loader.load_all_tokenizers(tokenizer_names)

        scores = []
        for name, token_to_id in tokenizers.items():
            score = self.evaluate_single_tokenizer(name, token_to_id)
            scores.append(score)

        # Filter and rank
        passed_scores = [s for s in scores if s.passed_hard_filters]
        failed_scores = [s for s in scores if not s.passed_hard_filters]

        passed_scores.sort(key=lambda x: x.overall_score, reverse=True)
        failed_scores.sort(key=lambda x: x.overall_score, reverse=True)

        # Generate results
        results = {
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
        output_path = Path(self.config['output']['evaluation_results'])
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
    parser = argparse.ArgumentParser(description="Tokenizer Evaluation Suite")
    parser.add_argument('--config', type=str, default='../config.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    evaluator = TokenizerEvaluator(args.config)
    results = evaluator.run_all_benchmarks()


if __name__ == "__main__":
    main()
