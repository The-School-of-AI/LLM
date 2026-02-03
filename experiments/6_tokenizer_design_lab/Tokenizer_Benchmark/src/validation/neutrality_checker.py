"""
Neutrality Checker - Detect benchmark content mirroring.

Validates that:
1. Tokenizer doesn't have artificially good performance on benchmark patterns
2. Probes contain only format/structure, not real benchmark content
3. Tokenizer merges are not tuned on specific benchmark data
"""

import re
import hashlib
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
else:
    TokenizerProtocol = Any


@dataclass
class ValidationResult:
    """Result of a neutrality validation check."""
    passed: bool
    score: float  # 0-1, higher is more neutral
    issues: List[str]
    details: Dict[str, Any]


# Known benchmark signatures to check against
# These are patterns that might indicate benchmark leakage
BENCHMARK_SIGNATURES = {
    # Common benchmark question patterns (format only, no content)
    'gsm8k_pattern': r'(?i)^Q:\s*\w+.*\?$',
    'mmlu_pattern': r'(?i)^Question:\s.*\n\s*A\)\s.*\n\s*B\)\s.*',
    'hellaswag_pattern': r'(?i)^Context:\s.*\nA\.\s.*\nB\.\s.*',
    'humaneval_pattern': r'^def\s+\w+\(.*\):\s*\n\s+""".*"""',
}

# Suspicious vocabulary patterns that might indicate benchmark tuning
SUSPICIOUS_PATTERNS = [
    # Overly specific numerical patterns
    r'\b\d{6,}\b',  # Very long numbers that might be benchmark-specific
    # Benchmark-specific terminology that shouldn't appear in probes
    r'(?i)\b(gsm8k|mmlu|hellaswag|arc|winogrande|truthfulqa)\b',
    r'(?i)\b(benchmark|eval|test_set|validation_set)\b',
]


class NeutralityChecker:
    """
    Checks tokenizer and probes for benchmark neutrality violations.
    """
    
    def __init__(self, sensitivity: str = "high"):
        """
        Initialize checker.
        
        Args:
            sensitivity: 'low', 'medium', or 'high'
        """
        self.sensitivity = sensitivity
        self.thresholds = {
            'low': {'variance_threshold': 0.5, 'signature_threshold': 10},
            'medium': {'variance_threshold': 0.3, 'signature_threshold': 5},
            'high': {'variance_threshold': 0.15, 'signature_threshold': 2},
        }[sensitivity]
    
    def check_probe_neutrality(self, probes: List[str]) -> ValidationResult:
        """
        Verify that probes don't contain real benchmark content.
        
        Args:
            probes: List of probe texts
        
        Returns:
            ValidationResult with neutrality assessment
        """
        issues = []
        suspicious_count = 0
        
        for i, probe in enumerate(probes):
            # Check for benchmark signatures
            for name, pattern in BENCHMARK_SIGNATURES.items():
                if re.search(pattern, probe):
                    suspicious_count += 1
                    if suspicious_count <= self.thresholds['signature_threshold']:
                        issues.append(f"Probe {i}: Matches '{name}' pattern")
            
            # Check for suspicious patterns
            for pattern in SUSPICIOUS_PATTERNS:
                if re.search(pattern, probe):
                    suspicious_count += 1
                    issues.append(f"Probe {i}: Contains suspicious pattern")
        
        # Calculate score
        total_probes = len(probes)
        score = 1.0 - (suspicious_count / max(total_probes, 1))
        passed = suspicious_count < self.thresholds['signature_threshold']
        
        return ValidationResult(
            passed=passed,
            score=max(0, score),
            issues=issues[:10],  # Limit to first 10 issues
            details={
                'total_probes': total_probes,
                'suspicious_count': suspicious_count,
                'sensitivity': self.sensitivity,
            }
        )
    
    def check_tokenizer_suspicion(
        self,
        tokenizer: TokenizerProtocol,
        known_good_texts: List[str],
        probe_texts: List[str]
    ) -> ValidationResult:
        """
        Check if tokenizer shows suspiciously good performance on probes
        compared to baseline texts.
        
        A tokenizer tuned on benchmark data might show unusually low
        tokens-per-byte on benchmark-like content.
        
        Args:
            tokenizer: Tokenizer to check
            known_good_texts: Baseline texts (known neutral)
            probe_texts: Probes to check
        
        Returns:
            ValidationResult
        """
        def compute_tpb(texts: List[str]) -> List[float]:
            """Compute tokens per byte for each text."""
            results = []
            for text in texts:
                tokens = tokenizer.encode(text)
                num_bytes = len(text.encode('utf-8'))
                if num_bytes > 0:
                    results.append(len(tokens) / num_bytes)
            return results
        
        baseline_tpb = compute_tpb(known_good_texts)
        probe_tpb = compute_tpb(probe_texts)
        
        if not baseline_tpb or not probe_tpb:
            return ValidationResult(
                passed=True,
                score=1.0,
                issues=["Insufficient data for comparison"],
                details={}
            )
        
        # Compare distributions
        baseline_mean = sum(baseline_tpb) / len(baseline_tpb)
        probe_mean = sum(probe_tpb) / len(probe_tpb)
        
        # If probe performance is much better than baseline, it's suspicious
        improvement = (baseline_mean - probe_mean) / baseline_mean if baseline_mean > 0 else 0
        
        issues = []
        if improvement > self.thresholds['variance_threshold']:
            issues.append(
                f"Suspicious: Probe TPB ({probe_mean:.4f}) is {improvement*100:.1f}% "
                f"better than baseline ({baseline_mean:.4f})"
            )
        
        passed = improvement <= self.thresholds['variance_threshold']
        score = max(0, 1.0 - improvement)
        
        return ValidationResult(
            passed=passed,
            score=score,
            issues=issues,
            details={
                'baseline_mean_tpb': baseline_mean,
                'probe_mean_tpb': probe_mean,
                'improvement_ratio': improvement,
                'threshold': self.thresholds['variance_threshold'],
            }
        )


def detect_benchmark_mirroring(
    tokenizer: TokenizerProtocol,
    benchmark_signatures: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Detect if tokenizer vocabulary mirrors benchmark content.
    
    Args:
        tokenizer: Tokenizer to analyze
        benchmark_signatures: Optional custom signatures to check
    
    Returns:
        Mirroring detection results
    """
    signatures = benchmark_signatures or BENCHMARK_SIGNATURES
    
    # Test tokenization of signature patterns
    results = {}
    for name, pattern in signatures.items():
        # Create a simple test case matching the pattern
        test_text = pattern.replace(r'(?i)', '').replace(r'\s*', ' ')
        test_text = re.sub(r'\\[wd\+\*\.\?\(\)\[\]]', 'x', test_text)
        
        tokens = tokenizer.encode(test_text)
        results[name] = {
            'test_text_length': len(test_text),
            'num_tokens': len(tokens),
            'tokens_per_char': len(tokens) / max(len(test_text), 1),
        }
    
    return results


def check_format_only_compliance(probes: List[str]) -> ValidationResult:
    """
    Verify probes contain only format/structure, not real content.
    
    Args:
        probes: List of probe texts
    
    Returns:
        ValidationResult
    """
    checker = NeutralityChecker(sensitivity="high")
    return checker.check_probe_neutrality(probes)
