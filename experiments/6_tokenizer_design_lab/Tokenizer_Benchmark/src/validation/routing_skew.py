"""
Routing Skew Analyzer - Detect token distribution imbalances.

Validates that tokenizer doesn't create uneven token distributions
that could affect model routing mechanisms in MoE architectures.
"""

import math
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
        def vocab_size(self) -> int: ...
else:
    TokenizerProtocol = Any


@dataclass
class RoutingAnalysisResult:
    """Result of routing skew analysis."""
    entropy: float
    normalized_entropy: float  # 0-1, higher is more uniform
    skew_detected: bool
    top_tokens: List[Dict[str, Any]]
    issues: List[str]


class RoutingSkewAnalyzer:
    """
    Analyzes token distribution for potential routing skew.
    
    In MoE (Mixture of Experts) architectures, token IDs often influence
    which expert processes them. Uneven token distributions can cause
    load imbalance.
    """
    
    def __init__(self, entropy_threshold: float = 0.6):
        """
        Initialize analyzer.
        
        Args:
            entropy_threshold: Minimum normalized entropy for acceptable balance
        """
        self.entropy_threshold = entropy_threshold
    
    def analyze(
        self,
        tokenizer: TokenizerProtocol,
        texts: List[str],
        top_k: int = 20
    ) -> RoutingAnalysisResult:
        """
        Analyze token distribution across texts.
        
        Args:
            tokenizer: Tokenizer to analyze
            texts: List of texts to analyze
            top_k: Number of top tokens to report
        
        Returns:
            RoutingAnalysisResult
        """
        # Collect all tokens
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return RoutingAnalysisResult(
                entropy=0,
                normalized_entropy=0,
                skew_detected=True,
                top_tokens=[],
                issues=["No tokens to analyze"],
            )
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        unique_tokens = len(token_counts)
        
        # Calculate entropy
        entropy = 0
        for count in token_counts.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Maximum possible entropy (uniform distribution)
        max_entropy = math.log2(unique_tokens) if unique_tokens > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Get top tokens
        try:
            vocab_size = tokenizer.vocab_size()
        except:
            vocab_size = max(all_tokens) + 1
        
        top_tokens = []
        for token_id, count in token_counts.most_common(top_k):
            top_tokens.append({
                'token_id': token_id,
                'count': count,
                'percentage': count / total_tokens * 100,
            })
        
        # Check for skew
        issues = []
        skew_detected = normalized_entropy < self.entropy_threshold
        
        if skew_detected:
            issues.append(
                f"Low token distribution entropy ({normalized_entropy:.3f} < {self.entropy_threshold})"
            )
        
        # Check for dominant tokens
        if top_tokens and top_tokens[0]['percentage'] > 10:
            issues.append(
                f"Dominant token: ID {top_tokens[0]['token_id']} "
                f"accounts for {top_tokens[0]['percentage']:.1f}% of all tokens"
            )
        
        # Check concentration
        top_10_concentration = sum(t['percentage'] for t in top_tokens[:10])
        if top_10_concentration > 50:
            issues.append(
                f"High concentration: Top 10 tokens = {top_10_concentration:.1f}% of all tokens"
            )
        
        return RoutingAnalysisResult(
            entropy=entropy,
            normalized_entropy=normalized_entropy,
            skew_detected=skew_detected,
            top_tokens=top_tokens,
            issues=issues,
        )
    
    def compare_to_baseline(
        self,
        tokenizer: TokenizerProtocol,
        baseline_tokenizer: TokenizerProtocol,
        texts: List[str]
    ) -> Dict[str, Any]:
        """
        Compare token distribution to a baseline tokenizer.
        
        Args:
            tokenizer: Tokenizer to evaluate
            baseline_tokenizer: Baseline for comparison
            texts: Texts to analyze
        
        Returns:
            Comparison results
        """
        result = self.analyze(tokenizer, texts)
        baseline_result = self.analyze(baseline_tokenizer, texts)
        
        entropy_diff = result.normalized_entropy - baseline_result.normalized_entropy
        
        return {
            'tokenizer_entropy': result.normalized_entropy,
            'baseline_entropy': baseline_result.normalized_entropy,
            'entropy_difference': entropy_diff,
            'skew_amplified': entropy_diff < -0.1,  # Significantly worse than baseline
            'issues': result.issues,
        }


def calculate_routing_entropy(
    tokenizer: TokenizerProtocol,
    texts: List[str]
) -> float:
    """
    Calculate normalized entropy of token distribution.
    
    Args:
        tokenizer: Tokenizer to analyze
        texts: Texts to analyze
    
    Returns:
        Normalized entropy (0-1, higher is more uniform)
    """
    analyzer = RoutingSkewAnalyzer()
    result = analyzer.analyze(tokenizer, texts)
    return result.normalized_entropy


def detect_skew_amplification(
    tokenizer: TokenizerProtocol,
    baseline_tokenizer: TokenizerProtocol,
    texts: List[str],
    threshold: float = 0.1
) -> bool:
    """
    Detect if tokenizer amplifies routing skew compared to baseline.
    
    Args:
        tokenizer: Tokenizer to evaluate
        baseline_tokenizer: Reference tokenizer
        texts: Texts to analyze
        threshold: Difference threshold for detection
    
    Returns:
        True if skew is amplified (bad), False otherwise
    """
    analyzer = RoutingSkewAnalyzer()
    comparison = analyzer.compare_to_baseline(tokenizer, baseline_tokenizer, texts)
    return comparison['skew_amplified']
