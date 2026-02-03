"""
Curriculum Analyzer - Check tokenizer behavior across difficulty bands.

Validates that tokenizer doesn't artificially distort difficulty levels:
- Easy content shouldn't become disproportionately token-heavy
- Hard content shouldn't become disproportionately efficient
- Consistent behavior across difficulty bands
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import statistics
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
    
    class TokenizerProtocol(Protocol):
        def encode(self, text: str) -> List[int]: ...
else:
    TokenizerProtocol = Any


@dataclass
class DifficultyBandResult:
    """Result for a single difficulty band."""
    difficulty: str
    num_samples: int
    mean_tokens_per_byte: float
    std_tokens_per_byte: float
    mean_tokens_per_word: float
    total_tokens: int
    total_chars: int


@dataclass
class CurriculumAnalysisResult:
    """Complete curriculum analysis result."""
    passed: bool
    distortion_detected: bool
    distortion_score: float  # 0 = no distortion, 1 = severe distortion
    bands: Dict[str, DifficultyBandResult]
    issues: List[str]


class CurriculumAnalyzer:
    """
    Analyzes tokenizer behavior across difficulty bands.
    
    Ensures tokenizer doesn't create unfair advantages/disadvantages
    for certain difficulty levels.
    """
    
    def __init__(self, distortion_threshold: float = 0.25):
        """
        Initialize analyzer.
        
        Args:
            distortion_threshold: Maximum allowed variance ratio between bands
        """
        self.distortion_threshold = distortion_threshold
    
    def _compute_band_metrics(
        self,
        tokenizer: TokenizerProtocol,
        texts: List[str]
    ) -> DifficultyBandResult:
        """Compute metrics for a single difficulty band."""
        if not texts:
            return DifficultyBandResult(
                difficulty="unknown",
                num_samples=0,
                mean_tokens_per_byte=0,
                std_tokens_per_byte=0,
                mean_tokens_per_word=0,
                total_tokens=0,
                total_chars=0,
            )
        
        tpb_values = []
        tpw_values = []
        total_tokens = 0
        total_chars = 0
        
        for text in texts:
            tokens = tokenizer.encode(text)
            num_bytes = len(text.encode('utf-8'))
            words = text.split()
            
            total_tokens += len(tokens)
            total_chars += len(text)
            
            if num_bytes > 0:
                tpb_values.append(len(tokens) / num_bytes)
            if words:
                tpw_values.append(len(tokens) / len(words))
        
        return DifficultyBandResult(
            difficulty="",  # Set by caller
            num_samples=len(texts),
            mean_tokens_per_byte=statistics.mean(tpb_values) if tpb_values else 0,
            std_tokens_per_byte=statistics.stdev(tpb_values) if len(tpb_values) > 1 else 0,
            mean_tokens_per_word=statistics.mean(tpw_values) if tpw_values else 0,
            total_tokens=total_tokens,
            total_chars=total_chars,
        )
    
    def analyze(
        self,
        tokenizer: TokenizerProtocol,
        difficulty_bands: Dict[str, List[str]]
    ) -> CurriculumAnalysisResult:
        """
        Analyze tokenizer behavior across difficulty bands.
        
        Args:
            tokenizer: Tokenizer to analyze
            difficulty_bands: Dict mapping difficulty level to list of texts
        
        Returns:
            CurriculumAnalysisResult
        """
        bands = {}
        tpb_means = []
        
        for difficulty, texts in difficulty_bands.items():
            result = self._compute_band_metrics(tokenizer, texts)
            result.difficulty = difficulty
            bands[difficulty] = result
            if result.mean_tokens_per_byte > 0:
                tpb_means.append(result.mean_tokens_per_byte)
        
        # Check for distortion
        issues = []
        distortion_score = 0.0
        
        if len(tpb_means) >= 2:
            # Calculate coefficient of variation
            mean_of_means = statistics.mean(tpb_means)
            std_of_means = statistics.stdev(tpb_means) if len(tpb_means) > 1 else 0
            cv = std_of_means / mean_of_means if mean_of_means > 0 else 0
            
            distortion_score = min(1.0, cv / self.distortion_threshold)
            
            if cv > self.distortion_threshold:
                issues.append(
                    f"High variance in tokens/byte across difficulty bands (CV={cv:.3f})"
                )
            
            # Check for monotonic distortion (harder = more/fewer tokens)
            sorted_bands = sorted(
                bands.items(),
                key=lambda x: ['easy', 'medium', 'hard'].index(x[0])
                if x[0] in ['easy', 'medium', 'hard'] else 999
            )
            
            if len(sorted_bands) >= 3:
                tpb_sequence = [b[1].mean_tokens_per_byte for b in sorted_bands]
                
                # Check if monotonically increasing or decreasing
                increasing = all(tpb_sequence[i] <= tpb_sequence[i+1] 
                               for i in range(len(tpb_sequence)-1))
                decreasing = all(tpb_sequence[i] >= tpb_sequence[i+1] 
                               for i in range(len(tpb_sequence)-1))
                
                if increasing and tpb_sequence[-1] / max(tpb_sequence[0], 0.001) > 1.3:
                    issues.append(
                        "Tokens/byte increases with difficulty (potential positive distortion)"
                    )
                elif decreasing and tpb_sequence[0] / max(tpb_sequence[-1], 0.001) > 1.3:
                    issues.append(
                        "Tokens/byte decreases with difficulty (potential negative distortion)"
                    )
        
        distortion_detected = len(issues) > 0
        passed = not distortion_detected
        
        return CurriculumAnalysisResult(
            passed=passed,
            distortion_detected=distortion_detected,
            distortion_score=distortion_score,
            bands=bands,
            issues=issues,
        )


def analyze_difficulty_bands(
    tokenizer: TokenizerProtocol,
    easy_texts: List[str],
    medium_texts: List[str],
    hard_texts: List[str]
) -> CurriculumAnalysisResult:
    """
    Quick function to analyze three difficulty bands.
    
    Args:
        tokenizer: Tokenizer to analyze
        easy_texts: Easy difficulty texts
        medium_texts: Medium difficulty texts
        hard_texts: Hard difficulty texts
    
    Returns:
        CurriculumAnalysisResult
    """
    analyzer = CurriculumAnalyzer()
    return analyzer.analyze(
        tokenizer,
        {'easy': easy_texts, 'medium': medium_texts, 'hard': hard_texts}
    )


def check_difficulty_distortion(result: CurriculumAnalysisResult) -> bool:
    """
    Check if difficulty distortion was detected.
    
    Args:
        result: CurriculumAnalysisResult from analyze_difficulty_bands
    
    Returns:
        True if distortion detected (bad), False if clean (good)
    """
    return result.distortion_detected
