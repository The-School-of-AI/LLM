# Curriculum Metrics

Built-in metrics for curriculum metadata extraction.

## Overview

These metrics compute various text features for curriculum-based LLM training. Each metric:

- Inherits from `MetricPlugin`
- Receives a **read-only record** (never modifies input)
- Returns an `ExtractionResult` with computed metrics
- Can optionally reject records that don't meet quality criteria

## Available Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| `DifficultyMetric` | 0 | Text difficulty scoring based on linguistic features |
| `ReadabilityMetric` | 0 | Readability scores (Flesch, SMOG, etc.) |
| `ModalityMetric` | 1 | Detect code, math, reasoning, agentic patterns |
| `EntropyMetric` | 1 | Character and token entropy |
| `DiversityMetric` | 1 | Lexical diversity measures |
| `StructuralDensityMetric` | 1 | Text structure analysis |
| `TokenizerDifficultyMetric` | 2 | Tokenizer-specific difficulty |

**Note**: `BandAssignmentMetric` is excluded from extraction. Band assignment happens in post-processing via the `assign_bands.py` script.

## Metric Levels

Metrics are grouped by "level" for ordered execution:

- **Level 0**: Core metrics (difficulty, readability) - run first
- **Level 1**: Content analysis (modality, entropy) - run after level 0
- **Level 2**: Advanced metrics (tokenizer) - run last

Metrics at the same level could potentially run in parallel (future enhancement).

## Configuration

Create `metrics_config.yaml`:

```yaml
metrics:
  # Level 0 - Core metrics
  - class: DifficultyMetric
    enabled: true
    level: 0
    
  - class: ReadabilityMetric
    enabled: true
    level: 0
    
  # Level 1 - Content analysis
  - class: ModalityMetric
    enabled: true
    level: 1
    
  - class: EntropyMetric
    enabled: true
    level: 1
    
  - class: DiversityMetric
    enabled: true
    level: 1
    
  - class: StructuralDensityMetric
    enabled: true
    level: 1
    
  # Level 2 - Advanced
  - class: TokenizerDifficultyMetric
    enabled: true
    level: 2
```

## Metric Details

### DifficultyMetric

Computes text difficulty based on:
- Average word length
- Rare word ratio (hapax legomena)
- Character entropy

**Output columns:**
- `difficulty_level` (L0-L5)
- `difficulty_score` (0.0-1.0)
- `difficulty_features_*` (component scores)

### ReadabilityMetric

Standard readability formulas:

**Output columns:**
- `readability_flesch` - Flesch Reading Ease
- `readability_flesch_kincaid` - Flesch-Kincaid Grade
- `readability_smog` - SMOG Index
- `readability_score` - Normalized composite

### ModalityMetric

Detects content types:

**Output columns:**
- `modality_has_code` (bool)
- `modality_has_math` (bool)
- `modality_has_reasoning` (bool) - Chain-of-thought patterns
- `modality_has_agentic` (bool) - Tool/agent traces
- `modality_primary` - Dominant modality

### EntropyMetric

Information theory metrics:

**Output columns:**
- `entropy_char` - Character-level entropy
- `entropy_word` - Word-level entropy
- `entropy_ratio` - Normalized entropy

### DiversityMetric

Lexical diversity measures:

**Output columns:**
- `diversity_ttr` - Type-Token Ratio
- `diversity_hapax` - Hapax Legomena ratio
- `diversity_score` - Composite score

### StructuralDensityMetric

Text structure analysis:

**Output columns:**
- `structural_paragraph_count`
- `structural_sentence_count`
- `structural_avg_sentence_length`
- `structural_density_score`

### TokenizerDifficultyMetric

Tokenizer-specific metrics:

**Output columns:**
- `tokenizer_tokens_per_word`
- `tokenizer_unk_ratio`
- `tokenizer_difficulty_score`

## Creating Custom Metrics

```python
from curriculum_extractor.core.plugin import (
    MetricPlugin,
    ExtractionResult,
    ReadOnlyRecord,
)

class CustomMetric(MetricPlugin):
    """Custom metric example."""
    
    name = "custom"
    level = 1  # Execution level
    
    def compute(self, record: ReadOnlyRecord) -> dict:
        """Compute metric values.
        
        Args:
            record: Read-only record wrapper. Use .get() for field access.
            
        Returns:
            Dictionary of metric values (will be flattened with prefix)
        """
        text = record.get("text", "")
        
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "features": {
                "has_numbers": any(c.isdigit() for c in text),
            }
        }
        # Output: custom_length, custom_word_count, custom_features_has_numbers
    
    def extract(self, record: ReadOnlyRecord) -> ExtractionResult:
        """Extract with optional rejection.
        
        Override this for rejection logic. Default just calls compute().
        """
        text = record.get("text", "")
        
        # Rejection example
        if len(text) < 10:
            return ExtractionResult(
                metrics={},
                rejected=True,
                rejection_reason="Text too short",
            )
        
        return ExtractionResult(metrics=self.compute(record))
```

### Register Custom Metric

Add to `metrics_config.yaml`:

```yaml
metrics:
  - class: CustomMetric
    module: my_package.metrics.custom  # Full module path
    enabled: true
    level: 1
```

Or pass directly:

```python
from curriculum_extractor import CurriculumExtractor
from my_metrics import CustomMetric

extractor = CurriculumExtractor(
    "curriculum.yaml",
    metrics=[
        CustomMetric(config),
        # ... other metrics
    ]
)
```

## Important Notes

1. **Records are immutable**: Never modify the input record
2. **Use `.get()` for field access**: ReadOnlyRecord supports dict-like access
3. **Return flat or nested dicts**: Nested dicts are auto-flattened with underscores
4. **Rejection stops the pipeline**: If your metric rejects, no further metrics run
5. **Errors are logged, not fatal**: If compute() raises, an error column is added

## Column Naming

All metric outputs are prefixed with the metric name:

```python
class MyMetric(MetricPlugin):
    name = "mymetric"  # prefix
    
    def compute(self, record):
        return {
            "score": 0.5,           # → mymetric_score
            "features": {
                "a": 1,             # → mymetric_features_a
                "b": 2,             # → mymetric_features_b
            }
        }
```

To customize the prefix:

```python
class MyMetric(MetricPlugin):
    name = "mymetric"
    column_prefix = "mm"  # Custom prefix
    
    # Output: mm_score, mm_features_a, mm_features_b
```
