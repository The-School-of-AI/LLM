# Adding New Metrics

This guide shows you how to add a new curriculum metric to the tagging system.

## Quick Start

**2 simple steps:**

1. Create your metric class file
2. Add entry to `metrics_config.yaml`

That's it! The system auto-discovers your metric from both the file system and config.

**Auto-discovery works at two levels:**
- **Import-level**: Package automatically imports all metrics from `.py` files in this directory
- **Runtime-level**: `CurriculumTagger` loads enabled metrics from `metrics_config.yaml`

## Step 1: Create Your Metric File

Create a new Python file in this directory: `your_metric_name.py`

```python
"""Your metric description."""

from curriculum_tags.core.plugin import MetricPlugin


class YourMetric(MetricPlugin):
    """
    Brief description of what this metric computes.
    
    Tags added:
        your_metric:
            field1: description
            field2: description
    """
    
    name = "your_metric"  # Must match config entry
    
    def compute(self, sample: dict) -> dict:
        """
        Compute your metric.
        
        Args:
            sample: Dict with keys:
                - text: str - Document text
                - curriculum_tags: dict - Tags from previous metrics
                - Any other fields from your dataset
        
        Returns:
            Dict with your metric results, e.g.:
            {
                "field1": value1,
                "field2": value2,
            }
        """
        text = sample.get("text", "")
        
        # Access curriculum config thresholds
        threshold = self.config.get("your_metric.threshold", 0.5)
        
        # See results from previous metrics (if chaining needed)
        previous_tags = sample.get("curriculum_tags", {})
        difficulty_band = previous_tags.get("difficulty", {}).get("band")
        
        # Your computation here
        score = len(text) / 1000  # Example
        
        return {
            "score": score,
            "category": "high" if score > threshold else "low",
        }
```

## Step 2: Add to Configuration

Edit `metrics_config.yaml` (in project root):

```yaml
metrics:
  # ... existing metrics ...
  
  - name: your_metric
    class: YourMetric
    enabled: true
```

**Auto-discovery magic:**
- ✅ Your metric is automatically imported when someone does `from curriculum_tags import YourMetric`
- ✅ The tagger loads it at runtime based on the config
- ✅ No need to edit `__init__.py` - it scans the directory!

**Naming convention:**
- File name: `your_metric_name.py` (lowercase, underscores)
- Class name: `YourMetric` (PascalCase, ends with "Metric")
- Config name: Must match the `name` attribute in your class

## Step 3: Verify It Works

```bash
# Test the import
uv run python -c 'from curriculum_tags import YourMetric; print(YourMetric.name)'

# Add your metric's test file
tests/test_your_metric.py

# Run tests
uv run pytest tests/test_your_metric.py -v
```

## Testing Template

```python
"""Tests for your metric."""

import tempfile
from pathlib import Path

import pytest
import yaml

from curriculum_tags.metrics.your_metric_name import YourMetric
from curriculum_tags.utils.curriculum_loader import CurriculumConfig


@pytest.fixture
def config():
    """Create test config."""
    config_data = {
        "version": "0.1",
        "your_metric": {
            "threshold": 0.5,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        path = Path(f.name)
    
    cfg = CurriculumConfig(str(path))
    yield cfg
    path.unlink()


def test_your_metric_basic(config):
    """Test basic functionality."""
    metric = YourMetric(config)
    
    sample = {"text": "Test text here"}
    result = metric.compute(sample)
    
    assert "score" in result
    assert "category" in result


def test_your_metric_chaining(config):
    """Test with previous tags."""
    metric = YourMetric(config)
    
    sample = {
        "text": "Test",
        "curriculum_tags": {
            "difficulty": {"band": "B3", "score": 0.65},
        },
    }
    result = metric.compute(sample)
    
    assert result is not None
```

## Accessing Curriculum Config

Your metric automatically gets `self.config` (CurriculumConfig instance):

```python
# Simple key
value = self.config.get("my_key", default_value)

# Nested with dots
band_name = self.config.get("difficulty_system.bands.B0.name")

# Check existence
if self.config.has("optional_feature.enabled"):
    # ...
```

## Metric Chaining

Metrics run in the order defined in `metrics_config.yaml`. Each metric sees accumulated tags:

```python
def compute(self, sample: dict) -> dict:
    # Get tags from all previous metrics
    previous_tags = sample.get("curriculum_tags", {})
    
    # Example: Use difficulty score in your calculation
    difficulty = previous_tags.get("difficulty", {})
    diff_score = difficulty.get("score", 0.5)
    
    # Your logic can depend on earlier metrics
    adjusted_score = your_score * diff_score
    
    return {"adjusted": adjusted_score}
```

## Best Practices

**Do:**
- ✅ Use descriptive field names in your output
- ✅ Add docstrings explaining what you compute
- ✅ Store thresholds in `curriculum.yaml`, not hardcoded
- ✅ Handle missing fields gracefully
- ✅ Write tests for edge cases
- ✅ Return consistent dict structure

**Don't:**
- ❌ Modify the input sample dict
- ❌ Raise exceptions for normal cases (empty text, etc.)
- ❌ Depend on specific order unless documented
- ❌ Store state in the metric instance between samples
- ❌ Return None (return empty dict instead)

## Examples

### Simple Independent Metric

```python
class TokenCountMetric(MetricPlugin):
    """Count tokens in text."""
    
    name = "token_count"
    
    def compute(self, sample: dict) -> dict:
        text = sample.get("text", "")
        tokens = text.split()  # Simplistic
        
        return {
            "count": len(tokens),
            "avg_token_length": sum(len(t) for t in tokens) / max(len(tokens), 1),
        }
```

### Chained Metric (Uses Previous Results)

```python
class ComplexityMetric(MetricPlugin):
    """Compute complexity using difficulty and readability."""
    
    name = "complexity"
    
    def compute(self, sample: dict) -> dict:
        tags = sample.get("curriculum_tags", {})
        
        # Get scores from earlier metrics
        diff_score = tags.get("difficulty", {}).get("score", 0.5)
        fk_grade = tags.get("readability", {}).get("flesch_kincaid_grade", 8)
        
        # Combine into complexity score
        complexity = (diff_score * 0.6) + (min(fk_grade / 20, 1.0) * 0.4)
        
        return {
            "score": complexity,
            "level": "high" if complexity > 0.7 else "medium" if complexity > 0.4 else "low",
        }
```

## Troubleshooting

**Metric not loading?**
- Check class name matches config exactly
- Verify file name follows convention: `your_metric.py` → `YourMetric`
- Ensure `enabled: true` in config
- Check for syntax errors in your file

**Import errors?**
- Make sure you inherit from `MetricPlugin`
- Use relative imports within the package

**Tests failing?**
- Run individual test: `uv run pytest tests/test_your_metric.py::test_name -v`
- Add print statements to debug
- Check curriculum.yaml has required config keys

## Need Help?

- See existing metrics in this folder for reference
- Check [main README](../../../README.md) for architecture overview
- Run tests: `uv run pytest tests/ -v`
- Check metrics are loading: Add print in `CurriculumTagger._load_metrics()`

---

**Remember:** The system auto-discovers your metric - just create the file and add to config!
