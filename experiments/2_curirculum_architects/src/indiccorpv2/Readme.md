# IndicCorpV2 Cleaning Pipeline

Cleans and categorizes IndicCorpV2 dataset by difficulty levels (B0-B4) for staged LLM training.

## Difficulty Ladder

| Level | Name | Description |
|-------|------|-------------|
| B0 | Nursery | Simple grammar, high repetition |
| B1 | Primary | Fluent general knowledge |
| B2 | High School | Structured narrative, longer docs |
| B3 | Undergrad | Technical, reasoning required |
| B4 | Graduate | Advanced technical content |

## Quick Start

# Process Hindi (use 2-letter or 3-letter lang code)
```bash
uv run indiccorp_cleaner.py \
    --lang hi \
    --script Deva \
    --output-dir ./processed_data/hi_Deva \
    --limit 1000
```

## Available Splits

The dataset uses ISO 639-3 (3-letter) codes. Common mappings:

| 2-letter | 3-letter | Script | Split name |
|----------|----------|--------|------------|
| hi | hin | Deva | `hin_Deva` |
| bn | ben | Beng | `ben_Beng` |
| te | tel | Telu | `tel_Telu` |
| ta | tam | Taml | `tam_Taml` |
| mr | mar | Deva | `mar_Deva` |
| gu | guj | Gujr | `guj_Gujr` |
| kn | kan | Knda | `kan_Knda` |
| ml | mal | Mlym | `mal_Mlym` |
| pa | pan | Guru | `pan_Guru` |
| or | ory | Orya | `ory_Orya` |
| ur | urd | Arab | `urd_Arab` |

All splits: `asm_Beng`, `ben_Beng`, `brx_Deva`, `doi_Deva`, `gom_Deva`, `guj_Gujr`, `hin_Deva`, `kan_Knda`, `kas_Arab`, `mai_Deva`, `mal_Mlym`, `mar_Deva`, `mni_Mtei`, `npi_Deva`, `ory_Orya`, `pan_Guru`, `san_Deva`, `snd_Deva`, `tam_Taml`, `tel_Telu`, `urd_Arab`, `khasi`, `santhali`

## CLI Options

```bash
uv run indiccorp_cleaner.py \
    --lang hi              # Language code (2 or 3 letter)
    --script Deva          # Script code
    --output-dir ./output  # Output directory
    --batch-size 10000     # Docs per batch file (default: 10000)
    --limit 1000           # Limit docs for testing (optional)
```

# Analyse Clean Data
```bash
uv run analyze_processed.py --processed-dir ./processed_data
```
## Output Structure

```
processed_data/hi_Deva/
├── B0/batch_*.jsonl
├── B1/batch_*.jsonl
├── B2/batch_*.jsonl
├── B3/batch_*.jsonl
├── B4/batch_*.jsonl
└── processing_stats.json
```

Each JSONL line:
```json
{
  "text": "...",
  "difficulty": "B2",
  "difficulty_confidence": 0.85,
  "category": "formal",
  "word_count": 423,
  "metadata": {"lang": "hi", "lexical_diversity": 0.52}
}
```

## Using Processed Data

```python
from datasets import load_dataset, concatenate_datasets

# Load specific difficulty
b1 = load_dataset('json', data_files='processed_data/hi_Deva/B1/*.jsonl', split='train')

# Load all
all_data = load_dataset('json', data_files='processed_data/hi_Deva/*/*.jsonl', split='train')

# Filter by category
news = b1.filter(lambda x: x['category'] == 'news')
```

## Quality Filters

| Filter | Threshold | Purpose |
|--------|-----------|---------|
| Min words | 10 | Remove snippets |
| Max words | 10,000 | Remove dumps |
| Symbol ratio | <30% | Remove spam |
| Lexical diversity | >0.15 | Remove repetitive text |
| Deduplication | Exact match | Remove duplicates |

## Customizing

Edit `FilterConfig` in `indiccorp_cleaner.py`:

```python
@dataclass
class FilterConfig:
    min_words: int = 10
    max_words: int = 10000
    max_symbol_ratio: float = 0.3
    min_lexical_diversity: float = 0.15
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Too much removed | Lower `min_words`, `min_lexical_diversity` |
| Out of memory | Use `--batch-size 1000` |
| Slow | Use `--limit` for testing first |

## Related Files

- `indiccorp_cleaner.py` — Main processing script
- `explore_indiccorpv2.ipynb` — Dataset exploration notebook
