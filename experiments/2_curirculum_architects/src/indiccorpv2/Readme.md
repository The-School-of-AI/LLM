# IndicCorpV2 Cleaning Pipeline

IndicCorpV2 is web-crawled data without inherent quality tiers or folder-based difficulty signals. It's essentially a mixed bag that spans B0-B3, but you can't easily separate them without using filtering techniques because:

1. No folder structure by difficulty
2. No metadata indicating content type
3. Mix of high-quality (news, wiki) and low-quality (social, spam) content

Alternatives:
You already have cleaner Indic sources:

1. Sangraha Verified (structured, likely pre-filtered)
2. FineWeb2 Indic subset (likely higher quality)

## Dataset Stats
- 24 languages (23 Indic + Indian English)
- 20.9 Billion tokens
- Web-crawled from Indian websites
- License: CC-0

Total: **20.9 billion tokens** covering 24 languages (23 Indic + English).
- Indic portion: 14.4 billion tokens
- Indian English: 6.5 billion tokens

### High-Resource Languages

| Split | Language | Tokens | Sentences |
|-------|----------|--------|-----------|
| `hin_Deva` | Hindi | ~6-7B | 349M |
| `tel_Telu` | Telugu | ~2.1B | 108.5M |
| `urd_Arab` | Urdu | ~1.5B | 76.2M |
| `tam_Taml` | Tamil | ~1.3B | 64.7M |
| `ben_Beng` | Bengali | ~1.2B | 60M |
| `mar_Deva` | Marathi | ~1.0B | 34M |
| `guj_Gujr` | Gujarati | ~850M | 43M |
| `pan_Guru` | Punjabi | ~770M | 38.6M |
| `mal_Mlym` | Malayalam | ~700M | 34M |
| `kan_Knda` | Kannada | ~500M | 24M |
| `ory_Orya` | Odia | ~270M | 13.4M |

### Low-Resource Languages (Bottom 11)

Collectively ~1.08 billion tokens.

| Split | Language | Tokens |
|-------|----------|--------|
| `asm_Beng` | Assamese | ~66M |
| `san_Deva` | Sanskrit | ~88M |
| `npi_Deva` | Nepali | ~30M |
| `brx_Deva` | Bodo | <50M |
| `doi_Deva` | Dogri | <50M |
| `gom_Deva` | Konkani | <50M |
| `kas_Arab` | Kashmiri | <50M |
| `mai_Deva` | Maithili | <50M |
| `mni_Mtei` | Manipuri | <50M |
| `snd_Deva` | Sindhi | <50M |
| `santhali` | Santali | <50M |
| `khasi` | Khasi | <50M |

> English split (`eng_Latn`): ~6.5 billion tokens.

## Plan of usage:

1B Model
1. B0 -If possible avoid it, use Sangraha Verified + Fine Web Edu Indic . use <5% (only extremely simple text) if needed because Grammar stability needs clean data)

3B Model
1. B1 - Use 30-40% of Indic mix, use "news" + "blog" (will need creation of tags)

8B Model
1. B1 +B2 - Use 30% of Indic mix, add data for  "formal" + "wiki"

70B Model
1. Load entire data in B0-B2

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
