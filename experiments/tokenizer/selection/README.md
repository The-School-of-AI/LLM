# Tokenizer Vocabulary Analysis

A comprehensive analysis of vocabulary composition across 6 major LLM tokenizers, with special focus on multilingual coverage and Indic language representation.

## Tokenizers Analyzed

| Tokenizer | Source Model | Vocabulary Size | Type |
|-----------|-------------|-----------------|------|
| DeepSeek V3 | DeepSeek | 128,000 | BPE (byte-level) |
| GPT-OSS | OpenAI-style | 199,998 | BPE (byte-level) |
| Qwen | Alibaba Qwen | 151,643 | BPE (byte-level) |
| Mistral | Mistral AI | 131,072 | BPE (byte-level) |
| OLMo | AI2 OLMo | 100,278 | BPE (byte-level) |
| ByteD | ByteDance | 155,121 | BPE (byte-level) |

## Setup & Installation

### Prerequisites
- Python 3.10+
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment**
   ```powershell
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   
   # Windows Command Prompt
   .\.venv\Scripts\activate.bat
   
   # Linux/macOS
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Place tokenizer JSON files**
   
   Ensure the following tokenizer files are in the project root:
   - `ds_tokenizer.json` (DeepSeek V3)
   - `gptoss_tokenizer.json` (GPT-OSS)
   - `qwen_tokenizer.json` (Qwen)
   - `mistral_tokenizer.json` (Mistral)
   - `olmo_tokenizer.json` (OLMo)
   - `byted_tokenizer.json` (ByteD)

### Running the Analysis

1. **Open the notebook**
   ```bash
   jupyter notebook tokenizer_analysis.ipynb
   ```
   Or open in VS Code with the Jupyter extension.

2. **Run all cells** - The notebook executes sequentially and generates:
   - Token statistics and largest tokens
   - Language/script classification
   - Histograms of token length distributions
   - Stacked bar charts for language composition
   - Comparative summary tables

## Analysis Methodology

### Byte-Level BPE Decoding
All tokenizers use GPT-2 style byte-level BPE encoding. The analysis:
1. Decodes raw BPE tokens back to readable strings
2. Classifies each token by Unicode script/language
3. Calculates distribution statistics

### Language Classification
Tokens are classified into:
- **Latin (ASCII)** - English and Western European languages
- **Chinese** - CJK Unified Ideographs
- **Cyrillic** - Russian, Ukrainian, etc.
- **Arabic** - Arabic script
- **Korean** - Hangul
- **Hebrew** - Hebrew script
- **Thai** - Thai script
- **Vietnamese** - Latin with Vietnamese diacritics
- **Greek** - Greek script
- **10 Indic Scripts**: Devanagari, Bengali, Tamil, Kannada, Telugu, Malayalam, Gujarati, Gurmukhi, Sinhala, Odia

## Key Insights

### 1. Indic Language Coverage Varies Dramatically

| Tokenizer | Indic Tokens | Percentage |
|-----------|-------------|------------|
| **GPT-OSS** | 13,634 | **6.82%** (Best) |
| Mistral | 5,197 | 3.96% |
| ByteD | 2,511 | 1.62% |
| DeepSeek | 1,473 | 1.15% |
| Qwen | 279 | 0.18% |
| **OLMo** | 43 | **0.04%** (Worst) |

**Insight**: GPT-OSS has 317x more Indic tokens than OLMo. This significantly impacts tokenization efficiency for Indic languages - poor coverage means more tokens needed to represent the same text.

### 2. Indic Script Distribution

| Script | Best Coverage | Token Count |
|--------|--------------|-------------|
| Devanagari (Hindi) | GPT-OSS | 3,982 |
| Bengali | GPT-OSS | 2,129 |
| Telugu | GPT-OSS | 1,326 |
| Malayalam | GPT-OSS | 1,657 |
| Gujarati | GPT-OSS | 1,620 |
| Kannada | GPT-OSS | 1,308 |
| Tamil | GPT-OSS | 975 |
| Gurmukhi (Punjabi) | GPT-OSS | 306 |
| Sinhala | GPT-OSS | 293 |
| Odia | GPT-OSS | 38 |

**Insight**: GPT-OSS leads in all Indic scripts. Odia has the weakest representation across all tokenizers.

### 3. Top 10 Non-Indic Languages

| Language | Leader | Token Count |
|----------|--------|-------------|
| Latin (ASCII) | GPT-OSS | 133,729 |
| Chinese | ByteD | 50,961 |
| Cyrillic | GPT-OSS | 14,205 |
| Arabic | Mistral | 9,431 |
| Korean | Mistral | 4,472 |
| Hebrew | Qwen | 3,164 |
| Thai | Qwen | 2,571 |

**Insight**: 
- **ByteD** has exceptional Chinese coverage (50,961 tokens) - nearly 5x GPT-OSS
- **DeepSeek** also strong in Chinese (35,277 tokens)
- **Mistral** leads in Arabic and Korean coverage

### 4. Vocabulary Efficiency

| Metric | Best | Worst |
|--------|------|-------|
| Largest Vocabulary | GPT-OSS (199,998) | OLMo (100,278) |
| Long Tokens (>32 chars) | Qwen (0.14%), OLMo (0.21%) | Mistral (0.03%) |

**Insight**: Qwen and OLMo have more long tokens, which can improve compression for common phrases but may increase vocabulary memory usage.

### 5. Design Philosophy Observations

- **GPT-OSS**: Optimized for multilingual coverage, especially Indic languages
- **ByteD/DeepSeek**: Strong focus on Chinese language support
- **Mistral**: Balanced coverage with good Arabic/Korean representation
- **Qwen**: Moderate Chinese focus, limited Indic support
- **OLMo**: English-centric design with minimal multilingual support

## Implications for Model Selection

| Use Case | Recommended Tokenizer |
|----------|----------------------|
| Hindi/Indic languages | GPT-OSS, Mistral |
| Chinese text processing | ByteD, DeepSeek |
| Arabic content | Mistral |
| Korean content | Mistral |
| English-only applications | Any (OLMo is smallest) |
| Multilingual general purpose | GPT-OSS |

## File Structure

```
tokenizer/
├── tokenizer_analysis.ipynb   # Main analysis notebook
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── ds_tokenizer.json          # DeepSeek V3 tokenizer
├── gptoss_tokenizer.json      # GPT-OSS tokenizer
├── qwen_tokenizer.json        # Qwen tokenizer
├── mistral_tokenizer.json     # Mistral tokenizer
├── olmo_tokenizer.json        # OLMo tokenizer
└── byted_tokenizer.json       # ByteD tokenizer
```

## Dependencies

Key packages used:
- `transformers` - Hugging Face tokenizers library
- `tokenizers` - Fast tokenizer implementations
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `pandas` - Data manipulation

See `requirements.txt` for full list.

## License

This analysis is provided for educational and research purposes.

## Contributing

Feel free to add more tokenizers or extend the analysis with additional metrics.
