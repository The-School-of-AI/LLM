# TSAI 131K Tokenizer - GPToss Pruned to 131,072 (2^17)

## Overview

This directory contains the **TSAI 131K Tokenizer** - a pruned GPToss tokenizer optimized for 131,072 (2^17) vocabulary size while retaining Indic language support.

## Tokenizer Layout

| Range | Type | Count |
|-------|------|-------|
| 0 - 117,073 | General tokens (English/Code) | 117,074 |
| 117,074 - 130,715 | Indic tokens | 13,642 |
| 130,716 - 131,071 | Special tokens (base + additional + reserved) | 356 |
| **Total** | | **131,072** |

## Pruning Process

### Starting Point
- **GPToss original**: ~200,000 tokens

### Step 1: Remove Long Tokens
- Removed tokens with >32 characters

### Step 2: Remove Non-Essential Languages
- Removed tokens for blocked scripts (CJK, Arabic, Cyrillic, etc.)
- Kept: English + Indic languages (Hindi, Tamil, Telugu, Bengali, etc.)

### Step 3: 131K Cutoff
- Reduced to fit 131,072 target (accounting for special tokens)
- Indic tokens preserved and placed before special tokens

### Step 4: Special Tokens at End
- Base special tokens (80): Document, chat, code, language tags, etc.
- Additional special tokens (26): FIM, Vision, Tool use, etc.
- Reserved tokens (250): For future expansion

## Special Tokens (IDs 130,716 - 131,071)

### Base Special Tokens (80 tokens)
- `<|begin_of_text|>` - BOS token
- `<|end_of_text|>` - EOS token / PAD token
- `<|pad|>`, `<|unk|>`, `<|sep|>`, `<|cls|>`, `<|mask|>`
- Chat roles: `<|system|>`, `<|user|>`, `<|assistant|>`, `<|tool|>`
- Code blocks: `<|code_begin|>`, `<|code_end|>`
- Language tags: `<|lang:python|>`, `<|lang:javascript|>`, etc.
- Thinking: `<|think_begin|>`, `<|think_end|>`, `<|step|>`, etc.

### Additional Special Tokens (26 tokens)
From Qwen-Code and DeepSeek-Code tokenizers:

| Token | Purpose |
|-------|---------|
| `<\|fim_prefix\|>` | Fill-in-the-Middle prefix |
| `<\|fim_middle\|>` | Fill-in-the-Middle middle |
| `<\|fim_suffix\|>` | Fill-in-the-Middle suffix |
| `<\|fim_pad\|>` | Fill-in-the-Middle padding |
| `<\|vision_start\|>` | Vision/multimodal start |
| `<\|vision_end\|>` | Vision/multimodal end |
| `<\|vision_pad\|>` | Vision padding |
| `<\|image_pad\|>` | Image padding |
| `<\|video_pad\|>` | Video padding |
| `<\|object_ref_start\|>` | Object reference start |
| `<\|object_ref_end\|>` | Object reference end |
| `<\|box_start\|>` | Bounding box start |
| `<\|box_end\|>` | Bounding box end |
| `<\|quad_start\|>` | Quad coordinates start |
| `<\|quad_end\|>` | Quad coordinates end |
| `<\|im_start\|>` | Instruction/message start |
| `<\|im_end\|>` | Instruction/message end |
| `<\|file_sep\|>` | File separator |
| `<\|repo_name\|>` | Repository name marker |
| `<tool_call>` | Tool call start |
| `</tool_call>` | Tool call end |
| `<tool_response>` | Tool response start |
| `</tool_response>` | Tool response end |
| `<think>` | Reasoning/thinking start |
| `</think>` | Reasoning/thinking end |
| `<\|EOT\|>` | End of turn |

### Reserved Tokens (250 tokens)
- `<|reserved_0|>` to `<|reserved_249|>` for future use

## Indic Language Support

The tokenizer retains **13,642 Indic tokens** covering:
- Hindi (Devanagari)
- Tamil
- Telugu
- Bengali
- Malayalam
- Gujarati
- Kannada
- Oriya
- Gurmukhi (Punjabi)
- Sinhala

## Files

```
tsai_131k_tokenizer/
├── tokenizer.json          # Main tokenizer file
├── tokenizer_config.json   # Tokenizer configuration
├── special_tokens_map.json # Special token mappings (BOS/EOS/PAD)
├── build_report.csv        # Token ID mapping audit
└── removed_tokens.csv      # Tokens removed during pruning

build_clean_tokenizer.py    # Script to build/prune the tokenizer
special_tokens.py           # Special token definitions
```

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tsai_131k_tokenizer")

# Test encoding
text = "Hello, यह एक परीक्षण है"
tokens = tokenizer.encode(text)
print(tokens)
```

## Token Mappings (special_tokens_map.json)

```json
{
  "bos_token": "<|begin_of_text|>",
  "eos_token": "<|end_of_text|>",
  "pad_token": "<|end_of_text|>"
}
```

## Metrics & Performance

The following graphs summarize the performance of the tokenizer across different domains:

![Bytes per Token](Tokenizer_metrics/graphs/Summary_Bytes_Token.png)
*Bytes per Token (Lower is Better)*

![Fertility](Tokenizer_metrics/graphs/Summary_Fertility.png)
*Fertility (Tokens per Word)*

![Speed](Tokenizer_metrics/graphs/Summary_Speed.png)
*Speed (Tokens/sec) (Higher is Better)*

![Fallback Rate](Tokenizer_metrics/graphs/Summary_Fallback.png)
*Byte Fallback Rate (Lower is Better)*

![Vocab Gini](Tokenizer_metrics/graphs/Summary_Vocab.png)
*Vocabulary Inequality (Higher = Less Balanced)*

### Comparative Analysis

Based on the evaluation metrics across Code, Indic languages, and NCERT textbooks:

*   **Academic & General Text (NCERT)**: **`our_tokenizer` demonstrates excellent performance**, often matching or outperforming `gemma_tokenizer` and `mistral_tokenizer` in compression efficiency (lower Bytes/Token). For subjects like Biology and Chemistry, it rivals the highly efficient `qwen_tokenizer`.
*   **Code**: Our tokenizer remains **highly competitive**, showing better compression than the original `gptoss_tokenizer` and `qwen_tokenizer` in languages like Python and Java. While `gemma_tokenizer` holds a slight edge in raw compression, our tokenizer strikes a balanced trade-off.
*   **Indic Languages**: The tokenizer retains functional support for languages like Hindi and Tamil. While `qwen_tokenizer` leads in this category due to its extensive multilingual vocabulary, `our_tokenizer` maintains stability consistent with the base GPToss model, ensuring these languages are processed correctly without fallback errors.

## Reproduction

To regenerate the tokenizer:

```bash
python build_clean_tokenizer.py
```

### Configuration

Edit `build_clean_tokenizer.py` to modify:
- `TARGET_VOCAB_SIZE`: Target vocabulary size (default: 131,072)
- `NUM_RESERVED`: Number of reserved tokens (default: 250)
- `MAX_TOKEN_LEN`: Maximum token length (default: 32)

Edit `special_tokens.py` to modify special token definitions.

### Verification

```bash
python -c "import json; d=json.load(open('tsai_131k_tokenizer/tokenizer.json', encoding='utf-8')); print(f'Vocab size: {len(d[\"model\"][\"vocab\"]):,}')"
```
