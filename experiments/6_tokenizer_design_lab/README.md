# Tokenizer Design Lab - GPToss Pruning to 131K (131,072)

## Overview

This directory contains the pruned GPToss tokenizer optimized for 131,072 (2^17) vocabulary size while retaining Indic language support.

## Pruning Process

### Starting Point
- **GPToss original**: 200,000 tokens

### Iteration 1: Remove Long Tokens
- Removed tokens with >32 characters
- **Result**: 200,000 - 266 = **199,734 tokens**

### Iteration 2: Remove Non-Essential Languages
- Removed tokens for languages not in our target set (kept English + Indic languages)
- **Result**: 199,734 - 43,081 = **156,653 tokens**

### Iteration 3: 131K Cutoff
- Since our target tokenizer needs to be 131,072 (2^17), removed tokens with higher token IDs
- **Removed**: 156,653 - 131,046 = **25,607 tokens**
- **Important**: Indic tokens were carefully retained during this step
- The removed tokens are documented in `gptoss_pruning/removed_tokens.csv`

### Result
- **For Indic languages, GPToss and our pruned tokenizer have the same performance**

## Final Tokenizer Structure

| Range | Type | Count |
|-------|------|-------|
| 0 - 511 | Low-level Special tokens | 512 |
| 512 - 117,403 | Regular tokens (English/Code) | 116,892 |
| 117,404 - 131,045 | Indic tokens | 13,642 |
| 131,046 - 131,071 | New Special tokens (FIM, Vision, etc.) | 26 |
| **Total** | | **131,072** |

## Special Tokens

### Core Special Tokens (IDs 127488-127587)
- `<|begin_of_text|>` (ID 127488) - BOS token
- `<|end_of_text|>` (ID 127489) - EOS token / PAD token
- `<|pad|>` (ID 127490)
- `<|unk|>` (ID 127491)
- `<|system|>`, `<|user|>`, `<|assistant|>` - Chat roles
- `<|code_begin|>`, `<|code_end|>` - Code blocks
- Language tags: `<|lang:python|>`, `<|lang:javascript|>`, etc.

### New Special Tokens (IDs 131046-131071)
Added from Qwen-Code and DeepSeek-Code tokenizers via `add_special_tokens.py`:

| ID | Token | Purpose |
|----|-------|---------|
| 131046 | `<\|fim_prefix\|>` | Fill-in-the-Middle prefix |
| 131047 | `<\|fim_middle\|>` | Fill-in-the-Middle middle |
| 131048 | `<\|fim_suffix\|>` | Fill-in-the-Middle suffix |
| 131049 | `<\|fim_pad\|>` | Fill-in-the-Middle padding |
| 131050 | `<\|vision_start\|>` | Vision/multimodal start |
| 131051 | `<\|vision_end\|>` | Vision/multimodal end |
| 131052 | `<\|vision_pad\|>` | Vision padding |
| 131053 | `<\|image_pad\|>` | Image padding |
| 131054 | `<\|video_pad\|>` | Video padding |
| 131055 | `<\|object_ref_start\|>` | Object reference start |
| 131056 | `<\|object_ref_end\|>` | Object reference end |
| 131057 | `<\|box_start\|>` | Bounding box start |
| 131058 | `<\|box_end\|>` | Bounding box end |
| 131059 | `<\|quad_start\|>` | Quad coordinates start |
| 131060 | `<\|quad_end\|>` | Quad coordinates end |
| 131061 | `<\|im_start\|>` | Instruction/message start |
| 131062 | `<\|im_end\|>` | Instruction/message end |
| 131063 | `<\|file_sep\|>` | File separator |
| 131064 | `<\|repo_name\|>` | Repository name marker |
| 131065 | `<tool_call>` | Tool call start |
| 131066 | `</tool_call>` | Tool call end |
| 131067 | `<tool_response>` | Tool response start |
| 131068 | `</tool_response>` | Tool response end |
| 131069 | `<think>` | Reasoning/thinking start |
| 131070 | `</think>` | Reasoning/thinking end |
| 131071 | `<\|EOT\|>` | End of turn |

### Reserved Tokens
- 256+ reserved tokens are included in the initial block (IDs 0-511) for future use (e.g., `<|reserved_511|>`).

## Indic Language Support

The tokenizer retains **13,642 Indic tokens** covering:
- Hindi (Devanagari)
- Tamil
- Telugu
- Bengali
- Malayalam
- Gujarati
- Kannada
- And more...

### Sample Indic Tokens
| ID | Token (encoded) | Decoded |
|----|-----------------|---------|
| 113846 | `à¤¾` | ा (Hindi vowel sign) |
| 113847 | `à¥ĩ` | े (Hindi vowel sign) |
| 113848 | `à¤°` | र (Hindi letter ra) |
| 113853 | `à¦¾` | া (Bengali vowel sign) |
| 113857 | `Ġà¤ķ` | क (Hindi letter ka with space) |
| 113859 | `à¯į` | ் (Tamil virama) |

## Files

```
gptoss_pruning/
├── tokenizer.json          # Main tokenizer file
├── tokenizer_config.json   # Tokenizer configuration
├── special_tokens_map.json # Special token mappings
└── removed_tokens.csv      # Tokens removed in pruning

add_special_tokens.py       # Script to add new special tokens
build_clean_tokenizer.py    # Script used to build/prune the tokenizer
```

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./gptoss_pruning")

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

## Reproduction & Modification

To regenerate the tokenizer or modify the vocabulary size/settings, follow these steps:

### 1. Pruning and Base Generation (`build_clean_tokenizer.py`)
This script prunes the original GPToss tokenizer and generates the base vocabulary.

**To modify the target size:**
Edit `TARGET_VOCAB_SIZE` in `build_clean_tokenizer.py`.
> [!NOTE]
> Ensure you account for the tokens added in the next step.
> For a final size of **131,072**, we set `TARGET_VOCAB_SIZE = 131046` (131072 - 26).

```bash
python build_clean_tokenizer.py
```

### 2. Adding Special Tokens (`add_special_tokens.py`)
This script appends the 26 new special tokens (FIM, Vision, Tool use) to the end of the vocabulary.

**To add/remove special tokens:**
Edit the `NEW_TOKENS` list in `add_special_tokens.py`.

```bash
python add_special_tokens.py
```

### 3. Verification
Check the total token count to ensure it matches your target (e.g., 2^17 = 131,072).

```bash
python -c "import json; d=json.load(open('gptoss_pruning/tokenizer.json', encoding='utf-8')); print(len(d['model']['vocab']))"
```
