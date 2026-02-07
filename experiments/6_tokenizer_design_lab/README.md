# Tokenizer Design Lab - GPToss Pruning to 128K

## Overview

This directory contains the pruned GPToss tokenizer optimized for 128K vocabulary size while retaining Indic language support.

## Pruning Process

### Starting Point
- **GPToss original**: 200,000 tokens

### Iteration 1: Remove Long Tokens
- Removed tokens with >32 characters
- **Result**: 200,000 - 266 = **199,734 tokens**

### Iteration 2: Remove Non-Essential Languages
- Removed tokens for languages not in our target set (kept English + Indic languages)
- **Result**: 199,734 - 43,081 = **156,653 tokens**

### Iteration 3: 128K Cutoff
- Since our target tokenizer needs to be 128K, removed tokens with higher token IDs
- **Removed**: 156,653 - 128,000 = **28,653 tokens**
- **Important**: Indic tokens were carefully retained during this step
- The removed tokens are documented in `gptoss_pruning/removed_128k_cutoff.csv`

### Result
- **For Indic languages, GPToss and our pruned tokenizer have the same performance**

## Final Tokenizer Structure

| Range | Type | Count |
|-------|------|-------|
| 0 - 127,487 | Regular tokens | 127,488 |
| 127,488 - 127,587 | Original special tokens | 100 |
| 127,588 - 127,613 | New special tokens (added from Qwen/DeepSeek) | 26 |
| 127,614 - 128,025 | Reserved tokens | 412 |
| **Total** | | **128,026** |

## Special Tokens

### Core Special Tokens (IDs 127488-127587)
- `<|begin_of_text|>` (ID 127488) - BOS token
- `<|end_of_text|>` (ID 127489) - EOS token / PAD token
- `<|pad|>` (ID 127490)
- `<|unk|>` (ID 127491)
- `<|system|>`, `<|user|>`, `<|assistant|>` - Chat roles
- `<|code_begin|>`, `<|code_end|>` - Code blocks
- Language tags: `<|lang:python|>`, `<|lang:javascript|>`, etc.

### New Special Tokens (IDs 127588-127613)
Added from Qwen-Code and DeepSeek-Code tokenizers:

| ID | Token | Purpose |
|----|-------|---------|
| 127588 | `<\|fim_prefix\|>` | Fill-in-the-Middle prefix |
| 127589 | `<\|fim_middle\|>` | Fill-in-the-Middle middle |
| 127590 | `<\|fim_suffix\|>` | Fill-in-the-Middle suffix |
| 127591 | `<\|fim_pad\|>` | Fill-in-the-Middle padding |
| 127592 | `<\|vision_start\|>` | Vision/multimodal start |
| 127593 | `<\|vision_end\|>` | Vision/multimodal end |
| 127594 | `<\|vision_pad\|>` | Vision padding |
| 127595 | `<\|image_pad\|>` | Image padding |
| 127596 | `<\|video_pad\|>` | Video padding |
| 127597 | `<\|object_ref_start\|>` | Object reference start |
| 127598 | `<\|object_ref_end\|>` | Object reference end |
| 127599 | `<\|box_start\|>` | Bounding box start |
| 127600 | `<\|box_end\|>` | Bounding box end |
| 127601 | `<\|quad_start\|>` | Quad coordinates start |
| 127602 | `<\|quad_end\|>` | Quad coordinates end |
| 127603 | `<\|im_start\|>` | Instruction/message start |
| 127604 | `<\|im_end\|>` | Instruction/message end |
| 127605 | `<\|file_sep\|>` | File separator |
| 127606 | `<\|repo_name\|>` | Repository name marker |
| 127607 | `<tool_call>` | Tool call start |
| 127608 | `</tool_call>` | Tool call end |
| 127609 | `<tool_response>` | Tool response start |
| 127610 | `</tool_response>` | Tool response end |
| 127611 | `<think>` | Reasoning/thinking start |
| 127612 | `</think>` | Reasoning/thinking end |
| 127613 | `<\|EOT\|>` | End of turn |

### Reserved Tokens (IDs 127614-128025)
- 412 reserved tokens for future use: `<|reserved_100|>` to `<|reserved_511|>`

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
└── removed_128k_cutoff.csv # Tokens removed in 128K cutoff (28,653 tokens)

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
