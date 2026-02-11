# TSAI 131K Tokenizer

## Overview

This directory contains the **TSAI 131K Tokenizer**, a pruned GPToss tokenizer optimized for 131,072 (2^17) vocabulary size while retaining Indic language support.

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./tsai_131k_tokenizer")

# Test encoding
text = "Hello, यह एक परीक्षण है"
tokens = tokenizer.encode(text)
print(tokens)
```

## Reproduction

To regenerate the tokenizer:

```bash
python build_clean_tokenizer.py
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
