# TSAI 131K Tokenizer

## Overview

This directory contains the **TSAI 131K Tokenizer**, a pruned GPToss tokenizer optimized for 131,072 (2^17) vocabulary size while retaining Indic language support.

## Usage

## Usage

The following example demonstrates how to load the tokenizer and the corresponding Kronecker embeddings together:

```python
import torch
from transformers import AutoTokenizer

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tsai_131k_tokenizer")

# Test encoding
text = "Hello, यह एक परीक्षण है"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# 2. Load Kronecker Embeddings
# These embeddings map each token ID directly to a vector (e.g., 8192-dim)
embeddings_path = "kronecker_embeddings/gptoss_kronecker_embeddings.pt"
if os.path.exists(embeddings_path):
    embeddings = torch.load(embeddings_path)
    print(f"Loaded embeddings: {embeddings.shape}")
    
    # Get embedding for a specific token
    token_id = tokens[0]
    token_emb = embeddings[token_id]
    print(f"Embedding for token {token_id}: {token_emb.shape}")
else:
    print(f"Embeddings file not found at {embeddings_path}. Please run reproduction steps.")
```

## Reproduction

To regenerate the tokenizer and embeddings from scratch:

1. **Build the tokenizer**:
   ```bash
   python build_clean_tokenizer.py
   ```
   This will generate the tokenizer in `tsai_131k_tokenizer/`.

2. **Generate Kronecker Embeddings**:
   ```bash
   cd kronecker_embeddings
   python convert_tokenizer_to_kronecker.py --tokenizer ../tsai_131k_tokenizer/tokenizer.json --output-dir .
   ```
   This will generate `gptoss_kronecker_embeddings.pt` (and `.npy`) in the `kronecker_embeddings/` directory.

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
