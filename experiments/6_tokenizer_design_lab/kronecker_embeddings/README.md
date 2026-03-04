# Kronecker Embeddings

Byte-Level Kronecker Product Embeddings for the GPT-OSS tokenizer (131K vocab).

## Files

| File | Description |
|---|---|
| `kronecker_decoder.py` | Core encoder/decoder engine (CPU + GPU) |
| `convert_tokenizer_to_kronecker.py` | Generates the `.pt` lookup table from `tokenizer.json` |
| `gptoss_kronecker_config.json` | Metadata (vocab size, dimensions, special tokens) |
| `KRONECKER_TEST_REPORT.md` | Test report |

## How It Works

```
"cat" -> UTF-8 bytes [99, 97, 116]
      -> 256x32 sparse matrix (1s at byte positions)
      -> Flatten to 8192-dim vector
      -> Project to 4096 (trainable)
      -> Feed to Transformer
```

## Usage

### Python API

You can use the `KroneckerEmbeddings` class directly to encode and decode text:

```python
from kronecker_decoder import KroneckerConfig, KroneckerEmbeddings

import numpy as np

# 1. Initialize Encoder
cfg = KroneckerConfig(
    CHAR_DIM=256,   # Byte vocabulary (0-255)
    POS_DIM=32,     # Max token length in bytes
    D=8192          # Total dimension (256 * 32)
)
encoder = KroneckerEmbeddings(cfg)

# 2. Encode a word
# Returns an 8192-dimensional numpy array
embedding = encoder.encode_word("hello")
print(f"Embedding shape: {embedding.shape}")  # (8192,)

# 3. Decode back to text
decoded = encoder.decode_word(embedding)
print(f"Decoded: {decoded}")  # "hello"

# 4. Batch encoding
words = ["hello", "world"]
batch_embeddings = encoder.encode_batch(words)
print(f"Batch shape: {batch_embeddings.shape}")  # (2, 8192)
```

## Generate Embeddings

If the tokenizer vocabulary changes or you need to regenerate the embeddings, run the following script:

### 1. Generate Embeddings (.npy)

Generates the raw Kronecker embeddings and configuration from the tokenizer JSON.

```bash
python convert_tokenizer_to_kronecker.py --tokenizer ../tsai_131k_tokenizer/tokenizer.json --output-dir .
```
*Output: `gptoss_kronecker_embeddings.npy` (float32), `gptoss_kronecker_config.json`*

## Requirements

```
numpy
```
