"""
convert_tokenizer_to_kronecker.py - Generate Kronecker Embeddings for GPT-OSS Tokenizer

This script reads a HuggingFace tokenizer.json file and generates a fixed
Kronecker product embedding matrix for all tokens in the vocabulary.

Usage:
    py convert_tokenizer_to_kronecker.py [--verify] [--limit N]

Output:
    - gptoss_kronecker_embeddings.npy: The embedding matrix (vocab_size, 8192)
    - gptoss_kronecker_config.json: Metadata (dimensions, vocab_size, special_tokens)

Author: Generated for GPT-OSS tokenizer conversion
Date: 2026-02-09
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Import local Kronecker encoder
from kronecker_decoder import KroneckerConfig, KroneckerEmbeddings


def load_tokenizer_vocab(tokenizer_path: str) -> Tuple[Dict[str, int], List[dict]]:
    """
    Load vocabulary and special tokens from HuggingFace tokenizer.json.

    Args:
        tokenizer_path: Path to tokenizer.json

    Returns:
        Tuple of (vocab dict, special_tokens list)
    """
    print(f"Loading tokenizer from: {tokenizer_path}")

    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary from model section
    model = tokenizer_data.get('model', {})
    vocab = model.get('vocab', {})

    # Extract special tokens
    added_tokens = tokenizer_data.get('added_tokens', [])
    special_tokens = [t for t in added_tokens if t.get('special', False)]

    print(f"  Loaded {len(vocab)} regular tokens")
    print(f"  Loaded {len(special_tokens)} special tokens")

    return vocab, special_tokens


def decode_token_string(token: str) -> str:
    """
    Decode HuggingFace token string to actual text.

    HuggingFace tokenizers often use special characters:
    - 'Ġ' (U+0120) represents a leading space
    - 'Ċ' (U+010A) represents a newline
    - Byte-level tokens like <0x00> to <0xFF>

    Args:
        token: Raw token string from vocab

    Returns:
        Decoded token string
    """
    # Handle byte tokens like <0x00>, <0xAB>, etc.
    if token.startswith('<0x') and token.endswith('>') and len(token) == 6:
        try:
            byte_val = int(token[3:5], 16)
            return bytes([byte_val]).decode('utf-8', errors='replace')
        except (ValueError, UnicodeDecodeError):
            return token

    # Replace HuggingFace special characters
    decoded = token.replace('Ġ', ' ')   # Leading space
    decoded = decoded.replace('Ċ', '\n')  # Newline
    decoded = decoded.replace('ĉ', '\t')  # Tab

    return decoded


def generate_kronecker_embeddings(
    vocab: Dict[str, int],
    special_tokens: List[dict],
    encoder: KroneckerEmbeddings,
    limit: Optional[int] = None
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Generate Kronecker embeddings for all tokens.

    Args:
        vocab: Token -> ID mapping
        special_tokens: List of special token dicts
        encoder: KroneckerEmbeddings instance
        limit: Optional limit on number of tokens to process

    Returns:
        Tuple of (embedding matrix, id_to_token mapping)
    """
    # Combine all tokens: special tokens + regular vocab
    all_tokens: Dict[int, str] = {}

    # Add special tokens first (they have explicit IDs)
    for st in special_tokens:
        token_id = st['id']
        content = st['content']
        all_tokens[token_id] = content

    # Add regular vocab (token_str -> id)
    for token_str, token_id in vocab.items():
        if token_id not in all_tokens:  # Don't overwrite special tokens
            all_tokens[token_id] = token_str

    # Determine vocab size
    vocab_size = max(all_tokens.keys()) + 1 if all_tokens else 0

    if limit and limit < vocab_size:
        vocab_size = limit
        print(f"Limiting to first {limit} tokens")

    print(f"Total vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {encoder.D}")
    print(f"Generating embeddings...")

    # Initialize embedding matrix
    embeddings = np.zeros((vocab_size, encoder.D), dtype=np.float32)

    # Track statistics
    processed = 0
    skipped = 0
    special_count = 0

    for token_id in range(vocab_size):
        if token_id in all_tokens:
            token_str = all_tokens[token_id]

            # Check if special token
            is_special = any(st['id'] == token_id for st in special_tokens)

            if is_special:
                # For special tokens, use the content directly (e.g., "<|begin_of_text|>")
                # This encodes the special token marker as regular text
                decoded = token_str
                special_count += 1
            else:
                # Decode regular token
                decoded = decode_token_string(token_str)

            # Generate Kronecker embedding
            embedding = encoder.encode_word(decoded)
            embeddings[token_id] = embedding
            processed += 1
        else:
            # Token ID not in vocab (gap in IDs) - leave as zeros
            skipped += 1

        # Progress indicator
        if (token_id + 1) % 10000 == 0:
            print(f"  Processed {token_id + 1}/{vocab_size} tokens...")

    print(f"  Completed: {processed} tokens processed, {skipped} gaps, {special_count} special tokens")

    return embeddings, all_tokens


def save_outputs(
    embeddings: np.ndarray,
    vocab_size: int,
    special_tokens: List[dict],
    output_dir: Path
):
    """
    Save embeddings and config to files.

    Args:
        embeddings: The embedding matrix
        vocab_size: Total vocabulary size
        special_tokens: List of special token dicts
        output_dir: Output directory
    """
    # Save embeddings as .npy
    embeddings_path = output_dir / "gptoss_kronecker_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / (1024**2):.2f} MB")

    # Save config as JSON
    config = {
        "vocab_size": vocab_size,
        "embedding_dim": 8192,
        "char_dim": 256,
        "pos_dim": 32,
        "encoding": "byte-level",
        "length_normalized": True,
        "special_token_ids": [st['id'] for st in special_tokens],
        "special_token_count": len(special_tokens),
    }

    config_path = output_dir / "gptoss_kronecker_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


def verify_embeddings(
    embeddings: np.ndarray,
    id_to_token: Dict[int, str],
    encoder: KroneckerEmbeddings,
    num_samples: int = 20
):
    """
    Verify embeddings by checking encoding/decoding consistency.

    Args:
        embeddings: The generated embedding matrix
        id_to_token: ID -> token mapping
        encoder: KroneckerEmbeddings instance
        num_samples: Number of random tokens to test
    """
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Only use token IDs that are within the embeddings matrix
    max_id = embeddings.shape[0] - 1
    valid_ids = [tid for tid in id_to_token.keys() if tid <= max_id]

    # Sample random token IDs from valid range
    np.random.seed(42)
    sample_ids = np.random.choice(valid_ids, size=min(num_samples, len(valid_ids)), replace=False)

    passed = 0
    failed = 0

    print(f"\nTesting {len(sample_ids)} random tokens:\n")

    for token_id in sample_ids:
        token_str = id_to_token[token_id]
        decoded_str = decode_token_string(token_str)

        # Get stored embedding
        stored_emb = embeddings[token_id]

        # Generate fresh embedding
        fresh_emb = encoder.encode_word(decoded_str)

        # Check if they match
        match = np.allclose(stored_emb, fresh_emb, atol=1e-6)

        # Try to decode the stored embedding
        decoded_back = encoder.decode_word(stored_emb)

        # Check reconstruction
        reconstruction_ok = decoded_back == decoded_str

        status = "[PASS]" if (match and reconstruction_ok) else "[FAIL]"
        if match and reconstruction_ok:
            passed += 1
        else:
            failed += 1

        # Truncate for display and make ASCII-safe
        display_token = ascii(decoded_str[:20]) if len(decoded_str) > 20 else ascii(decoded_str)
        display_decoded = ascii(decoded_back[:20]) if len(decoded_back) > 20 else ascii(decoded_back)

        print(f"  {status} ID {token_id:6d}: {display_token:30s} -> {display_decoded:30s} | match={match}")

    print(f"\nResults: {passed}/{passed + failed} passed")

    if failed == 0:
        print("[OK] All verification tests PASSED!")
    else:
        print(f"[WARNING] {failed} tests FAILED")


def main():
    parser = argparse.ArgumentParser(description="Convert GPT-OSS tokenizer to Kronecker embeddings")
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='../tsai_131k_tokenizer/tokenizer.json',
        help='Path to tokenizer.json'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification after generation'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tokens to process (for testing)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    tokenizer_path = script_dir / args.tokenizer
    output_dir = script_dir / args.output_dir

    if not tokenizer_path.exists():
        print(f"ERROR: Tokenizer not found: {tokenizer_path}")
        sys.exit(1)

    print("=" * 60)
    print("GPT-OSS TOKENIZER TO KRONECKER EMBEDDINGS CONVERTER")
    print("=" * 60)
    print()

    # Initialize Kronecker encoder
    cfg = KroneckerConfig(
        CHAR_DIM=256,
        POS_DIM=32,
        D=8192,
        length_normalize=True,
        truncate_long_words=True
    )
    encoder = KroneckerEmbeddings(cfg)
    print(f"Kronecker config: CHAR_DIM={cfg.CHAR_DIM}, POS_DIM={cfg.POS_DIM}, D={cfg.D}")
    print()

    # Load tokenizer
    vocab, special_tokens = load_tokenizer_vocab(str(tokenizer_path))
    print()

    # Generate embeddings
    embeddings, id_to_token = generate_kronecker_embeddings(
        vocab, special_tokens, encoder, limit=args.limit
    )
    print()

    # Save outputs
    save_outputs(embeddings, embeddings.shape[0], special_tokens, output_dir)
    print()

    # Optionally verify
    if args.verify:
        verify_embeddings(embeddings, id_to_token, encoder)

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
