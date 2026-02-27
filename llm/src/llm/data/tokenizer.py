import os

from transformers import AutoTokenizer

from llm.utils import print_rank_0


def get_tokenizer(tokenizer_path: str):
    """
    Load and configure the TSAI 131K tokenizer.

    Args:
        tokenizer_path: Path to the tokenizer directory (default: src/tokenizer/)

    Returns:
        Configured tokenizer instance (TSAI 131K - 2^17 vocab size)
    """
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"TSAI 131K tokenizer not found at: {tokenizer_path}\n"
            "Expected directory structure: src/tokenizer/ with tokenizer.json, "
            "tokenizer_config.json, and special_tokens_map.json"
        )

    print_rank_0(f"  Loading TSAI 131K tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print_rank_0("  Tokenizer loaded:")
    print_rank_0(f"    - Vocab size: {tokenizer.vocab_size:,}")
    print_rank_0(f"    - Total tokens (with special): {len(tokenizer):,}")
    print_rank_0(
        f"    - BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})"
    )
    print_rank_0(
        f"    - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
    )
    print_rank_0(
        f"    - PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})"
    )

    return tokenizer
