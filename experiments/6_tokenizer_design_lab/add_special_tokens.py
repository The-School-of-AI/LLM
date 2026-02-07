#!/usr/bin/env python3
"""
Add meaningful special tokens from qwen_code and deepseek_code to gptoss_pruning.
"""

import json
from copy import deepcopy

# Meaningful tokens to add (filtered from comparison)
# Excluding: deepseek's fullwidth chars, redundant tokens
NEW_TOKENS = [
    # FIM (Fill-in-the-Middle) for code completion
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",

    # Vision/Multimodal
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",

    # Object reference (for grounding)
    "<|object_ref_start|>",
    "<|object_ref_end|>",

    # Bounding box/quad (for visual grounding)
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",

    # Chat format (im = instruction/message)
    "<|im_start|>",
    "<|im_end|>",

    # Code context
    "<|file_sep|>",
    "<|repo_name|>",

    # Tool use (XML style from qwen)
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",

    # Thinking (for reasoning models)
    "<think>",
    "</think>",

    # End of turn
    "<|EOT|>",
]


def load_tokenizer(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_tokenizer(data: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    tokenizer_path = 'gptoss_pruning/tokenizer.json'
    config_path = 'gptoss_pruning/tokenizer_config.json'

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)
    config = load_tokenizer(config_path)

    # Get current vocab and added_tokens
    vocab = tokenizer['model']['vocab']
    added_tokens = tokenizer['added_tokens']

    # Get existing token contents
    existing_tokens = {t['content'] for t in added_tokens}
    existing_vocab = set(vocab.keys())

    print(f"Current vocab size: {len(vocab)}")
    print(f"Current added_tokens count: {len(added_tokens)}")

    # Find next available ID
    max_id = max(vocab.values())
    next_id = max_id + 1

    print(f"Max vocab ID: {max_id}")
    print(f"Starting new tokens at ID: {next_id}")

    # Add new tokens
    tokens_added = []
    for token in NEW_TOKENS:
        if token in existing_tokens or token in existing_vocab:
            print(f"  SKIP (exists): {token}")
            continue

        # Add to added_tokens
        added_tokens.append({
            "id": next_id,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True
        })

        # Add to vocab
        vocab[token] = next_id

        tokens_added.append((token, next_id))
        print(f"  ADD: {token} -> ID {next_id}")
        next_id += 1

    print(f"\nTokens added: {len(tokens_added)}")
    print(f"New vocab size: {len(vocab)}")

    # Save updated tokenizer
    save_tokenizer(tokenizer, tokenizer_path)
    print(f"Saved tokenizer to: {tokenizer_path}")

    # Update tokenizer_config.json - add new tokens to added_tokens_decoder
    for token, token_id in tokens_added:
        config['added_tokens_decoder'][str(token_id)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True
        }

    save_tokenizer(config, config_path)
    print(f"Saved config to: {config_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
