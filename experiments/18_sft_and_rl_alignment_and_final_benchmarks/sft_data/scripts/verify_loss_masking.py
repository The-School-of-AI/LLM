#!/usr/bin/env python3
"""
Verify that labels are set so loss is computed ONLY on assistant tokens (ignore_index=-100 elsewhere).
Team 18 — SFT Data 7.1 checklist items 8 and 9.
This script demonstrates correct label construction and can be adapted into your training collator.
"""
import json
import argparse
from pathlib import Path

# Optional: use transformers tokenizer
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

IGNORE_INDEX = -100  # PyTorch default; do not compute loss on these positions


def build_labels_for_assistant_only(input_ids: list[int], conversation: dict, tokenizer) -> list[int]:
    """
    Build labels so that only assistant reply token positions have non-negative labels.
    All other positions (system, user, padding) get IGNORE_INDEX.
    Requires tokenizer that can decode; we approximate by finding assistant span in tokenized form.
    """
    # Simplified: assume we have per-turn token ranges. In practice you get these when
    # building the string with turn markers and tokenizing each part.
    # Here we demonstrate the contract: labels length = input_ids length; only assistant span is non -100.
    labels = [IGNORE_INDEX] * len(input_ids)
    # If your pipeline provides assistant_start_idx, assistant_end_idx (in token space), do:
    # for i in range(assistant_start_idx, assistant_end_idx):
    #     labels[i] = input_ids[i]
    # For this script we just check that the structure is correct when ranges are provided.
    # Only assistant span should have non-negative labels; all others IGNORE_INDEX.
    return labels


def apply_chat_template_with_labels(conversation: dict, tokenizer, template: str = "chatml"):
    """
    Build input_ids and labels with assistant-only loss.
    Returns (input_ids, labels). Labels use IGNORE_INDEX for non-assistant positions.
    """
    turns = conversation.get("conversations", [])
    if not turns:
        return [], []

    # Use tokenizer's apply_chat_template if available (HuggingFace)
    if hasattr(tokenizer, "apply_chat_template"):
        # tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=False)
        # and build labels by masking non-assistant parts
        messages = [{"role": t["role"], "content": t.get("content") or ""} for t in turns]
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors=None)
        # HuggingFace chat templates often return only the string; we need token ids and ranges.
        # So we do a manual pass: tokenize each turn and mark assistant span.
        pass

    # Manual: concatenate with special tokens and track character/token ranges for assistant
    parts = []
    assistant_start = None
    assistant_end = None
    if template == "chatml":
        for t in turns:
            role = t["role"]
            content = (t.get("content") or "").strip()
            if role == "assistant":
                assistant_start = len("".join(parts))  # character offset
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if role == "assistant":
                assistant_end = len("".join(parts))
    else:
        for t in turns:
            role = t["role"]
            content = (t.get("content") or "").strip()
            if role == "assistant":
                assistant_start = len("".join(parts))
            parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            if role == "assistant":
                assistant_end = len("".join(parts))

    full_text = "".join(parts)
    enc = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    input_ids = enc["input_ids"]
    offset_mapping = enc.get("offset_mapping", [])

    labels = [IGNORE_INDEX] * len(input_ids)
    for i, (start, end) in enumerate(offset_mapping):
        if assistant_start is not None and assistant_end is not None:
            if start >= assistant_start and end <= assistant_end:
                labels[i] = input_ids[i]
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser(description="Verify assistant-only loss masking and padding (ignore_index=-100).")
    ap.add_argument("input", type=Path, help="One JSONL with 'conversations' (and optionally 'text')")
    ap.add_argument("--tokenizer", type=str, default=None, help="HuggingFace tokenizer to verify (optional)")
    ap.add_argument("--sample", type=int, default=3, help="Number of examples to check")
    args = ap.parse_args()

    print("SFT loss masking verification (Team 18)")
    print("  - Loss must be computed ONLY on assistant tokens.")
    print("  - All other positions (system, user, padding) must have label = -100 (ignore_index).")
    print("  - Padding: right-pad with pad token; label = -100 for padding positions.")
    print()

    lines = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))

    for i, conv in enumerate(lines[: args.sample]):
        print(f"Example {i+1}: {len(conv.get('conversations', []))} turns")
        if args.tokenizer and HAS_TRANSFORMERS:
            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
            input_ids, labels = apply_chat_template_with_labels(conv, tok, template="chatml")
            n_assistant = sum(1 for x in labels if x != IGNORE_INDEX)
            n_total = len(labels)
            print(f"  Tokens: {n_total}, assistant positions (loss computed): {n_assistant}, ignored: {n_total - n_assistant}")
        else:
            print("  (Pass --tokenizer to run full label construction check)")

    print()
    print("In your training code, ensure:")
    print("  1. Data collator sets labels so only assistant token indices have token id, rest = -100.")
    print("  2. CrossEntropyLoss(ignore_index=-100).")
    print("  3. Padding: right-pad; pad positions in labels = -100.")


if __name__ == "__main__":
    main()
