#!/usr/bin/env python3
"""
Apply a chat template to standardized conversations and optionally tokenize.
Team 18 — SFT Data 7.1 checklist item 3.
"""
import json
import argparse
from pathlib import Path

# Optional: use HuggingFace tokenizer if available
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def chatml_format(turns: list[dict], tokenizer=None) -> str:
    """ChatML: <|im_start|>role\ncontent<|im_end|>"""
    parts = []
    for t in turns:
        role = t["role"]
        content = (t.get("content") or "").strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def llama_format(turns: list[dict], tokenizer=None) -> str:
    """Llama 3 style: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>"""
    parts = []
    for t in turns:
        role = t["role"]
        content = (t.get("content") or "").strip()
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    return "".join(parts)


def apply_template(conversation: dict, template: str) -> str:
    """Apply chosen template to one conversation. template: 'chatml' | 'llama'."""
    turns = conversation.get("conversations", [])
    if template == "chatml":
        return chatml_format(turns)
    if template == "llama":
        return llama_format(turns)
    raise ValueError(f"Unknown template: {template}")


def main():
    ap = argparse.ArgumentParser(description="Apply chat template to conversation JSONL.")
    ap.add_argument("input", type=Path, help="Input JSONL (standardized conversations)")
    ap.add_argument("output", type=Path, help="Output JSONL with 'text' field (and optional 'input_ids' if tokenizer given)")
    ap.add_argument("--template", choices=["chatml", "llama"], default="chatml", help="Chat template")
    ap.add_argument("--tokenizer", type=str, default=None, help="HuggingFace model name or path to tokenize (optional)")
    ap.add_argument("--max-length", type=int, default=None, help="Truncate to max length (optional)")
    args = ap.parse_args()

    if args.tokenizer and not HAS_TRANSFORMERS:
        raise SystemExit("--tokenizer requires transformers. pip install transformers")

    tokenizer = None
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    out_records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            text = apply_template(conv, args.template)
            rec = {"text": text, "conversations": conv.get("conversations", [])}
            if tokenizer:
                enc = tokenizer(text, truncation=args.max_length is not None, max_length=args.max_length or 2048)
                rec["input_ids"] = enc["input_ids"]
            out_records.append(rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for rec in out_records:
            # avoid writing large lists in JSON if not needed for inspection
            out = {"text": rec["text"]}
            if "input_ids" in rec:
                out["input_ids"] = rec["input_ids"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records)} templated examples to {args.output} (template={args.template})")


if __name__ == "__main__":
    main()
