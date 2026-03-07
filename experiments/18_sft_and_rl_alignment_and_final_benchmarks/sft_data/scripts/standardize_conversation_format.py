#!/usr/bin/env python3
"""
Standardize SFT data to conversation format: system / user / assistant turns.
Supports Alpaca-style (instruction/input/output) and ShareGPT-style (conversations).
Team 18 — SFT Data 7.1 checklist item 2.
"""
import json
import argparse
from pathlib import Path


def alpaca_to_conversations(instruction: str, input_: str, output: str, system: str | None = None) -> list[dict]:
    """Convert one Alpaca-style example to list of turns."""
    turns = []
    if system:
        turns.append({"role": "system", "content": system})
    user_content = instruction.strip()
    if (input_ or "").strip():
        user_content += "\n\n" + input_.strip()
    turns.append({"role": "user", "content": user_content})
    turns.append({"role": "assistant", "content": (output or "").strip()})
    return turns


def sharegpt_to_conversations(conversations: list) -> list[dict]:
    """Convert ShareGPT-style conversation list to standardized roles. Maps 'human'->user, 'gpt'->assistant."""
    out = []
    for turn in conversations:
        role = (turn.get("from") or turn.get("role", "")).lower()
        content = (turn.get("value") or turn.get("content", "")).strip()
        if role in ("human", "user"):
            out.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant", "bot"):
            out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content})
    return out


def normalize_to_conversation(record: dict, format_: str) -> dict | None:
    """
    Normalize one record to { "conversations": [ {"role": "...", "content": "..."}, ... ] }.
    format_ one of: "alpaca", "sharegpt", "already_conversation".
    """
    if format_ == "already_conversation":
        conv = record.get("conversations") or record.get("messages") or []
        if not conv:
            return None
        normalized = []
        for t in conv:
            r = (t.get("role") or t.get("from", "")).lower()
            c = (t.get("content") or t.get("value", "")).strip()
            if r in ("human", "user"):
                normalized.append({"role": "user", "content": c})
            elif r in ("gpt", "assistant", "bot"):
                normalized.append({"role": "assistant", "content": c})
            elif r == "system":
                normalized.append({"role": "system", "content": c})
        if not normalized:
            return None
        return {"conversations": normalized}

    if format_ == "alpaca":
        inst = record.get("instruction") or record.get("instruction_plain", "")
        inp = record.get("input") or record.get("input_plain", "")
        out = record.get("output") or record.get("response", "")
        system = record.get("system") or None
        turns = alpaca_to_conversations(inst, inp, out, system)
        return {"conversations": turns}

    if format_ == "sharegpt":
        conv = record.get("conversations") or record.get("conversation", [])
        turns = sharegpt_to_conversations(conv)
        if not turns:
            return None
        return {"conversations": turns}

    return None


def main():
    ap = argparse.ArgumentParser(description="Standardize SFT data to system/user/assistant conversation format.")
    ap.add_argument("input", type=Path, help="Input JSONL or JSON file")
    ap.add_argument("output", type=Path, help="Output JSONL (one conversation object per line)")
    ap.add_argument("--format", choices=["alpaca", "sharegpt", "already_conversation"], required=True,
                    help="Input format")
    ap.add_argument("--require-assistant", action="store_true", default=True,
                    help="Skip examples with no assistant turn (default: True)")
    args = ap.parse_args()

    data = []
    if args.input.suffix.lower() == ".jsonl":
        with open(args.input) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    else:
        with open(args.input) as f:
            raw = json.load(f)
            data = raw if isinstance(raw, list) else [raw]

    out_lines = []
    for record in data:
        norm = normalize_to_conversation(record, args.format)
        if not norm:
            continue
        conv = norm["conversations"]
        if args.require_assistant and not any(t["role"] == "assistant" for t in conv):
            continue
        out_lines.append(json.dumps(norm, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"Wrote {len(out_lines)} conversations to {args.output}")


if __name__ == "__main__":
    main()
