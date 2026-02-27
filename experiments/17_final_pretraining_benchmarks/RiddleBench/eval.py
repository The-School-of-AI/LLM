import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────

@dataclass
class RiddleItem:
    id: int
    type: str       # e.g. "sequence tasks"
    question: str   # full question text including instructions
    answer: str     # gold answer string


# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset(type_filter: Optional[str] = None) -> list[RiddleItem]:
    from datasets import load_dataset as hf_load

    print("[INFO] Loading ai4bharat/RiddleBench (split: train) ...")
    ds = hf_load("ai4bharat/RiddleBench", split="train", trust_remote_code=True)

    items = []
    for row in ds:
        item = RiddleItem(
            id=row["id"],
            type=row["type"],
            question=row["question"],
            answer=str(row["answer"]).strip(),
        )
        if type_filter is None or item.type.lower() == type_filter.lower():
            items.append(item)

    if type_filter:
        print(f"[INFO] Loaded {len(items)} items (type filter: '{type_filter}').")
    else:
        print(f"[INFO] Loaded {len(items)} items across all types.")
    return items


# ─────────────────────────────────────────────
# ANSWER SCORING
# ─────────────────────────────────────────────

def normalise(s: str) -> str:
    """Normalise answer string for comparison: strip, lowercase, remove trailing zeros."""
    s = s.strip().lower()
    # Try to normalise numeric answers (e.g. "5435.0" == "5435", "1417.50" == "1417.5")
    try:
        f = float(s)
        # Use int if it's a whole number
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def is_correct(predicted: str, gold: str) -> bool:
    return normalise(predicted) == normalise(gold)


def extract_answer(raw: str) -> str:
    """
    Pull the model's final answer from its output.
    The prompt asks the model to reply with only the answer,
    so we take the last non-empty line as the answer.
    Also handles common patterns like "Answer: X" or "= X".
    """
    raw = raw.strip()

    # "Answer: X" or "ANSWER: X"
    m = re.search(r"(?:answer|result)\s*[:\-=]\s*(.+)$", raw, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).strip()

    # "= X" at end of line (common for numeric reasoning)
    m = re.search(r"=\s*([\d\.\-]+)\s*$", raw)
    if m:
        return m.group(1).strip()

    # Fall back: last non-empty line
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    return lines[-1] if lines else ""


# ─────────────────────────────────────────────
# MODEL RUNNER
# ─────────────────────────────────────────────

class ModelRunner:
    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "auto",
                 max_new_tokens: int = 64):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print(f"[INFO] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        print(f"[INFO] Model loaded on {self.device}.")

    def generate(self, item: RiddleItem) -> tuple[str, str]:
        """Returns (raw_output, extracted_answer)."""
        import torch

        # The question already embeds its own instructions (e.g. "only reply with the answer")
        # so we just pass it through as the user message.
        messages = [
            {"role": "user", "content": item.question},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            input_text = item.question

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
            )

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return raw, extract_answer(raw)


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

@dataclass
class EvalResult:
    item: RiddleItem
    predicted: str
    raw_output: str
    correct: bool
    latency_s: float


def evaluate(items: list[RiddleItem], runner: ModelRunner,
             verbose: bool = False, limit: Optional[int] = None) -> list[EvalResult]:
    results = []
    subset = items[:limit] if limit else items

    for i, item in enumerate(subset, 1):
        t0 = time.time()
        raw, pred = runner.generate(item)
        elapsed = time.time() - t0
        correct = is_correct(pred, item.answer)

        results.append(EvalResult(item=item, predicted=pred, raw_output=raw,
                                  correct=correct, latency_s=elapsed))

        status = "V" if correct else "X"
        print(f"[{i:>4}/{len(subset)}] {status}  id={item.id:<6} "
              f"type={item.type:<22}  pred={pred!r:<15}  gold={item.answer!r}  ({elapsed:.1f}s)")

        if verbose:
            print(f"        Q:   {item.question[:120]}...")
            print(f"        OUT: {raw[:200]}...\n")

    return results


# ─────────────────────────────────────────────
# SCORING & REPORT
# ─────────────────────────────────────────────

def score_report(results: list[EvalResult], model_name: str) -> dict:
    from collections import defaultdict

    total   = len(results)
    correct = sum(r.correct for r in results)
    acc     = correct / total if total else 0.0

    type_totals:  dict[str, int] = defaultdict(int)
    type_correct: dict[str, int] = defaultdict(int)
    for r in results:
        type_totals[r.item.type]  += 1
        if r.correct:
            type_correct[r.item.type] += 1

    avg_lat = sum(r.latency_s for r in results) / total if total else 0.0

    report = {
        "model":         model_name,
        "total_items":   total,
        "correct":       correct,
        "accuracy":      round(acc * 100, 2),
        "avg_latency_s": round(avg_lat, 2),
        "per_type": {
            t: {
                "correct":  type_correct[t],
                "total":    type_totals[t],
                "accuracy": round(type_correct[t] / type_totals[t] * 100, 2),
            }
            for t in sorted(type_totals)
        },
    }

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  RiddleBench -- {model_name}")
    print(f"  {total} items  |  {correct} correct  |  {acc*100:.1f}% accuracy")
    print(f"  Avg latency: {avg_lat:.2f}s / item")
    print(f"{bar}")
    print(f"  {'Type':<28} {'Correct':>7}  {'Total':>6}  {'Acc':>7}")
    print(f"  {'-'*28} {'-'*7}  {'-'*6}  {'-'*7}")
    for t, s in sorted(report["per_type"].items()):
        print(f"  {t:<28} {s['correct']:>7}  {s['total']:>6}  {s['accuracy']:>6.1f}%")
    print(f"{bar}\n")

    return report


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RiddleBench LLM Evaluation")
    p.add_argument("--model",          default=ModelRunner.DEFAULT_MODEL,
                   help="HuggingFace model name or local path")
    p.add_argument("--type",           default=None,
                   help="Filter to a specific puzzle type (e.g. 'sequence tasks')")
    p.add_argument("--limit",          type=int, default=None,
                   help="Cap number of items (useful for quick smoke tests)")
    p.add_argument("--verbose",        action="store_true",
                   help="Print question and raw model output for each item")
    p.add_argument("--output",         default="results.json",
                   help="Path for JSON results file")
    p.add_argument("--device",         default="auto")
    p.add_argument("--max-new-tokens", type=int, default=64,
                   help="Max tokens to generate (answers are short, 64 is plenty)")
    return p.parse_args()


def main():
    args = parse_args()

    items   = load_dataset(type_filter=args.type)
    runner  = ModelRunner(model_name=args.model, device=args.device,
                          max_new_tokens=args.max_new_tokens)
    results = evaluate(items, runner, verbose=args.verbose, limit=args.limit)
    report  = score_report(results, model_name=args.model)

    output = {
        "summary": report,
        "items": [
            {
                "id":         r.item.id,
                "type":       r.item.type,
                "question":   r.item.question,
                "gold":       r.item.answer,
                "predicted":  r.predicted,
                "correct":    r.correct,
                "raw_output": r.raw_output,
                "latency_s":  round(r.latency_s, 3),
            }
            for r in results
        ],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved -> {args.output}")


if __name__ == "__main__":
    main()