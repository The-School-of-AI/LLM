import os
import re
import json
import time
import argparse
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

INDIC_LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada",
    "Malayalam", "Marathi", "Odia", "Punjabi",
    "Tamil", "Telugu",
]

BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
RESULTS_DIR = "results"

ANSWER_MAP = {
    "option1": "A",
    "option2": "B",
    "option3": "C",
    "option4": "D",
}

# ─────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────

def load_language(language, limit=None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN=your_token")

    dataset = load_dataset(
        "ai4bharat/MILU",
        data_dir=language,
        split="test",
        token=token,
    )

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    return dataset

# ─────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────

def build_prompt(row):
    return (
        f"Question: {row['question']}\n"
        f"A) {row['option1']}\n"
        f"B) {row['option2']}\n"
        f"C) {row['option3']}\n"
        f"D) {row['option4']}\n"
        f"Answer:"
    )

# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────

def load_model(model_name, device="auto"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    print("Model loaded.\n")
    return model, tokenizer


def run_batch(model, tokenizer, prompts):
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i, output in enumerate(outputs):
        input_len = inputs["input_ids"][i].shape[0]
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True).strip()
        results.append(text)

    return results

# ─────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────

def extract_answer(raw):
    match = re.search(r"\b([A-D])\b", raw.strip())
    if match:
        return match.group(1)
    match = re.search(r"Answer[:\s]+([A-D])", raw, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def score_answer(predicted, gold_option):
    return predicted == ANSWER_MAP.get(gold_option)

# ─────────────────────────────────────────
# REPORTER
# ─────────────────────────────────────────

def save_language_results(language, summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{language}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved → {path}")


def print_language_summary(summary):
    lang = summary["language"]
    print(f"\n{'='*55}")
    print(f"  {lang}")
    print(f"  {summary['total']} items | {summary['correct']} correct | {summary['accuracy']:.1f}% accuracy")
    print(f"  Avg latency: {summary['avg_latency_s']:.3f}s / item")
    print(f"  Subject Breakdown:")
    for subject, stats in summary.get("per_subject", {}).items():
        print(f"    {subject:<35} {stats['correct']:>3}/{stats['total']:<5} {stats['accuracy']:.1f}%")
    print(f"{'='*55}")


def save_final_summary(model_name, all_summaries):
    total_correct = sum(s["correct"] for s in all_summaries.values())
    total_items = sum(s["total"] for s in all_summaries.values())
    overall = round(total_correct / total_items * 100, 2) if total_items > 0 else 0

    print(f"\n{'='*55}")
    print(f"  MILU-IN FINAL SUMMARY")
    print(f"  Model: {model_name}")
    print(f"{'='*55}")
    for language, summary in all_summaries.items():
        print(f"  {language:<20} {summary['accuracy']:.1f}%")
    print(f"  {'─'*40}")
    print(f"  {'Overall':<20} {overall:.1f}%")
    print(f"{'='*55}\n")

    per_language_clean = {
        lang: {
            "language": s["language"],
            "model": s["model"],
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": s["accuracy"],
            "avg_latency_s": s["avg_latency_s"],
        }
        for lang, s in all_summaries.items()
    }

    path = os.path.join(RESULTS_DIR, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "overall_accuracy": overall,
            "total_correct": total_correct,
            "total_items": total_items,
            "per_language": per_language_clean,
        }, f, ensure_ascii=False, indent=2)
    print(f"Final summary saved → {path}")

# ─────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────

def evaluate_language(language, model, tokenizer, limit=None):
    print(f"Evaluating: {language}")
    dataset = load_language(language, limit=limit)

    per_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_latency = 0.0
    wrong_predictions = []

    for batch_start in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[batch_start: batch_start + BATCH_SIZE]
        n = len(batch["question"])

        prompts = [
            build_prompt({
                "question": batch["question"][i],
                "option1": batch["option1"][i],
                "option2": batch["option2"][i],
                "option3": batch["option3"][i],
                "option4": batch["option4"][i],
            })
            for i in range(n)
        ]

        t0 = time.time()
        raw_outputs = run_batch(model, tokenizer, prompts)
        elapsed = (time.time() - t0) / n
        total_latency += elapsed * n

        for i, raw in enumerate(raw_outputs):
            predicted = extract_answer(raw)
            gold = batch["target"][i]
            correct = score_answer(predicted, gold) if predicted else False
            subject = batch["subject"][i] if "subject" in batch else "Unknown"

            if correct:
                total_correct += 1
            else:
                wrong_predictions.append({
                    "question": batch["question"][i],
                    "predicted": predicted,
                    "gold": ANSWER_MAP.get(gold),
                    "subject": subject,
                })

            per_subject[subject]["total"] += 1
            if correct:
                per_subject[subject]["correct"] += 1

        print(f"  [{min(batch_start + BATCH_SIZE, len(dataset))}/{len(dataset)}]", end="\r")

    total = len(dataset)
    accuracy = round(total_correct / total * 100, 2) if total > 0 else 0
    avg_latency = round(total_latency / total, 3) if total > 0 else 0

    per_subject_summary = {
        subj: {
            "correct": v["correct"],
            "total": v["total"],
            "accuracy": round(v["correct"] / v["total"] * 100, 1) if v["total"] > 0 else 0,
        }
        for subj, v in per_subject.items()
    }

    summary = {
        "language": language,
        "model": tokenizer.name_or_path,
        "total": total,
        "correct": total_correct,
        "accuracy": accuracy,
        "avg_latency_s": avg_latency,
        "per_subject": per_subject_summary,
        "wrong_predictions": wrong_predictions,
    }

    save_language_results(language, summary)
    print_language_summary(summary)

    return summary

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MILU-IN Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    languages = [args.language] if args.language else INDIC_LANGUAGES
    all_summaries = {}

    for language in languages:
        summary = evaluate_language(language, model, tokenizer, limit=args.limit)
        all_summaries[language] = summary

    if len(languages) > 1:
        save_final_summary(args.model, all_summaries)