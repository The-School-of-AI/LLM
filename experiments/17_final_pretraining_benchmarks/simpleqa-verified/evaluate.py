#!/usr/bin/env python3
"""
SimpleQA Verified Benchmark Evaluation Script

Evaluates an LLM against the google/simpleqa-verified dataset using:
  - vLLM for fast batched inference
  - Google Gemini as LLM judge (correct / incorrect / not_attempted)

Usage:
    python evaluate.py --model meta-llama/Llama-3.2-1B-Instruct
    python evaluate.py --model ./my-local-checkpoint --dry-run
    python evaluate.py --model ./my-local-checkpoint --max-samples 100 --output results.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Gemini grader — prompt and parsing (adapted from sample-reference.py)
# ---------------------------------------------------------------------------

GRADER_TEMPLATE = """
  Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
  First, I will give examples of each grade, and then you will grade a new example.


  The following are examples of CORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia Obama and Sasha Obama
  Predicted answer 1: sasha and malia obama
  Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
  Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
  Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
  ```
  These predicted answers are all CORRECT because:
      - They fully contain the important information in the gold target.
      - They do not contain any information that contradicts the gold target.
      - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
      - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
      - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


  The following are examples of INCORRECT predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: Malia.
  Predicted answer 2: Malia, Sasha, and Susan.
  Predicted answer 3: Barack Obama does not have any children.
  Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
  Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
  Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
  ```
  These predicted answers are all INCORRECT because:
      - A factual statement in the answer contradicts the gold target.
      - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


  The following are examples of NOT_ATTEMPTED predicted answers.
  ```
  Question: What are the names of Barack Obama's children?
  Gold target: Malia and Sasha
  Predicted answer 1: I don't know.
  Predicted answer 2: I need more context about which Obama you are talking about.
  Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
  Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
  Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
  Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
  Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
  ```
  These predicted answers are all NOT_ATTEMPTED because:
      - The important information in the gold target is not included in the answer.
      - No statements in the answer contradict the gold target.
      - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


  Also note the following things:
  - For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k (acceptable range: anything between 118k and 122k)".
      - Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall within the range specified in the gold target.
      - Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the range specified in the gold target.
      - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
  - The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
      - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
  - Do not punish predicted answers if they omit information that would be clearly inferred from the question.
      - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
      - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
      - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer "1.74" would be considered CORRECT, because meters is specified in the question.
      - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
  - Do not punish for typos in people's name if it's clearly the same name.
      - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


  Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
  ```
  Question: {question}
  Gold target: {target}
  Predicted answer: {predicted_answer}
  ```

  Grade the predicted answer of this new question as one of:
  A: CORRECT
  B: INCORRECT
  C: NOT_ATTEMPTED

  Just return the letters "A", "B", or "C", with no text around it.
""".strip()

CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))
DEFAULT_GRADE_IF_UNPARSEABLE = "C"  # NOT_ATTEMPTED


def format_grader_prompt(question: str, target: str, predicted_answer: str) -> str:
    return GRADER_TEMPLATE.format(
        question=question,
        target=target,
        predicted_answer=predicted_answer,
    )


def parse_grade_letter(response_text: str) -> str:
    """Extract A/B/C grade letter from Gemini response, defaulting to C."""
    match = re.search(r"\b(A|B|C)\b", response_text)
    if match:
        return match.group(0)
    upper = response_text.upper()
    if "CORRECT" in upper and "INCORRECT" not in upper and "NOT" not in upper:
        return "A"
    if "INCORRECT" in upper:
        return "B"
    if "NOT_ATTEMPTED" in upper or "NOT ATTEMPTED" in upper:
        return "C"
    print(
        f"  [warn] Could not parse grade from: '{response_text}'. Defaulting to {DEFAULT_GRADE_IF_UNPARSEABLE}."
    )
    return DEFAULT_GRADE_IF_UNPARSEABLE


def grade_with_gemini(
    google_client: genai.Client,
    gemini_model_name: str,
    question: str,
    target: str,
    predicted_answer: str,
    retries: int = 3,
) -> tuple[str, str]:
    """Returns (grade_letter, grade_str)."""
    prompt = format_grader_prompt(question, target, predicted_answer)
    for attempt in range(retries):
        try:
            response = google_client.models.generate_content(
                model=gemini_model_name,
                contents=prompt,
            )
            grade_letter = parse_grade_letter(response.text.strip())
            grade_str = CHOICE_LETTER_TO_STRING.get(grade_letter, "NOT_ATTEMPTED")
            return grade_letter, grade_str
        except Exception as e:
            if attempt == retries - 1:
                print(f"  [warn] Gemini grading failed after {retries} attempts: {e}")
                return DEFAULT_GRADE_IF_UNPARSEABLE, "NOT_ATTEMPTED"
            time.sleep(2**attempt)
    return DEFAULT_GRADE_IF_UNPARSEABLE, "NOT_ATTEMPTED"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question with a short, direct answer. "
    "Do not include explanations or hedging — just state the answer."
)


def build_prompt(question: str) -> str:
    """Build a simple instruction prompt for the model."""
    return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question}\n<|assistant|>\n"


def run_inference(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    device: str,
) -> list[str]:
    """Run batched HF Transformers inference, return list of generated answers."""
    prompts = [build_prompt(q) for q in questions]
    outputs = []
    total = len(prompts)

    do_sample = temperature > 0

    for start in range(0, total, batch_size):
        batch = prompts[start : start + batch_size]
        print(
            f"  Generating batch {start // batch_size + 1} / "
            f"{(total + batch_size - 1) // batch_size} "
            f"({start + 1}-{min(start + batch_size, total)} of {total})"
        )

        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract only the newly generated tokens
        input_len = inputs.input_ids.shape[1]
        batch_outputs = tokenizer.batch_decode(
            generated_ids[:, input_len:], skip_special_tokens=True
        )

        for text in batch_outputs:
            outputs.append(text.strip())

    return outputs


# ---------------------------------------------------------------------------
# Metrics (formulas from sample-reference.py)
# ---------------------------------------------------------------------------


def get_accuracy_given_attempted(correct: int, incorrect: int) -> float:
    attempted = correct + incorrect
    if attempted == 0:
        return 0.0
    return correct / attempted


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    incorrect = sum(1 for r in results if r["is_incorrect"])
    not_attempted = sum(1 for r in results if r["is_not_attempted"])

    mean_correct = correct / total if total else 0.0
    accuracy_given_attempted = get_accuracy_given_attempted(correct, incorrect)

    numerator = 2 * accuracy_given_attempted * mean_correct
    denominator = accuracy_given_attempted + mean_correct
    f1 = numerator / denominator if denominator else 0.0

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "not_attempted": not_attempted,
        "accuracy": round(mean_correct * 100, 2),
        "accuracy_given_attempted": round(accuracy_given_attempted * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }


def print_summary(metrics: dict, model_name: str, elapsed: float) -> None:
    sep = "=" * 55
    print(f"\n{sep}")
    print("  SimpleQA Verified — Evaluation Summary")
    print(sep)
    print(f"  Model                    : {model_name}")
    print(f"  Total samples            : {metrics['total']}")
    print(
        f"  Correct                  : {metrics['correct']}  ({metrics['accuracy']:.2f}%)"
    )
    print(f"  Incorrect                : {metrics['incorrect']}")
    print(f"  Not attempted            : {metrics['not_attempted']}")
    print(f"  Accuracy (given attempt) : {metrics['accuracy_given_attempted']:.2f}%")
    print(f"  F1 score                 : {metrics['f1_score']:.2f}%")
    print(f"  Elapsed time             : {elapsed:.1f}s")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an LLM on google/simpleqa-verified benchmark"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID or path to local checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for HF backend (default: cuda if available)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash",
        help="Gemini model to use for grading (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation to N samples (default: all 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Generation batch size (default: 32)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per answer (default: 128)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to save JSON results (default: results.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run on 5 examples only; do not save output file",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Google Gemini API key (or set GEMINI_API_KEY env var)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.time()

    # --- Gemini setup ---
    api_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "ERROR: Gemini API key required. Set GEMINI_API_KEY env var or use --gemini-api-key.",
            file=sys.stderr,
        )
        sys.exit(1)
    google_client = genai.Client(api_key=api_key)
    print(f"[grader] Using Gemini model: {args.gemini_model}")

    # --- Load dataset ---
    print("[dataset] Loading google/simpleqa-verified from HuggingFace ...")
    dataset = load_dataset("google/simpleqa-verified", split="eval")
    print(f"[dataset] Loaded {len(dataset)} examples")

    # Determine sample count
    if args.dry_run:
        n_samples = 5
        print("[dry-run] Limiting to 5 examples")
    elif args.max_samples is not None:
        n_samples = min(args.max_samples, len(dataset))
    else:
        n_samples = len(dataset)

    dataset = dataset.select(range(n_samples))
    questions = dataset["problem"]
    reference_answers = dataset["answer"]
    original_indices = dataset["original_index"]

    # --- Inference ---
    print(f"[model] Loading {args.model} with HF Transformers ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map=args.device,
        trust_remote_code=True,
    )
    print(f"[inference] Generating answers for {n_samples} questions with HF ...")
    model_answers = run_inference(
        model,
        tokenizer,
        questions,
        args.max_new_tokens,
        args.temperature,
        args.batch_size,
        args.device,
    )

    # --- Grading ---
    print(
        f"[grading] Grading {n_samples} answers with Gemini ({args.gemini_model}) ..."
    )
    results = []
    for i, (q, ref, pred, idx) in enumerate(
        zip(questions, reference_answers, model_answers, original_indices)
    ):
        grade_letter, grade_str = grade_with_gemini(
            google_client, args.gemini_model, q, ref, pred
        )
        results.append(
            {
                "original_index": idx,
                "question": q,
                "gold_target": ref,
                "predicted_answer": pred,
                "grade_letter": grade_letter,
                "grade_str": grade_str,
                "is_correct": grade_letter == "A",
                "is_incorrect": grade_letter == "B",
                "is_not_attempted": grade_letter == "C",
            }
        )
        if (i + 1) % 50 == 0 or (i + 1) == n_samples:
            print(f"  Graded {i + 1}/{n_samples} ...")

    # --- Metrics ---
    metrics = compute_metrics(results)
    elapsed = time.time() - start_time
    print_summary(metrics, args.model, elapsed)

    # --- Save output ---
    if not args.dry_run:
        output = {
            "metadata": {
                "model": args.model,
                "gemini_grader": args.gemini_model,
                "dataset": "google/simpleqa-verified",
                "n_samples": n_samples,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "elapsed_seconds": round(elapsed, 1),
            },
            "metrics": metrics,
            "results": results,
        }
        out_path = Path(args.output)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"\n[output] Results saved to: {out_path.resolve()}")
    else:
        print("\n[dry-run] Skipping file output.")


if __name__ == "__main__":
    main()
