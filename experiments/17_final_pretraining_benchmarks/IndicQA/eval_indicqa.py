import argparse
import json
import os
import re
import requests
import unicodedata
from collections import defaultdict
import numpy as np
import random

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

_LANG = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/data/"
CACHE_DIR = "./cache/indicqa"
random.seed(42)

def download_file(url, save_path):
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)


# =========================
# 1. NORMALIZATION
# =========================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = " ".join(text.split())
    return text


# =========================
# 2. METRICS
# =========================

def compute_em(pred, gold_list):
    pred_norm = normalize_text(pred)
    return max(int(pred_norm == normalize_text(g)) for g in gold_list)


def compute_f1(pred, gold_list):
    pred_tokens = normalize_text(pred).split()

    best_f1 = 0
    for gold in gold_list:
        gold_tokens = normalize_text(gold).split()

        common = set(pred_tokens) & set(gold_tokens)
        if len(common) == 0:
            continue

        precision = len(common) / max(len(pred_tokens), 1)
        recall = len(common) / max(len(gold_tokens), 1)

        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def copy_ratio(pred, context):
    pred_tokens = normalize_text(pred).split()
    context_tokens = set(normalize_text(context).split())

    if len(pred_tokens) == 0:
        return 0.0

    overlap = [t for t in pred_tokens if t in context_tokens]
    return len(overlap) / len(pred_tokens)


def first_token_accuracy(pred, gold_list):
    pred_tokens = normalize_text(pred).split()
    if len(pred_tokens) == 0:
        return 0

    first_pred = pred_tokens[0]

    for gold in gold_list:
        gold_tokens = normalize_text(gold).split()
        if len(gold_tokens) > 0 and first_pred == gold_tokens[0]:
            return 1

    return 0


def token_density(text, tokenizer):
    tokens = tokenizer(text)["input_ids"]
    return len(tokens) / max(len(text), 1)


def get_position_bucket(answer_starts, context):
    if not answer_starts:
        return "unknown"

    start = answer_starts[0]  # take first gold span

    if start is None:
        return "unknown"

    norm_pos = start / max(len(context), 1)

    if norm_pos < 0.33:
        return "early"
    elif norm_pos < 0.66:
        return "middle"
    else:
        return "late"


def analyze_answer_lengths(dataset):

    lengths = []
    per_lang = defaultdict(list)

    for ex in dataset:
        golds = ex["answers"]["text"]
        avg_len = sum(len(g.split()) for g in golds) / len(golds)
        lengths.append(avg_len)
        per_lang[ex["language"]].append(avg_len)

    print("Overall Avg Gold Length:", round(np.mean(lengths), 2))
    print("Overall Max Gold Length:", max(lengths))
    print("Overall 95th percentile:", np.percentile(lengths, 95))

    print("\nPer Language:")
    for lang in per_lang:
        print(
            lang,
            "avg:", round(np.mean(per_lang[lang]), 2),
            "max:", max(per_lang[lang])
        )

# =========================
# 3. PROMPT FORMAT
# =========================

def build_prompt(context, question):
    return f"""Context:
{context}

Question:
{question}

Answer:"""


# =========================
# 4. MODEL AND DATASET LOADING
# =========================

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    EDIT THIS FUNCTION to match your infra.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tokenizer_path:
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    model.eval()
    return model, tokenizer


def load_indicqa_all_languages(force_download=False, selected_langs=None, max_samples_per_lang=None):

    os.makedirs(CACHE_DIR, exist_ok=True)
    langs_to_load = selected_langs if selected_langs else _LANG

    all_examples = []

    for lang in langs_to_load:
        filename = f"indicqa.{lang}.json"
        cache_path = os.path.join(CACHE_DIR, filename)
        url = BASE_URL + filename

        # Download only if needed
        if force_download or not os.path.exists(cache_path):
            print(f"Downloading {lang}...")
            download_file(url, cache_path)
        else:
            print(f"Loading cached {lang}...")

        # Load JSON from disk
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()

                for qa in paragraph["qas"]:

                    if max_samples_per_lang and count >= max_samples_per_lang:
                        break

                    question = qa["question"].strip()
                    answers = []
                    answer_starts = []

                    for a in qa["answers"]:
                        text = a.get("text", "").strip()
                        start = a.get("answer_start", None)

                        # Only keep answers with valid integer start
                        if text and isinstance(start, int):
                            answers.append(text)
                            answer_starts.append(start)

                    # Skip example if no valid answers
                    if not answers:
                        continue

                    all_examples.append({
                        "context": context,
                        "question": question,
                        "answers": {"text": answers},
                        "answer_starts": answer_starts,
                        "language": lang
                    })

                    count += 1

                if max_samples_per_lang and count >= max_samples_per_lang:
                    break

            if max_samples_per_lang and count >= max_samples_per_lang:
                break

    random.shuffle(all_examples)
    return Dataset.from_list(all_examples)

# =========================
# 5. GENERATION
# =========================

def generate_batch(model, tokenizer, prompts, max_new_tokens=8):

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            use_cache=True
        )

    preds = []

    input_lengths = inputs["input_ids"].shape[1]

    for i in range(len(prompts)):
        gen_tokens = outputs[i][input_lengths:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        preds.append(text.strip().split("\n")[0])

    return preds

# =========================
# 6. MAIN EVAL LOOP
# =========================

def evaluate(
    model,
    tokenizer,
    split,
    max_samples=None,
    force_download=False,
    languages=None,
    batch_size=4,
    max_new_tokens=8
):

    dataset = load_indicqa_all_languages(
        force_download=force_download,
        selected_langs=languages,
        max_samples_per_lang=max_samples
    )
    print(dataset[0])
    for i in range(5):
        print(dataset[i]["answer_starts"])
    analyze_answer_lengths(dataset)

    results = defaultdict(lambda: {
        "em": [],
        "f1": [],
        "copy_ratio": [],
        "first_token_acc": [],
        "pred_len": [],
        "gold_len": [],
        "empty": [],
        "token_density": []
    })

    position_stats = defaultdict(lambda: {
        "early": [],
        "middle": [],
        "late": []
    })
    hallucinations = defaultdict(int)
    total_per_lang = defaultdict(int)

    # -------------------------------
    # Collect valid examples first
    # -------------------------------

    filtered_examples = []
    dataset = sorted(dataset, key=lambda x: len(tokenizer(x["context"])["input_ids"]))
    for example in dataset:
        # lang = example["language"]
        filtered_examples.append(example)

    # -------------------------------
    # Batched Evaluation
    # -------------------------------
    # lengths = [len(tokenizer(p)["input_ids"]) for p in prompts]
    # print("Max length in batch:", max(lengths))

    # for start_idx in tqdm(range(0, len(filtered_examples), batch_size)):
    pbar = tqdm(
        range(0, len(filtered_examples), batch_size),
        desc="Evaluating",
        dynamic_ncols=True
    )

    for start_idx in pbar:
        batch = filtered_examples[start_idx:start_idx + batch_size]

        prompts = []
        contexts = []
        golds = []
        langs = []
        answer_starts_batch = []

        for example in batch:
            context = example["context"]
            question = example["question"]

            prompts.append(build_prompt(context, question))
            contexts.append(context)
            golds.append(example["answers"]["text"])
            langs.append(example["language"])
            answer_starts_batch.append(example["answer_starts"])

        preds = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens
        )

        # --------------------------------
        # Process metrics per example
        # --------------------------------

        for i in range(len(preds)):

            pred = preds[i]
            context = contexts[i]
            gold_answers = golds[i]
            language = langs[i]
            answer_starts = answer_starts_batch[i]

            em = compute_em(pred, gold_answers)
            f1 = compute_f1(pred, gold_answers)

            results[language]["em"].append(em)
            results[language]["f1"].append(f1)

            bucket = get_position_bucket(answer_starts, context)
            if bucket in ["early", "middle", "late"]:
                position_stats[language][bucket].append(f1)

            cr = copy_ratio(pred, context)
            results[language]["copy_ratio"].append(cr)

            if cr < 0.5:
                hallucinations[language] += 1

            results[language]["first_token_acc"].append(
                first_token_accuracy(pred, gold_answers)
            )

            results[language]["pred_len"].append(len(pred.split()))

            avg_gold_len = sum(len(g.split()) for g in gold_answers) / max(len(gold_answers), 1)
            results[language]["gold_len"].append(avg_gold_len)

            results[language]["empty"].append(int(len(pred.strip()) == 0))

            results[language]["token_density"].append(
                token_density(context, tokenizer)
            )

            total_per_lang[language] += 1

        # Update live language stats
        total_done = sum(total_per_lang.values())
        lang_counts_str = ", ".join(
            f"{lang}:{total_per_lang[lang]}"
            for lang in sorted(total_per_lang.keys())
        )
        pbar.set_postfix_str(lang_counts_str)

    # -------------------------------
    # Aggregation
    # -------------------------------

    summary = {}
    pos_summary = {}

    for lang in results:
        total = len(results[lang]["em"])

        summary[lang] = {
            "EM": round(sum(results[lang]["em"]) / total * 100, 2),
            "F1": round(sum(results[lang]["f1"]) / total * 100, 2),
            "HallucinationRate": round(hallucinations[lang] / total_per_lang[lang] * 100, 2),
            "AvgPredLen": round(sum(results[lang]["pred_len"]) / total, 2),
            "AvgGoldLen": round(sum(results[lang]["gold_len"]) / total, 2),
            "CopyRatio": round(sum(results[lang]["copy_ratio"]) / total * 100, 2),
            "FirstTokenAcc": round(sum(results[lang]["first_token_acc"]) / total * 100, 2),
            "EmptyAnswerRate": round(sum(results[lang]["empty"]) / total * 100, 2),
            "AvgTokensPerChar": round(sum(results[lang]["token_density"]) / total, 4),
            "Samples": total
        }

    for lang in position_stats:
        pos_summary[lang] = {}
        for bucket in ["early", "middle", "late"]:
            values = position_stats[lang][bucket]
            pos_summary[lang][bucket] = (
                round(sum(values) / len(values) * 100, 2)
                if values else None
            )

    for lang in summary:
        summary[lang]["PositionF1"] = pos_summary.get(
            lang,
            {"early": None, "middle": None, "late": None}
        )

    return summary

# =========================
# 7. ENTRY POINT
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigscience/bloom-560m")
    parser.add_argument("--tokenizer_path", type=str, default="") #../tsai_131k_tokenizer
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default="indicqa_results.json")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="List of languages to evaluate (e.g., hi ta ml)"
    )

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    summary = evaluate(
        model,
        tokenizer,
        split=args.split,
        max_samples=args.max_samples,
        force_download=args.force_download,
        languages=args.languages
    )

    print("\n===== IndicQA Results =====")
    for lang, stats in summary.items():
        print(f"{lang}: {stats}")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()