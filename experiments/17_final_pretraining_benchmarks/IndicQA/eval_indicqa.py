import argparse
import json
import os
import re
import requests
import unicodedata
from collections import defaultdict

import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm

_LANG = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicQA/resolve/main/data/"
CACHE_DIR = "./cache/indicqa"


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


def load_indicqa_all_languages(force_download=False):

    os.makedirs(CACHE_DIR, exist_ok=True)

    all_examples = []

    for lang in _LANG:
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

        for article in data["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()

                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    answers = [a["text"].strip() for a in qa["answers"]]

                    all_examples.append({
                        "context": context,
                        "question": question,
                        "answers": {"text": answers},
                        "language": lang
                    })

    return Dataset.from_list(all_examples)

# =========================
# 5. GENERATION
# =========================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=8):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][input_ids.shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    # stop at first newline
    text = text.strip().split("\n")[0]
    return text.strip()


# =========================
# 6. MAIN EVAL LOOP
# =========================

def evaluate(model, tokenizer, split, max_samples=None):

    dataset = load_indicqa_all_languages(force_download=False)
    print(dataset[0])

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

    hallucinations = defaultdict(int)
    total_per_lang = defaultdict(int)

    for i, example in enumerate(tqdm(dataset)):

        if max_samples and i >= max_samples:
            break

        context = example["context"]
        question = example["question"]
        gold_answers = example["answers"]["text"]
        language = example["language"]

        prompt = build_prompt(context, question)
        pred = generate_answer(model, tokenizer, prompt)
        # pred = pred.split(".")[0]

        print(f"#{i}:  Context={context}, question={question}, gold={gold_answers}, language={language}, pred={pred}")

        em = compute_em(pred, gold_answers)
        f1 = compute_f1(pred, gold_answers)
        results[language]["em"].append(em)
        results[language]["f1"].append(f1)

        # Additional diagnostics
        results[language]["copy_ratio"].append(copy_ratio(pred, context))
        results[language]["first_token_acc"].append(first_token_accuracy(pred, gold_answers))
        results[language]["pred_len"].append(len(pred.split()))

        avg_gold_len = sum(len(g.split()) for g in gold_answers) / max(len(gold_answers), 1)
        results[language]["gold_len"].append(avg_gold_len)

        results[language]["empty"].append(int(len(pred.strip()) == 0))
        results[language]["token_density"].append(token_density(context, tokenizer))

        total_per_lang[language] += 1

        # Hallucination tracking
        # if pred not in context:
        if copy_ratio(pred, context) < 0.5:
            hallucinations[language] += 1

    # Aggregate
    summary = {}

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
    # parser.add_argument("--force_download", action="store_true")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path)

    summary = evaluate(
        model,
        tokenizer,
        split=args.split,
        max_samples=args.max_samples
    )

    print("\n===== IndicQA Results =====")
    for lang, stats in summary.items():
        print(f"{lang}: {stats}")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()