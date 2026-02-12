import argparse
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import evaluate

def parse_model_args(args_str):
    args = {}
    if not args_str: return args
    for part in args_str.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            args[k.strip()] = v.strip()
    return args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args, unknown = parser.parse_known_args()

    # Load Model (Copy-paste boilerplate)
    model_args_dict = parse_model_args(args.model_args)
    model_name = model_args_dict.get("pretrained")
    if not model_name:
        print(json.dumps({"name": "IndicQA", "status": "failed", "error": "No pretrained model specified"}))
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"name": "IndicQA", "status": "failed", "error": str(e)}))
        return

    # Coverage for all 10 supported languages in IndicQA
    languages = [
        "indicqa.hi", "indicqa.bn", "indicqa.ta", "indicqa.te", 
        "indicqa.ml", "indicqa.mr", "indicqa.gu", "indicqa.kn", 
        "indicqa.pa", "indicqa.as"
    ]
    lang_results = {}
    overall_f1 = 0
    lang_count = 0

    # Metrics
    try:
        squad_metric = evaluate.load("squad") # Standard F1/EM
    except Exception as e:
        print(f"  [Warning] SQuAD metric load failed: {e}. Using fallback F1.", file=sys.stderr)
        squad_metric = None

    def calculate_f1(prediction, reference):
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        common = set(pred_tokens) & set(ref_tokens)
        num_common = len(common)
        if num_common == 0: return 0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    for lang in languages:
        print(f"  [IndicQA] Evaluating language: {lang}", file=sys.stderr)
        try:
            dataset = load_dataset("ai4bharat/IndicQA", lang, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  [Warning] Dataset load failed for {lang}: {e}", file=sys.stderr)
            continue

        if args.limit:
            dataset = dataset.select(range(min(len(dataset), args.limit)))

        predictions = []
        references = []
        lang_scores = []

        for item in tqdm(dataset, desc=f"Eval {lang}"):
            id_ = str(item.get('id', ''))
            context = item.get('context', '')
            question = item.get('question', '')
            answers = item.get('answers', {})
            
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            # IndicQA contexts are often long; increase limit to 8k
            if inputs['input_ids'].shape[1] > 8192: continue

            with torch.no_grad():
                 outputs = model.generate(**inputs, max_new_tokens=32)
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
            
            if squad_metric:
                predictions.append({'prediction_text': gen, 'id': id_})
                references.append({'answers': answers, 'id': id_})
            else:
                # Fallback word-level F1
                if answers.get('text'):
                    f1 = calculate_f1(gen, answers['text'][0])
                    lang_scores.append(f1)

        if squad_metric and predictions:
            results = squad_metric.compute(predictions=predictions, references=references)
            f1 = (results.get('f1', 0.0) / 100.0) if results else 0.0
            print(f"  [Debug] {lang} SQuAD F1: {f1}", file=sys.stderr)
            lang_results[lang] = f1
            overall_f1 += f1
            lang_count += 1
        elif lang_scores:
            f1 = sum(lang_scores) / len(lang_scores)
            print(f"  [Debug] {lang} Fallback F1: {f1}", file=sys.stderr)
            lang_results[lang] = f1
            overall_f1 += f1
            lang_count += 1
        else:
            print(f"  [Debug] {lang} No scores calculated. Predictions: {len(predictions)}, Scores list: {len(lang_scores)}", file=sys.stderr)

    score = overall_f1 / lang_count if lang_count > 0 else 0
    
    print(json.dumps({
        "name": "IndicQA",
        "status": "success",
        "score": score,
        "subtasks": [{"task": k, "score": v} for k, v in lang_results.items()],
        "details": f"Ran on {len(lang_results)} languages"
    }))

if __name__ == "__main__":
    main()
