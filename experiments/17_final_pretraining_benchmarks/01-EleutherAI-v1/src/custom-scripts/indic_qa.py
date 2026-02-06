import argparse
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

    # Load IndicQA
    try:
        # ai4bharat/IndicQA. Using 'indicqa.hi' (Hindi). Note: Only 'test' split available.
        dataset = load_dataset("ai4bharat/IndicQA", "indicqa.hi", split="test", trust_remote_code=True)
    except Exception as e:
        print(json.dumps({"name": "IndicQA", "status": "failed", "error": f"Dataset load failed (ai4bharat/IndicQA): {str(e)}"}))
        return

    if args.limit:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    # Metrics
    try:
        squad_metric = evaluate.load("squad") # Standard F1/EM
    except:
        squad_metric = None

    predictions = []
    references = []

    for item in tqdm(dataset, desc="Eval IndicQA"):
        context = item.get('context', '')
        question = item.get('question', '')
        answers = item.get('answers', {}) # Dict with 'text' and 'answer_start'
        id_ = str(item.get('id', ''))
        
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # Limit context len
        if inputs['input_ids'].shape[1] > 2048:
             continue # Skip long contexts for simple script

        with torch.no_grad():
             outputs = model.generate(**inputs, max_new_tokens=32)
        
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        
        predictions.append({'prediction_text': gen, 'id': id_})
        references.append({'answers': answers, 'id': id_})

    score = 0.0
    if squad_metric and predictions:
        results = squad_metric.compute(predictions=predictions, references=references)
        score = results.get('f1', 0.0)

    print(json.dumps({
        "name": "IndicQA",
        "status": "success",
        "score": score,
        "details": "F1 Score on Hindi subset"
    }))

if __name__ == "__main__":
    main()
