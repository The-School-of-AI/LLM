import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
        print(json.dumps({"name": "IndicGLUE", "status": "failed", "error": "No pretrained model specified"}))
        return
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"name": "IndicGLUE", "status": "failed", "error": str(e)}))
        return

    # Load IndicGLUE (subset: wnli.hi or similar as default)
    try:
        dataset = load_dataset("ai4bharat/indic_glue", "wnli.hi", split="validation", trust_remote_code=True)
    except Exception as e:
        try:
             # Fallback
             dataset = load_dataset("ai4bharat/indic_glue", "wnli.hi", split="train", trust_remote_code=True)
        except Exception as e:
             print(json.dumps({"name": "IndicGLUE", "status": "failed", "error": f"Dataset load failed (ai4bharat/indic_glue): {str(e)}"}))
             return

    if args.limit:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    correct = 0
    total = 0

    # WNLI: Binary classification usually.
    # Simplified evaluation: Check if model generates the correct entailment label (0 or 1, or text)
    
    for item in tqdm(dataset, desc="Eval IndicGLUE"):
        premise = item.get('premise', '')
        hypothesis = item.get('hypothesis', '')
        label = item.get('label') # 0 or 1
        
        prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? (Yes/No):"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
             outputs = model.generate(**inputs, max_new_tokens=5)
        
        gen = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].lower()
        
        # Heuristic mapping
        pred = 1 if "yes" in gen else 0
        if pred == label:
            correct += 1
        total += 1

    score = correct / total if total > 0 else 0
    
    print(json.dumps({
        "name": "IndicGLUE",
        "status": "success",
        "score": score,
        "details": f"Ran on wnli.hi ({total} samples)"
    }))

if __name__ == "__main__":
    main()
