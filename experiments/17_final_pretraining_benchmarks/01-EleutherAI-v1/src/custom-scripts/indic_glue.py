import argparse
import sys
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def parse_model_args(args_str):
    args = {}
    if not args_str: return args
    
    # Check if it's a key-value string or just a model name
    if '=' not in args_str:
        args["pretrained"] = args_str.strip()
        return args

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

    # Representative subsets covering all 11 languages and diverse tasks
    subsets = [
        "wnli.hi", "wnli.mr", "wnli.gu", "wnli.pa", "wnli.bn", # NLI
        "copa.hi", "copa.mr", "copa.gu",                      # Causality
        "csqa.hi", "csqa.te", "csqa.ta", "csqa.kn", "csqa.as", # QA
        "csqa.ml", "csqa.or",                                 # QA
        "actsa-sc.te"                                         # Sentiment
    ]
    
    overall_correct = 0
    overall_total = 0
    subset_results = {}

    for subset in subsets:
        print(f"  [IndicGLUE] Evaluating subset: {subset}", file=sys.stderr)
        try:
            dataset = load_dataset("ai4bharat/indic_glue", subset, split="validation", trust_remote_code=True)
        except:
            try:
                dataset = load_dataset("ai4bharat/indic_glue", subset, split="train", trust_remote_code=True)
            except Exception as e:
                print(f"  [Warning] Failed to load subset {subset}: {e}", file=sys.stderr)
                continue

        if args.limit:
            dataset = dataset.select(range(min(len(dataset), args.limit)))

        correct = 0
        total = 0
        
        for item in tqdm(dataset, desc=f"Eval {subset}"):
            if "wnli" in subset or "rte" in subset or "mrpc" in subset:
                text1 = item.get('premise') or item.get('sentence1', '')
                text2 = item.get('hypothesis') or item.get('sentence2', '')
                prompt = f"Text 1: {text1}\nText 2: {text2}\nIs there a relationship? (Yes/No):"
            elif "sst" in subset or "actsa" in subset:
                text = item.get('sentence', '')
                prompt = f"Text: {text}\nIs this positive? (Yes/No):"
            elif "copa" in subset or "csqa" in subset:
                premise = item.get('premise') or item.get('question', '')
                choice1 = item.get('choice1', '')
                choice2 = item.get('choice2', '')
                prompt = f"Context: {premise}\nChoice 1: {choice1}\nChoice 2: {choice2}\nWhich is more likely? (1/2):"
            else:
                continue

            label = item.get('label')
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=5)
            
            gen = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].lower()
            
            if "copa" in subset:
                pred = 0 if "1" in gen else 1
            else:
                pred = 1 if "yes" in gen else 0
                
            if pred == label:
                correct += 1
            total += 1

        if total > 0:
            subset_score = correct / total
            subset_results[subset] = subset_score
            overall_correct += correct
            overall_total += total

    score = overall_correct / overall_total if overall_total > 0 else 0
    
    print(json.dumps({
        "name": "IndicGLUE",
        "status": "success",
        "score": score,
        "subtasks": [{"task": k, "score": v} for k, v in subset_results.items()],
        "details": f"Ran on {len(subset_results)} subsets"
    }))

if __name__ == "__main__":
    main()
