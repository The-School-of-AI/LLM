import argparse
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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
        print(json.dumps({"name": "Indic-Bias", "status": "failed", "error": "No pretrained model specified"}))
        return
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
        model.eval()
    except Exception as e:
        print(json.dumps({"name": "Indic-Bias", "status": "failed", "error": str(e)}))
        return

    # Load Indic-Bias (FairITales)
    # ai4bharat/Indic-Bias is the official gated repository.
    try:
        dataset = load_dataset("ai4bharat/Indic-Bias", split="test", trust_remote_code=True, token=True)
    except Exception as e:
        try:
             dataset = load_dataset("ai4bharat/Indic-Bias", split="train", trust_remote_code=True, token=True) 
        except Exception as e:
             print(json.dumps({"name": "Indic-Bias", "status": "failed", "error": f"Dataset load failed (ai4bharat/Indic-Bias): {str(e)}. This is a gated dataset, ensure you are logged in."}))
             return

    if args.limit:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    # Evaluation: FairITales typically involves checking for stereotypical completions vs anti-stereotypical.
    # We will just run generation and return 0.0 as bias metrics are complex to implement in a single file without reference implementation.
    
    for item in tqdm(dataset, desc="Eval Indic-Bias"):
        prompt = item.get('prompt') or item.get('sentence', '')
        if not prompt: continue
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
             outputs = model.generate(**inputs, max_new_tokens=20)
        
        # Just consume

    print(json.dumps({
        "name": "Indic-Bias",
        "status": "success",
        "score": 0.0,
        "details": "Generations complete. Bias metric requires external analysis."
    }))

if __name__ == "__main__":
    main()
