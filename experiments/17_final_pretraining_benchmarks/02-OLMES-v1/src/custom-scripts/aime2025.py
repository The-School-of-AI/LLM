import argparse
import json
import os
import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def parse_model_args(args_str):
    args = {}
    if not args_str:
        return args
    for part in args_str.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            args[k.strip()] = v.strip()
    return args

def extract_answer(text):
    # AIME answers are integers 0-999.
    # We look for \boxed{...} first
    boxed_match = re.search(r'\\boxed\{([^{}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # If no boxed, look for the last number
    # This is a naive fallback
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return numbers[-1]
    return None

def normalize_answer(ans):
    try:
        return str(int(ans))
    except:
        return str(ans).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_args", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args, unknown = parser.parse_known_args()

    model_args_dict = parse_model_args(args.model_args)
    model_name = model_args_dict.get("pretrained")
    
    if not model_name:
        print(json.dumps({"name": "AIME 2025", "status": "failed", "error": "No pretrained model specified"}))
        return

    # Device selection
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
             device = "mps"
        else:
            device = "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            device_map=device
        )
        model.eval()
    except Exception as e:
        print(json.dumps({"name": "AIME 2025", "status": "failed", "error": f"Model load failed: {str(e)}"}))
        return

    # Load Dataset
    try:
        # Using opencompass/AIME2025 or compatible
        dataset = load_dataset("MathArena/aime_2025", split="test") 
        # Note: Split name might vary, usually 'train' or 'test'. 
        # AIME 2025 is small (~30 questions), so usually in one split.
        # Fallback to 'train' if 'test' fails or check structure.
    except Exception:
        try:
             dataset = load_dataset("MathArena/aime_2025", split="train") 
        except Exception as e:
            print(json.dumps({"name": "AIME 2025", "status": "failed", "error": f"Dataset load failed: {str(e)}"}))
            return

    if args.limit:
        dataset = dataset.select(range(min(len(dataset), args.limit)))

    correct = 0
    total = 0
    
    details = []

    for item in tqdm(dataset, desc="Evaluating AIME 2025"):
        problem = item.get('problem') or item.get('question')
        gold_answer = item.get('answer')
        
        # Simple CoT Prompt
        prompt = f"Problem: {problem}\n\nPlease generate a step-by-step solution and put the final integer answer in \\boxed{{}}.\nSolution:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the new part
        response = generated_text[len(prompt):]
        
        pred_raw = extract_answer(response)
        
        is_correct = False
        if pred_raw:
            norm_pred = normalize_answer(pred_raw)
            norm_gold = normalize_answer(gold_answer)
            is_correct = (norm_pred == norm_gold)
        
        if is_correct:
            correct += 1
        total += 1
        
        details.append({
            "problem": problem[:50] + "...",
            "gold": gold_answer,
            "pred": pred_raw,
            "correct": is_correct
        })

    score = correct / total if total > 0 else 0.0
    
    output = {
        "name": "AIME 2025",
        "status": "success",
        "score": score,
        "details": details
    }
    
    print(json.dumps(output))

if __name__ == "__main__":
    main()
