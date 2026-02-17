
import argparse
import json
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_haystack(tokenizer, context_length, needle, depth_percent):
    """
    Generates a haystack of `context_length` tokens.
    """
    
    # "Paul Graham essays" style filler text (simulated repeated text for efficiency)
    filler_text = "The quick brown fox jumps over the lazy dog. " * 50 
    filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
    
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    
    # Calculate how many filler tokens we need
    num_needle_tokens = len(needle_tokens)
    num_filler_needed = context_length - num_needle_tokens
    
    # Repeat filler to fill context
    haystack_tokens = []
    while len(haystack_tokens) < num_filler_needed:
        haystack_tokens.extend(filler_tokens)
    
    # Trim to exact size
    haystack_tokens = haystack_tokens[:num_filler_needed]
    
    # Insert needle
    insertion_point = int(len(haystack_tokens) * (depth_percent / 100))
    haystack_tokens = haystack_tokens[:insertion_point] + needle_tokens + haystack_tokens[insertion_point:]
    
    return tokenizer.decode(haystack_tokens)

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
    parser.add_argument("--context_length", type=int, default=4096)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None) # Ignored, but kept for compatibility
    parser.add_argument("--batch_size", type=int, default=1) # Ignored, but kept for compatibility
    args = parser.parse_args()

    # Configuration
    # We will test a single specific needle/question pair for simplicity in this MVP.
    # ideally this would iterate over many.
    needle = " The special pass key is '74921'. Remember it. "
    question = "What is the special pass key?"
    answer = "74921"
    
    # Depths to test
    depths = [0, 25, 50, 75, 100]
    scores = []
    
    try:
        model_args_dict = parse_model_args(args.model_args)
        model_name = model_args_dict.get("pretrained")

        if not model_name:
             raise ValueError("No pretrained model specified")

        device = args.device
        if device is None:
             if torch.cuda.is_available(): device = "cuda"
             elif torch.backends.mps.is_available(): device = "mps"
             else: device = "cpu"

        # Correct MPS device check
        if device == "mps" and not torch.backends.mps.is_available(): device = "cpu"


        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
        model.eval()

        for depth in depths:
            prompt_text = generate_haystack(tokenizer, args.context_length, needle, depth)
            
            # Add question at the end
            full_prompt = f"{prompt_text}\n\nQ: {question}\nA:"
            
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, temperature=0.0)
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Score
            score = 1.0 if answer in generated_text else 0.0
            scores.append(score)
        
        avg_score = sum(scores) / len(scores)
        
        print(json.dumps({
            "name": f"niah_{args.context_length}",
            "status": "success",
            "score": avg_score,
            "details": {
                "depths": depths,
                "scores": scores,
                "generated": generated_text # Just show the last one for debugging
            }
        }))

    except Exception as e:
        # Use full length if defined, else fallback
        name_reported = f"niah_{args.context_length}" if args.context_length > 1000 else f"niah_{args.context_length}k"
        print(json.dumps({
            "name": name_reported,
            "status": "failed",
            "error": str(e)
        }))
        return

if __name__ == "__main__":
    set_seed(42)
    main()
