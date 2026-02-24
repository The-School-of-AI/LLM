import os
import glob
import json
import time
from collections import Counter
import numpy as np
import pandas as pd
import tiktoken
from tokenizers import Tokenizer
from transformers import AutoTokenizer

# Configuration
LIMIT_LINES = None # Set to an integer (e.g., 1000) for testing, or None for full run
# Use a smaller sample for speed if needed, e.g., 5MB per file
MAX_BYTES_PER_FILE = 5 * 1024 * 1024 

TOKENIZERS_CONFIG = {}
tokenizer_files = glob.glob("tokenizers/*.json")
for path in tokenizer_files:
    # Use filename as label, e.g. "gemma_tokenizer"
    name = os.path.splitext(os.path.basename(path))[0]
    TOKENIZERS_CONFIG[name] = {
        "type": "file",
        "path": path
    }

# Explicitly add our main tokenizer from gptoss_pruning if it exists
gptoss_path = os.path.join("..", "gptoss_pruning", "tokenizer.json")
if os.path.exists(gptoss_path):
    print(f"Found main tokenizer at {gptoss_path}")
    TOKENIZERS_CONFIG["our_tokenizer"] = {
        "type": "file",
        "path": gptoss_path
    }

DATA_DIRS = {
    "Code": "datasets/code",
    "Indic": "datasets/indic",
    "NCERT": "datasets/ncert"
}

def load_tokenizer(config):
    try:
        if config["type"] == "file":
            return Tokenizer.from_file(config["path"])
        elif config["type"] == "tiktoken":
            return tiktoken.get_encoding(config["name"])
        elif config["type"] == "transformers":
            return AutoTokenizer.from_pretrained(config["name"], trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer {config}: {e}")
        return None

def get_vocab_size(name, tokenizer):
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    elif hasattr(tokenizer, "n_vocab"): # tiktoken
        return tokenizer.n_vocab
    elif hasattr(tokenizer, "get_vocab"):
        return len(tokenizer.get_vocab())
    # Fallback for Tokenizers library object
    if hasattr(tokenizer, "get_vocab_size"):
        return tokenizer.get_vocab_size()
    return "N/A"

def count_tokens(tokenizer, text, tokenizer_type):
    if tokenizer_type == "tiktoken":
        tokens = tokenizer.encode(text)
        return len(tokens), tokens
    elif tokenizer_type == "transformers":
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens), tokens
    elif tokenizer_type == "file":
        # tokenizers library
        encoding = tokenizer.encode(text)
        return len(encoding.ids), encoding.ids
    return 0, []

def is_byte_token(token_id, tokenizer, tokenizer_type):
    # Heuristic for byte fallback: check if token represents a single byte
    # This is complex across different tokenizers.
    # For now, we might skip this unless we have a reliable way for each.
    # Placeholder: return False
    return False

def get_dataset_files():
    dataset_map = {} # Category -> [files]
    
    # Code
    if os.path.exists(DATA_DIRS["Code"]):
        code_langs = glob.glob(os.path.join(DATA_DIRS["Code"], "*"))
        found_any_code = False
        for lang_dir in code_langs:
            if not os.path.isdir(lang_dir):
                continue
            lang = os.path.basename(lang_dir)
            files = glob.glob(os.path.join(lang_dir, "*.jsonl"))
            if files:
                dataset_map[f"Code - {lang}"] = files
                found_any_code = True
            else:
                print(f"Warning: No .jsonl files found in '{lang_dir}'. Please ensure dataset files are in .jsonl format.")
        
        if not found_any_code:
            print(f"Warning: No valid language subdirectories with .jsonl files found in '{DATA_DIRS['Code']}'.")
    else:
        print(f"Warning: Dataset directory '{DATA_DIRS['Code']}' not found.")

    # Indic
    if os.path.exists(DATA_DIRS["Indic"]):
        indic_langs = glob.glob(os.path.join(DATA_DIRS["Indic"], "*"))
        found_any_indic = False
        for lang_dir in indic_langs:
            if not os.path.isdir(lang_dir):
                continue
            lang = os.path.basename(lang_dir)
            files = glob.glob(os.path.join(lang_dir, "*.jsonl"))
            if files:
                dataset_map[f"Indic - {lang}"] = files
                found_any_indic = True
            else:
                print(f"Warning: No .jsonl files found in '{lang_dir}'. Please ensure dataset files are in .jsonl format.")
        
        if not found_any_indic:
            print(f"Warning: No valid language subdirectories with .jsonl files found in '{DATA_DIRS['Indic']}'.")
    else:
        print(f"Warning: Dataset directory '{DATA_DIRS['Indic']}' not found.")

    # NCERT (Group by Subject)
    # We need to read the files to know the subjects, or just process them all and group dynamically
    if os.path.exists(DATA_DIRS["NCERT"]):
        ncert_files = glob.glob(os.path.join(DATA_DIRS["NCERT"], "*.jsonl"))
        if ncert_files:
            dataset_map["NCERT"] = ncert_files # Special handling for subject grouping later
        else:
             print(f"Warning: No .jsonl files found in '{DATA_DIRS['NCERT']}'. Please ensure dataset files are in .jsonl format.")
    else:
        print(f"Warning: Dataset directory '{DATA_DIRS['NCERT']}' not found.")
        
    return dataset_map

def sample_data(files, category_name):
    # Generator that yields (Category, Text)
    # For NCERT, category will be "NCERT - Subject"
    
    for fpath in files:
        file_bytes_read = 0
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    if LIMIT_LINES and file_bytes_read > MAX_BYTES_PER_FILE: # Primitive limiter
                        break
                        
                    try:
                        data = json.loads(line)
                        text = ""
                        sub_category = category_name
                        
                        if "NCERT" in category_name:
                            subject = data.get("subject", "Unknown")
                            sub_category = f"NCERT - {subject}"
                            # NCERT structure: often "Question", "Answer", "Explanation" or just keys.
                            # We'll join all string values or look for specific fields.
                            # Based on file view: "Explanation", "Question", "Answer"
                            parts = [data.get("Explanation", ""), data.get("Question", ""), data.get("Answer", "")]
                            text = "\n".join([p for p in parts if p])
                        else:
                            # Code/Indic usually have 'content' or 'text'
                            text = data.get("content", data.get("text", ""))
                        
                        if not text:
                            continue
                            
                        # Limit text chunk size if needed, but metrics are per token
                        encoded_text = text.encode("utf-8")
                        text_bytes = len(encoded_text)
                        
                        file_bytes_read += text_bytes
                        if file_bytes_read > MAX_BYTES_PER_FILE:
                             # Just yield this last one and break file loop
                            yield sub_category, text, text_bytes
                            break
                        
                        yield sub_category, text, text_bytes
                        
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

def calculate_gini(counts):
    """Calculate Gini coefficient of inequality for token usage."""
    if not counts: return 0.0
    array = np.array(list(counts.values()), dtype=np.float64)
    if np.amin(array) < 0:
        return 0.0 # Should not happen for counts
    array += 1e-9 # Avoid div by zero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((2 * index - n - 1) * array).sum() / (n * array.sum())

def count_byte_fallbacks(tokenizer, ids, tokenizer_type):
    """Count how many tokens are byte fallbacks (<0x..> pattern)."""
    count = 0
    
    if tokenizer_type == "transformers" or tokenizer_type == "file":
        try:
            # Different libraries have different ways to get token text
            if hasattr(tokenizer, "convert_ids_to_tokens"):
                tokens = tokenizer.convert_ids_to_tokens(ids)
                for t in tokens:
                    if isinstance(t, str) and t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                        count += 1
            # For 'tokenizers' library (fast)
            elif hasattr(tokenizer, "id_to_token"):
                for i in ids:
                    t = tokenizer.id_to_token(i)
                    if t and t.startswith("<0x") and t.endswith(">") and len(t) == 6:
                        count += 1
        except Exception:
            pass
            
    # Tiktoken handling is harder without explicit map, skipping for now or assumed 0 for base text
    return count

def evaluate():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit_lines", type=int, default=None)
    args = parser.parse_args()
    
    global LIMIT_LINES
    if args.limit_lines:
        LIMIT_LINES = args.limit_lines
        print(f"Limiting to first {LIMIT_LINES} lines/items per file (approx).")

    # 1. Load Tokenizers
    print("Loading tokenizers...")
    loaded_tokenizers = {}
    tokenizer_vocab_sizes = {}
    
    for name, config in TOKENIZERS_CONFIG.items():
        print(f"  Loading {name}...")
        tok = load_tokenizer(config)
        if tok:
            loaded_tokenizers[name] = (tok, config["type"])
            tokenizer_vocab_sizes[name] = get_vocab_size(name, tok)
        else:
            print(f"Failed to load {name}")

    if not loaded_tokenizers:
        print("No tokenizers loaded. Exiting.")
        return

    # 2. Prepare Data Sources
    dataset_map = get_dataset_files()
    
    # 3. specific storage for aggregated stats
    stats = {tok: {} for tok in loaded_tokenizers}
    
    print("Starting evaluation...")
    
    # Suppress transformers warnings
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Iterate datasets
    for base_category, files in dataset_map.items():
        print(f"Processing {base_category}...")
        
        data_gen = sample_data(files, base_category)
        
        count_files = 0
        for specific_category, text, num_bytes in data_gen:
            count_files += 1
            if count_files % 10 == 0:
                print(f"  Processed {count_files} chunks in {base_category}...")
                
            # Basic pre-calc
            words = len(text.split())
            
            for tok_name, (tok, tok_type) in loaded_tokenizers.items():
                if specific_category not in stats[tok_name]:
                    stats[tok_name][specific_category] = {
                        "bytes": 0, "tokens": 0, "words": 0, "time": 0,
                        "token_counts": Counter(), "byte_fallbacks": 0
                    }
                
                s = stats[tok_name][specific_category]
                s["bytes"] += num_bytes
                s["words"] += words
                
                # Tokenization
                try:
                    start_t = time.perf_counter()
                    count, tokens = count_tokens(tok, text, tok_type)
                    end_t = time.perf_counter()
                    
                    s["tokens"] += count
                    s["time"] += (end_t - start_t)
                    s["token_counts"].update(tokens)
                    s["byte_fallbacks"] += count_byte_fallbacks(tok, tokens, tok_type)
                    
                except Exception as e:
                    print(f"  Error tokenizing with {tok_name}: {e}")

    # 4. Compile Results
    results = []
    
    # Get all unique categories encountered
    all_categories = set()
    for tok_stats in stats.values():
        all_categories.update(tok_stats.keys())
    
    for category in sorted(list(all_categories)):
        for tok_name in loaded_tokenizers:
            s = stats[tok_name].get(category)
            if not s or s["tokens"] == 0:
                continue
                
            bytes_per_token = s["bytes"] / s["tokens"]
            fertility = s["tokens"] / s["words"] if s["words"] > 0 else 0
            speed = s["tokens"] / s["time"] if s["time"] > 0 else 0
            vocab_size = tokenizer_vocab_sizes[tok_name]
            fallback_rate = (s["byte_fallbacks"] / s["tokens"]) * 100 if s["tokens"] > 0 else 0
            gini = calculate_gini(s["token_counts"])
            
            results.append({
                "Tokenizer": tok_name,
                "Vocab Size": vocab_size,
                "Category": category,
                "Bytes/Token": round(bytes_per_token, 4),
                "Fertility": round(fertility, 4),
                "Speed (Tokens/sec)": round(speed, 2),
                "Fallback (%)": round(fallback_rate, 4),
                "Vocab Gini": round(gini, 4),
                "Total Tokens": s["tokens"]
            })
            
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results generated.")
        return

    # Sort for readability
    df = df.sort_values(["Category", "Bytes/Token"]) # Lower B/T is better usually
    
    df.to_csv("tokenizer_metrics.csv", index=False)
    print("\nSaved to tokenizer_metrics.csv")
    
    print("\nEvaluation Results:")
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df.to_string(index=False))
    except Exception as e:
        print(f"Could not print markdown table: {e}")
        print(df)
    
    # Metadata footer
    print("\nDatasets Used:")
    for cat, files in dataset_map.items():
        print(f"{cat}: {len(files)} files")

if __name__ == "__main__":
    evaluate()
