import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import sys
import os

# Configuration
OUTPUT_DIR = Path('tokenizer_less_than_32')
BASE_DIR = Path(r'I:\school_of_ai\LLM\experiments\6_tokenizer_design_lab\LLM-vinodjtokenizer_selection\experiments\tokenizer\selection')

# Tokenizer files map
TOKENIZER_FILES = {
    'ByteDance': BASE_DIR / 'byted_tokenizer.json',
    'DeepSeek': BASE_DIR / 'ds_tokenizer.json',
    'GPT-OSS': BASE_DIR / 'gptoss_tokenizer.json',
    'Mistral': BASE_DIR / 'mistral_tokenizer.json',
    'OLMo': BASE_DIR / 'olmo_tokenizer.json',
    'Qwen': BASE_DIR / 'qwen_tokenizer.json'
}

# --- Helper Functions (Reused) ---

def create_byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

byte_decoder = create_byte_decoder()

def decode_bpe_token(token):
    try:
        bytes_list = [byte_decoder.get(c, ord(c)) for c in token]
        return bytes(bytes_list).decode('utf-8', errors='replace')
    except:
        return token

def classify_decoded_token(decoded):
    scripts = Counter()
    for char in decoded:
        if char == '': continue
        code = ord(char)
        # Detailed Classification Logic...
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF: scripts["Chinese"] += 1
        elif 0x3040 <= code <= 0x309F: scripts["Japanese (Hiragana)"] += 1
        elif 0x30A0 <= code <= 0x30FF: scripts["Japanese (Katakana)"] += 1
        elif 0xAC00 <= code <= 0xD7AF: scripts["Korean"] += 1
        elif 0x0400 <= code <= 0x04FF: scripts["Cyrillic"] += 1
        elif 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F: scripts["Arabic"] += 1
        elif 0x0590 <= code <= 0x05FF: scripts["Hebrew"] += 1
        elif 0x0370 <= code <= 0x03FF: scripts["Greek"] += 1
        elif 0x0E00 <= code <= 0x0E7F: scripts["Thai"] += 1
        elif 0x1E00 <= code <= 0x1EFF: scripts["Vietnamese"] += 1
        elif 0x0900 <= code <= 0x097F: scripts["Devanagari"] += 1
        elif 0x0980 <= code <= 0x09FF: scripts["Bengali"] += 1
        elif 0x0A00 <= code <= 0x0A7F: scripts["Gurmukhi"] += 1
        elif 0x0A80 <= code <= 0x0AFF: scripts["Gujarati"] += 1
        elif 0x0B00 <= code <= 0x0B7F: scripts["Odia"] += 1
        elif 0x0B80 <= code <= 0x0BFF: scripts["Tamil"] += 1
        elif 0x0C00 <= code <= 0x0C7F: scripts["Telugu"] += 1
        elif 0x0C80 <= code <= 0x0CFF: scripts["Kannada"] += 1
        elif 0x0D00 <= code <= 0x0D7F: scripts["Malayalam"] += 1
        elif 0x0D80 <= code <= 0x0DFF: scripts["Sinhala"] += 1
        elif code <= 0x007F:
            if char.isalpha(): scripts["Latin (ASCII)"] += 1
            elif char.isdigit(): scripts["Digits"] += 1
            elif char.isspace(): scripts["Whitespace"] += 1
            else: scripts["Punctuation/Symbols"] += 1
        elif 0x0080 <= code <= 0x024F: scripts["Latin Extended"] += 1
        else: scripts["Other"] += 1
    
    if scripts:
        return max(scripts.items(), key=lambda x: x[1])[0]
    return "Unknown"

# --- Main Analysis Script ---

def main():
    print("Starting Token Analysis (< 32 chars)...")
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = OUTPUT_DIR / 'tokens_less_than_32.csv'
    all_data = []

    # 1. Gather Data
    for model_name, path in TOKENIZER_FILES.items():
        print(f"Processing {model_name}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                vocab = data['model']['vocab']  # format: {"token": id, ...}
                
                for token_str, token_id in enumerate(vocab.items()):
                    # vocab.items() returns (key, value), enumeration adds index. 
                    # Wait, enumerate(vocab.items()) yields (index, (key, value))
                    # Previous code was `for i, (token_str, token_id) in enumerate(vocab.items()):`
                    # I want to revert to `for token_str, token_id in vocab.items():` but I need to be careful with indentation.
                    pass 
                
                # Let's rewrite the loop cleanly
                for token_str, token_id in vocab.items():
                    decoded = decode_bpe_token(token_str)
                    
                    # Filter: Less than 32 chars, valid, and no replacement chars (\ufffd)
                    if len(decoded) < 32 and '\ufffd' not in decoded:
                        category = classify_decoded_token(decoded)
                        
                        # Add to list
                        all_data.append({
                            'Model': model_name,
                            'TokenID': token_id, 
                            'Token': decoded,
                            'Length': len(decoded),
                            'Category': category,
                            'HexRep': token_str if token_str != decoded else '' 
                        })
                        
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    # 2. Create CSV
    print(f"Saving data to {csv_file_path}...")
    df = pd.DataFrame(all_data)
    df.to_csv(csv_file_path, index=False, encoding='utf-8')
    print(f"Saved {len(df)} rows.")

    # 3. generate Visualization (Top 10 Categories per Model)
    print("Generating visualizations...")
    
    # Aggregated counts: Model vs Category
    category_counts = df.groupby(['Model', 'Category']).size().reset_index(name='Count')
    
    # Get Top 10 Categories overall to normalize the chart or do per-model subplots
    # Strategy: FacetGrid barplots
    
    sns.set_theme(style="whitegrid")
    
    # We want top 10 categories for EACH model. 
    # It's better to visualize the distribution of just the top global categories or per model.
    # Let's do a large plot for Top 10 Categories by Count for each Model
    
    models = df['Model'].unique()
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(3, 2, figsize=(20, 18), constrained_layout=True)
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = category_counts[category_counts['Model'] == model].sort_values('Count', ascending=False).head(10)
        
        sns.barplot(x='Count', y='Category', data=model_data, ax=ax, palette='viridis')
        ax.set_title(f'Top 10 Token Categories (< 32 chars) - {model}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Token Count')
        ax.set_ylabel('')
        
        # Add values on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3)

    plt.suptitle('Token Category Distribution (Top 10) for Tokens < 32 Characters', fontsize=20, y=1.02)
    
    viz_path = OUTPUT_DIR / 'top_10_categories_visualization.png'
    plt.savefig(viz_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {viz_path}")
    
    # 4. Also generate a simple text summary of Top 10 list as JSON/Text for quick viewing
    summary_path = OUTPUT_DIR / 'top_10_categories_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("TOP 10 CATEGORIES PER MODEL (< 32 chars)\n")
        f.write("="*50 + "\n\n")
        for model in models:
            f.write(f"MODEL: {model}\n")
            f.write("-" * 20 + "\n")
            model_df = df[df['Model'] == model]
            top_cats = model_df['Category'].value_counts().head(10)
            for cat, count in top_cats.items():
                pct = (count / len(model_df)) * 100
                f.write(f"{cat}: {count:,} ({pct:.2f}%)\n")
            f.write("\n")
            
    print("Analysis Complete.")

if __name__ == "__main__":
    main()
