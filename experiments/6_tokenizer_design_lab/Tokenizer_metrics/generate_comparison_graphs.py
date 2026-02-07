import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Configuration
INPUT_FILES = {
    "Custom (gptoss_128k_reindexed)": "gptoss_128k_reindexed.csv",
    "Custom (Mashed 128k)": "gptoss_mashed_128k_reindexed.csv",
    "Custom (Selected)": "tokenizer_metrics.csv"
}
OUTPUT_DIR = "graphs_comparison"
CUSTOM_TOKENIZER_NAME_IN_CSV = "Custom (GPTOSS Reindexed)" # The name used in the CSV rows

def generate_comparison_graphs():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    sns.set_theme(style="whitegrid")
    
    combined_df = pd.DataFrame()
    baselines_added = False
    
    # helper
    def get_broad_category(cat):
        if "Code" in cat: return "Code"
        if "Indic" in cat: return "Indic"
        if "NCERT" in cat: return "NCERT"
        return "Other"

    for label, filepath in INPUT_FILES.items():
        try:
            df = pd.read_csv(filepath)
            df['BroadCategory'] = df['Category'].apply(get_broad_category)
            
            # Extract Custom Tokenizer rows
            # The tokenizer name in the CSV is likely the filename without extension
            expected_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Fallback: if expected_name isn't there, try to find a "Custom" one or just take the one that isn't a baseline
            custom_rows = df[df['Tokenizer'] == expected_name].copy()
            
            if custom_rows.empty:
                # Try finding any row that looks like it might be the main one (heuristic)
                # Or just print available names to help debug
                print(f"Warning: Could not find tokenizer '{expected_name}' in {filepath}. Available: {df['Tokenizer'].unique()}")
                continue
                
            custom_rows['Tokenizer'] = label # Rename to the distinguishing label
            
            combined_df = pd.concat([combined_df, custom_rows], ignore_index=True)
            
            # Optionally add baselines (only once)
            if not baselines_added:
                # Add Baselines as reference
                baseline_names = ["gemma_tokenizer", "gptoss_tokenizer"] 
                baselines = df[df['Tokenizer'].isin(baseline_names)].copy()
                combined_df = pd.concat([combined_df, baselines], ignore_index=True)
                baselines_added = True
                
        except FileNotFoundError:
            print(f"Warning: {filepath} not found. Skipping.")

    if combined_df.empty:
        print("No data loaded. Exiting.")
        return

    # Generate Plots
    metrics = [
        ("Bytes/Token", "Bytes per Token (Lower is Better)"),
        ("Fertility", "Fertility (Tokens per Word)"),
        ("Speed (Tokens/sec)", "Speed (Tokens/sec) (Higher is Better)"),
        ("Fallback (%)", "Byte Fallback Rate (Lower is Better)"),
        ("Vocab Gini", "Vocabulary Inequality (Higher = Less Balanced)")
    ]
    
    broad_cats = ["Code", "Indic", "NCERT"]
    
    for broad_cat in broad_cats:
        subset = combined_df[combined_df['BroadCategory'] == broad_cat].copy()
        
        if subset.empty:
            continue
            
        # Clean up category names
        subset['SubCategory'] = subset['Category'].apply(lambda x: x.split(' - ')[-1] if ' - ' in x else x)
        
        for metric, title in metrics:
            plt.figure(figsize=(14, 8))
            
            # Define order: Baselines first, then Customs? Or just sort by performance?
            # Let's let seaborn handle it, but hue works well.
            
            chart = sns.barplot(
                data=subset,
                x="SubCategory",
                y=metric,
                hue="Tokenizer",
                palette="tab10" # Distinct colors
            )
            
            plt.title(f"{title} - {broad_cat} (Comparison)", fontsize=16)
            plt.xlabel("Language / Subject", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            filename = f"Comparison_{broad_cat}_{metric.split(' ')[0].replace('/', '_')}.png"
            path = os.path.join(OUTPUT_DIR, filename)
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"Saved {path}")
            
    # Summary Plot
    print("Generating Summary Comparison...")
    summary_df = combined_df.groupby(['Tokenizer', 'BroadCategory'])[
        ['Bytes/Token', 'Fertility', 'Speed (Tokens/sec)', 'Fallback (%)', 'Vocab Gini']
    ].mean().reset_index()
    
    for metric, title in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=summary_df,
            x="BroadCategory",
            y=metric,
            hue="Tokenizer",
            palette="tab10"
        )
        plt.title(f"Average {title} (Comparison)", fontsize=16)
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel(f"Avg {metric}", fontsize=12)
        plt.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"Summary_Comparison_{metric.split(' ')[0].replace('/', '_')}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved {path}")

if __name__ == "__main__":
    generate_comparison_graphs()
