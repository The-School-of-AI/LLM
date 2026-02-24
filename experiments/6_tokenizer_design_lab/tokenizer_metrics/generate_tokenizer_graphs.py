import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_graphs(input_csv="tokenizer_metrics.csv", output_dir="graphs"):
    # 1. Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found.")
        return

    # Set theme
    sns.set_theme(style="whitegrid")
    
    # helper to clean category names for plotting explanation
    def get_broad_category(cat):
        if "Code" in cat: return "Code"
        if "Indic" in cat: return "Indic"
        if "NCERT" in cat: return "NCERT"
        return "Other"

    df['BroadCategory'] = df['Category'].apply(get_broad_category)
    
    # Metrics to plot
    metrics = [
        ("Bytes/Token", "Bytes per Token (Lower is Better)"),
        ("Fertility", "Fertility (Tokens per Word)"),
        ("Speed (Tokens/sec)", "Speed (Tokens/sec) (Higher is Better)"),
        ("Fallback (%)", "Byte Fallback Rate (Lower is Better)"),
        ("Vocab Gini", "Vocabulary Inequality (Higher = Less Balanced)")
    ]
    
    # 2. Generate Plots per Broad Category
    broad_cats = ["Code", "Indic", "NCERT"]
    
    for broad_cat in broad_cats:
        subset = df[df['BroadCategory'] == broad_cat].copy()
        
        if subset.empty:
            continue
            
        # Clean up category names for x-axis (remove "Code - ", "Indic - ", "NCERT - ")
        subset['SubCategory'] = subset['Category'].apply(lambda x: x.split(' - ')[-1] if ' - ' in x else x)
        
        for metric, title in metrics:
            plt.figure(figsize=(14, 8))
            
            # Create Bar Plot
            sns.barplot(
                data=subset,
                x="SubCategory",
                y=metric,
                hue="Tokenizer",
                palette="viridis"
            )
            
            plt.title(f"{title} - {broad_cat} Datasets", fontsize=16)
            plt.xlabel("Language / Subject", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save
            filename = f"{broad_cat}_{metric.split(' ')[0].replace('/', '_')}.png"
            path = os.path.join(output_dir, filename)
            plt.savefig(path, dpi=300)
            plt.close()
            print(f"Saved {path}")

    # 3. Overall Summary (Average across all categories? Or just simple aggregated bar chart)
    # Let's do an aggregated average for high-level summary
    print("Generating Summary Plots...")
    summary_df = df.groupby(['Tokenizer', 'BroadCategory'])[
        ['Bytes/Token', 'Fertility', 'Speed (Tokens/sec)', 'Fallback (%)', 'Vocab Gini']
    ].mean().reset_index()
    
    for metric, title in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=summary_df,
            x="BroadCategory",
            y=metric,
            hue="Tokenizer",
            palette="viridis"
        )
        plt.title(f"Average {title} by Domain", fontsize=16)
        plt.xlabel("Domain", fontsize=12)
        plt.ylabel(f"Avg {metric}", fontsize=12)
        plt.legend(title="Tokenizer", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"Summary_{metric.split(' ')[0].replace('/', '_')}.png"
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved {path}")

if __name__ == "__main__":
    generate_graphs()
