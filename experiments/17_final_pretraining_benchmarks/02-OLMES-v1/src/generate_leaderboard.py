import json
import os
import argparse
import glob
from datetime import datetime

def extract_model_name(model_args):
    if "pretrained=" in model_args:
        try:
            return model_args.split("pretrained=")[1].split(",")[0].split("/")[-1]
        except Exception:
            return "unknown"
    return model_args

def generate_leaderboard(results_dirs, output_file):
    # Mapping for publishable pillars
    pillars = {
        "English": ["MMLU", "ARC", "HellaSwag", "Winogrande", "PIQA", "TriviaQA", "qa_rc", "qa_bpb"],
        "Reasoning": ["GSM8K", "BBH", "MATH", "ZebraLogic"],
        "Coding": ["HumanEval", "MBPP", "code_bpb", "DS1000", "BigCodeBench"],
        "Multilingual": ["IndicGLUE", "IndicQA", "Indic-Bias"]
    }

    # 1. Collect and filter for the latest run per (Model, Stage)
    json_files = []
    for d in results_dirs:
        json_files.extend(glob.glob(os.path.join(d, "**", "final_results.json"), recursive=True))
    
    latest_runs = {} # Key: (Model, Stage), Value: (Timestamp, RowData)

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
            
            meta = data["metadata"]
            agg = data.get("aggregates", {})
            full = data.get("granular", {})
            
            model_name = extract_model_name(meta["model_args"])
            stage = meta["stage"]
            timestamp = meta["timestamp"]
            
            # Combine all scores for lookup
            all_scores = {**agg, **full}
            
            row = {
                "Model": model_name,
                "Stage": stage,
                "Timestamp": timestamp.split("T")[0],
                "Scores": {}
            }
            
            # Calculate Pillar Averages
            for pillar, tasks in pillars.items():
                pillar_scores = []
                for t_match in tasks:
                    for k, v in all_scores.items():
                        if t_match.lower() in k.lower() and v is not None:
                            pillar_scores.append(v)
                
                row["Scores"][pillar] = sum(pillar_scores) / len(pillar_scores) if pillar_scores else 0.0
            
            # Keep only the latest
            key = (model_name, stage)
            if key not in latest_runs or timestamp > latest_runs[key]["full_ts"]:
                row["full_ts"] = timestamp
                latest_runs[key] = row
                
        except Exception as e:
            print(f"Error parsing {jf}: {e}")

    rows = list(latest_runs.values())

    # Sort by Stage (order of training) and then Model
    stage_order = {"ci_breadth": 0, "pretrain_1b": 1, "pretrain_3b": 2, "pretrain_8b": 3, "pretrain_70b": 4, "sft": 5}
    rows.sort(key=lambda x: (stage_order.get(x["Stage"], 99), x["Model"]))

    # 2. Generate Markdown Table
    with open(output_file, "w") as f:
        f.write("# OLMES Milestone Leaderboard\n\n")
        f.write("Comparative analysis across training stages and capability pillars.\n\n")
        
        headers = ["Model", "Stage", "English", "Reasoning", "Coding", "Multilingual", "Date"]
        header_row = "| " + " | ".join(headers) + " |"
        sep_row = "| " + " | ".join([":---"] * len(headers)) + " |"
        
        f.write(header_row + "\n")
        f.write(sep_row + "\n")
        
        for r in rows:
            cols = [
                f"**{r['Model']}**",
                f"`{r['Stage']}`",
                f"{r['Scores'].get('English', 0.0):.2%}",
                f"{r['Scores'].get('Reasoning', 0.0):.2%}",
                f"{r['Scores'].get('Coding', 0.0):.2%}",
                f"{r['Scores'].get('Multilingual', 0.0):.2%}",
                r["Timestamp"]
            ]
            f.write("| " + " | ".join(cols) + " |\n")
        
        f.write("\n\n*Scores are normalized averages across representative pillar tasks.*")

    print(f"Leaderboard generated at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="benchmark-results", help="Root directory for results")
    parser.add_argument("--out", type=str, default="LEADERBOARD.md", help="Output Markdown file")
    args = parser.parse_args()
    
    generate_leaderboard([args.dir], args.out)
