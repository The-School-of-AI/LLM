import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_manifest(manifest_path):
    """Loads a JSONL manifest into a DataFrame."""
    data = []
    with open(manifest_path, "r") as f:
        for line in f:
            try:
                # Last line might be summary, skip if not dict or has 'distribution' key
                item = json.loads(line)
                if "distribution" in item or "total_tokens" in item:
                    continue  # Skip summary footer
                data.append(item)
            except (KeyError, ValueError, TypeError):
                continue
    return pd.DataFrame(data)


def plot_distributions(df, output_dir, stage_name):
    """Plots Band and Modality distributions."""
    if df.empty:
        print(f"Warning: Stage {stage_name} is empty. Skipping plots.")
        return

    # 1. Band Distribution
    plt.figure(figsize=(10, 6))
    if "assigned_band" in df.columns:
        sns.countplot(
            x="assigned_band", data=df, order=["B0", "B1", "B2", "B3", "B4", "B5"]
        )
        plt.title(f"Difficulty Band Distribution - {stage_name}")
        plt.savefig(os.path.join(output_dir, f"{stage_name}_band_dist.png"))
        plt.close()

    # 2. Modality Distribution
    plt.figure(figsize=(10, 6))
    if "assigned_modality" in df.columns:
        sns.countplot(y="assigned_modality", data=df)
        plt.title(f"Modality Distribution - {stage_name}")
        plt.savefig(os.path.join(output_dir, f"{stage_name}_modality_dist.png"))
        plt.close()


def plot_quality_box(df, output_dir, stage_name):
    """Plots Quality Score Box Plot per Band."""
    if (
        df.empty
        or "difficulty_score" not in df.columns
        or "assigned_band" not in df.columns
    ):
        return

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        x="assigned_band",
        y="difficulty_score",
        data=df,
        order=["B0", "B1", "B2", "B3", "B4", "B5"],
    )
    plt.title(f"Difficulty Score Metrics by Band - {stage_name}")
    plt.savefig(os.path.join(output_dir, f"{stage_name}_score_box.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Visualization Audits for Coresets"
    )
    parser.add_argument(
        "--manifest_dir",
        required=True,
        help="Directory containing manifest JSONL files",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_files = glob.glob(os.path.join(args.manifest_dir, "*_index.jsonl"))

    for mf in manifest_files:
        stage_name = os.path.basename(mf).replace("_index.jsonl", "")
        print(f"Processing {stage_name}...")

        df = load_manifest(mf)

        plot_distributions(df, args.output_dir, stage_name)
        plot_quality_box(df, args.output_dir, stage_name)

    print(f"Audits generated in {args.output_dir}")


if __name__ == "__main__":
    main()
