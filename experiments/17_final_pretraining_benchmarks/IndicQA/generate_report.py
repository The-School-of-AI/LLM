import argparse
import json
from datetime import datetime


def compute_macro_average(results, key):
    values = [results[lang][key] for lang in results if results[lang][key] is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def generate_markdown(results, model_name, checkpoint):

    today = datetime.today().strftime("%Y-%m-%d")

    languages = list(results.keys())

    macro_f1 = compute_macro_average(results, "F1")
    macro_em = compute_macro_average(results, "EM")
    macro_hall = compute_macro_average(results, "HallucinationRate")
    macro_copy = compute_macro_average(results, "CopyRatio")

    lines = []

    # Header
    lines.append(f"# IndicQA Evaluation Report")
    lines.append(f"Model: {model_name}")
    lines.append(f"Checkpoint: {checkpoint}")
    lines.append(f"Date: {today}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary Table
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| Language | EM | F1 | Hallucination | CopyRatio | FirstTokenAcc | AvgPredLen | Tokens/Char |")
    lines.append("|----------|----|----|---------------|-----------|---------------|------------|-------------|")

    for lang in languages:
        r = results[lang]
        lines.append(
            f"| {lang} | {r['EM']} | {r['F1']} | {r['HallucinationRate']} | "
            f"{r['CopyRatio']} | {r['FirstTokenAcc']} | "
            f"{r['AvgPredLen']} | {r['AvgTokensPerChar']} |"
        )

    lines.append("")
    lines.append("### Macro Averages")
    lines.append("")
    lines.append(f"- Macro EM: {macro_em}")
    lines.append(f"- Macro F1: {macro_f1}")
    lines.append(f"- Macro Hallucination: {macro_hall}")
    lines.append(f"- Macro CopyRatio: {macro_copy}")
    lines.append("")

    # Position Sensitivity
    lines.append("## Position Sensitivity (F1 %)")
    lines.append("")
    lines.append("| Language | Early | Middle | Late |")
    lines.append("|----------|--------|--------|------|")

    for lang in languages:
        pos = results[lang].get("PositionF1", {})
        early = pos.get("early")
        middle = pos.get("middle")
        late = pos.get("late")

        lines.append(f"| {lang} | {early} | {middle} | {late} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Behavioral Interpretation Section (auto hints)
    lines.append("## Automated Observations")
    lines.append("")

    for lang in languages:
        r = results[lang]
        lines.append(f"### {lang}")

        if r["CopyRatio"] > 80:
            lines.append("- Strong extractive copy behavior.")
        elif r["CopyRatio"] < 50:
            lines.append("- High paraphrastic or hallucination tendency.")

        if r["HallucinationRate"] > 40:
            lines.append("- Elevated hallucination rate.")

        if r["AvgPredLen"] > r["AvgGoldLen"] * 2:
            lines.append("- Over-generation relative to gold answers.")

        pos = r.get("PositionF1", {})
        if pos.get("late") is not None and pos.get("early") is not None:
            if pos["late"] < pos["early"]:
                lines.append("- Performance drops for late-context answers.")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    with open(args.json_path) as f:
        results = json.load(f)

    report_md = generate_markdown(results, args.model_name, args.checkpoint)

    output_file = args.output or f"indicqa_report_{args.model_name}.md"

    with open(output_file, "w") as f:
        f.write(report_md)

    print(f"Report written to {output_file}")


if __name__ == "__main__":
    main()