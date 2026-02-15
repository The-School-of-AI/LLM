import glob
import os
import yaml

def load_config_with_inheritance(config_path):
    """Mirroring the merge logic from eval_runner.py"""
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error parsing {config_path}: {e}")
            return None
    
    if not isinstance(config, dict):
        return None

    base_file = config.get("base_config")
    if base_file:
        config_dir = os.path.dirname(config_path)
        base_path = os.path.join(config_dir, base_file)
        
        if os.path.exists(base_path):
            with open(base_path, "r") as f:
                base_config = yaml.safe_load(f)
            
            merged_benchmarks_dict = {b["name"]: b for b in base_config.get("benchmarks", [])}
            stage_benchmarks = config.get("benchmarks", [])
            for b in stage_benchmarks:
                merged_benchmarks_dict[b["name"]] = b
            config["benchmarks"] = list(merged_benchmarks_dict.values())
            
    return config

def generate_table():
    base_dir = os.getcwd()
    if base_dir.endswith("src"):
        base_dir = os.path.dirname(os.path.dirname(base_dir))

    config_dir = os.path.join(base_dir, "01-EleutherAI-v1", "configs")
    if not os.path.exists(config_dir):
        print(f"Config directory not found: {config_dir}")
        return

    files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    # Filter out base_benchmarks.yaml from being processed as a stage
    files = [f for f in files if "base_benchmarks.yaml" not in f]

    priority = {
        "stage_1b.yaml": 1,
        "stage_3b.yaml": 2,
        "stage_8b.yaml": 3,
        "stage_70b.yaml": 4,
        "sft_stage.yaml": 5,
    }
    files.sort(key=lambda x: priority.get(os.path.basename(x), 99))

    headers = ["Stage", "Phase", "Benchmark", "Type", "Shots", "Limit", "CoT", "Paradigm/Mode", "Tasks"]
    widths = [10, 12, 30, 10, 6, 6, 5, 20, 50]

    def format_row(cols):
        return "| " + " | ".join(f"{str(c):<{widths[i]}}" for i, c in enumerate(cols)) + " |"

    output_file = os.path.join(base_dir, "01-EleutherAI-v1", "benchmarks_config_table.md")

    with open(output_file, "w") as f:
        f.write("# Benchmark Configuration Overview\n\n")
        f.write(format_row(headers) + "\n")
        f.write("| " + " | ".join("-" * w for w in widths) + " |\n")

        for file_path in files:
            config = load_config_with_inheritance(file_path)
            if not config: continue
            
            stage = config.get("stage", "Unknown")
            for bench in config.get("benchmarks", []):
                if not bench.get("enabled", True): continue

                name = bench.get("name", "Unknown")
                phases = bench.get("phases", [])
                b_type = "Custom" if "custom_script" in bench else "Harness"
                shots = bench.get("shots", 0)
                limit = bench.get("limit", "All")
                cot = "Yes" if bench.get("cot", False) else "No"
                pm_val = bench.get("paradigm", bench.get("mode", "-"))

                task_list = bench.get("tasks", bench.get("subjects", bench.get("subset")))
                if not task_list:
                    tasks_str = "(Default/All)"
                elif isinstance(task_list, list):
                    tasks_str = ", ".join(task_list)
                else:
                    tasks_str = str(task_list)

                for phase in phases:
                    row = [stage, phase, name, b_type, shots, limit, cot, pm_val, tasks_str]
                    f.write(format_row(row) + "\n")

    print(f"Successfully generated table at: {output_file}")

if __name__ == "__main__":
    generate_table()
