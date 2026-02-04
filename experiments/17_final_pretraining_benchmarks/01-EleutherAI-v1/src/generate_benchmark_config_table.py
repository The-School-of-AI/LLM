import yaml
import glob
import os
import sys

def generate_table():
    # Determine the directory containing configs
    # Assuming this script is run from project root or src, we locate 01-EleutherAI-v1/configs
    base_dir = os.getcwd()
    if base_dir.endswith("src"):
        base_dir = os.path.dirname(os.path.dirname(base_dir))
    
    # Try multiple paths to find the configs directory
    possible_paths = [
        os.path.join(base_dir, "01-EleutherAI-v1", "configs"),
        os.path.join(base_dir, "..", "configs"), # If run from src/
        "01-EleutherAI-v1/configs" # Fallback relative
    ]
    
    config_dir = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            config_dir = p
            break
            
    if not config_dir:
        print(f"Config directory not found. Searched: {possible_paths}")
        return

    files = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
    
    # Sort files logically
    priority = {
        'stage_1b.yaml': 1, 
        'stage_3b.yaml': 2, 
        'stage_8b.yaml': 3, 
        'stage_70b.yaml': 4, 
        'sft_stage.yaml': 5
    }
    files.sort(key=lambda x: priority.get(os.path.basename(x), 99))

    # Define Column Widths
    # Stage | Phase | Benchmark | Type | Shots | CoT | Paradigm/Mode | Tasks
    headers = ["Stage", "Phase", "Benchmark", "Type", "Shots", "CoT", "Paradigm/Mode", "Tasks"]
    widths = [10, 12, 30, 10, 6, 5, 20, 50]
    
    # Helper to format row
    def format_row(cols):
        row_str = "| "
        for i, col in enumerate(cols):
            row_str += f"{str(col):<{widths[i]}} | "
        return row_str

    output_file = os.path.join(base_dir, "01-EleutherAI-v1", "benchmarks_config_table.md")
    
    with open(output_file, "w") as f:
        # Title
        f.write("# Benchmark Configuration Overview\n\n")
        
        # Print Header
        f.write(format_row(headers) + "\n")
        # Print Separator
        sep_cols = ["-" * w for w in widths]
        f.write(format_row(sep_cols) + "\n")

        for file_path in files:
            with open(file_path, 'r') as stream:
                try:
                    config = yaml.safe_load(stream)
                    stage = config.get('stage', 'Unknown')
                    
                    for bench in config.get('benchmarks', []):
                        if not bench.get('enabled', True):
                            continue
                            
                        name = bench.get('name', 'Unknown')
                        phases = bench.get('phases', [])
                        
                        # Determine Type
                        b_type = "Custom" if "custom_script" in bench else "Harness"
                        
                        # Determine Shots
                        shots = bench.get('shots', 0)
                        
                        # Determine CoT
                        cot = "Yes" if bench.get('cot', False) else "No"
                        
                        # Determine Paradigm/Mode
                        paradigm = bench.get('paradigm', "")
                        mode = bench.get('mode', "")
                        pm_val = paradigm if paradigm else mode
                        if not pm_val:
                            pm_val = "-"
                        
                        # Consolidate task definitions
                        task_list = []
                        if 'tasks' in bench:
                            task_list = bench['tasks']
                        elif 'subjects' in bench:
                            task_list = bench['subjects']
                        elif 'subset' in bench:
                            task_list = [bench['subset']]
                        else:
                            task_list = ["(Default/All)"]
                        
                        # Truncate or format long task lists to fit widely
                        tasks_str = ", ".join(task_list)
                        
                        for phase in phases:
                            row = [stage, phase, name, b_type, shots, cot, pm_val, tasks_str]
                            f.write(format_row(row) + "\n")
                            
                except Exception as exc:
                    print(f"Error parsing {file_path}: {exc}")
    
    print(f"Successfully generated markdown table at: {output_file}")

if __name__ == "__main__":
    generate_table()
