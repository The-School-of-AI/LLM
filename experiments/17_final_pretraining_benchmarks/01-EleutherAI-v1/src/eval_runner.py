import yaml
import json
import logging
import argparse
import subprocess
import sys
import os
from datetime import datetime
from typing import Optional

# Check if running in Google Colab
IS_COLAB = os.path.exists('/content') and os.path.exists('/usr/local/lib/python3.12/dist-packages')

def check_torchvision_nms(logger):
    """
    Proactively checks if torchvision and its NMS operation are valid.
    Specifically targets the common 'operator torchvision::nms does not exist' error.
    """
    try:
        import torch
        import torchvision
        # Attempt to access nms
        _ = torchvision.ops.nms
        return True
    except (ImportError, AttributeError, RuntimeError) as e:
        logger.warning(f"  [Auto-Heal] Detected torchvision/torch sync issue: {str(e)}")
        return False

def sync_colab_dependencies(logger):
    """
    Explicitly synchronizes torch and torchvision in Colab to prevent NMS errors.
    """
    if not IS_COLAB:
        return

    logger.info("  [Colab] Synchronizing torch and torchvision...")
    try:
        # Specifically target the latest stable versions that are compatible
        # In Colab, we often need to force-reinstall torchvision after updating torch
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("  [Colab] Synchronization complete.")
    except Exception as e:
        logger.error(f"  [Error] Failed to sync Colab dependencies: {str(e)}")


def setup_logging(run_dir, timestamp):
    log_dir = os.path.join(run_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, "execution.log")

    # Configure logging
    logger = logging.getLogger()
    # Clear existing handlers if any (to avoid duplicate logs in same process)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path


def generate_summary_report(results, run_dir, baselines=None):
    report_dir = os.path.join(run_dir, "reports")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, "summary_report.md")
    
    if baselines is None:
        baselines = {}

    with open(report_path, "w") as f:
        f.write("# Evaluation Summary Report\n\n")
        f.write(f"- **Stage**: {results['metadata']['stage'].upper()}\n")
        f.write(f"- **Phase**: {results['metadata']['phase'].upper()}\n")
        f.write(f"- **Timestamp**: {results['metadata']['timestamp']}\n")
        f.write(f"- **Model Args**: `{results['metadata']['model_args']}`\n")
        f.write(f"- **Device**: `{results['metadata']['device']}`\n")
        f.write(f"- **Limit**: {results['metadata']['limit']}\n\n")

        # ---------------------------------------------------------
        # Pre-process: Calculate Comparative Baselines (Generic)
        # ---------------------------------------------------------
        if baselines:
            f.write(f"### Comparative Baselines\n")
            f.write(f"Metrics common to all stages for longitudinal tracking.\n\n")
            
            for bench_match, baseline_info in baselines.items():
                label = baseline_info.get("label", f"{bench_match} Baseline")
                target_tasks = baseline_info.get("tasks", [])
                
                baseline_score = None
                for b in results["benchmarks"]:
                    if b["name"] == bench_match:
                        # Case 1: Result has subtasks (multiple subjects run)
                        if b.get("subtasks"):
                            scores = [st["score"] for st in b["subtasks"] if st["task"] in target_tasks]
                            if scores:
                                baseline_score = sum(scores) / len(scores)
                        # Case 2: Result is a single task, check if it matches a baseline task
                        elif b.get("task") in target_tasks:
                             baseline_score = b.get("score")
                
                if baseline_score is not None:
                    display_val = f"{baseline_score:.4f}" if isinstance(baseline_score, (int, float)) else str(baseline_score)
                    f.write(f"- **{label}**: {display_val}\n")
            f.write("\n")

        f.write("## Benchmark Overview\n\n")
        f.write("| Benchmark | Type | Status | Agg. Score |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

        for b in results["benchmarks"]:
            score = b.get("score", "N/A")
            if isinstance(score, (int, float)):
                score = f"{score:.4f}"

            f.write(
                f"| **{b['name']}** | {b.get('type', 'N/A')} | {b['status']} | **{score}** |\n"
            )

        # CSV section remains for data export but report is kept high-level
        f.write("\n\n---\n*Detailed subtask data is available in `incremental_results.json` and the CSV section below.*\n")

        # ---------------------------------------------------------
        # Add CSV-Friendly Section for Comparative Analysis
        # ---------------------------------------------------------
        f.write("\n\n## Competitive Analysis Data (CSV Format)\n")
        f.write("Use this data to compare across multiple runs/models.\n\n")
        f.write("```csv\n")
        f.write("Stage,Phase,Model,Timestamp,Benchmark,Subtask,Metric,Score\n")

        stage = results["metadata"]["stage"]
        phase = results["metadata"]["phase"]
        timestamp = results["metadata"]["timestamp"]
        # Extract model name from args
        model_name = "unknown"
        if "pretrained=" in results["metadata"]["model_args"]:
            try:
                model_name = (
                    results["metadata"]["model_args"]
                    .split("pretrained=")[1]
                    .split(",")[0]
                )
            except Exception:
                pass

        for b in results["benchmarks"]:
            bench_name = b["name"]

            # Aggregate Row
            agg_score = b.get("score", "N/A")
            f.write(
                f"{stage},{phase},{model_name},{timestamp},{bench_name},AGGREGATE,primary,{agg_score}\n"
            )

            # Subtask Rows - Only write if count is small to avoid junk in CSV too (or make it optional)
            # User specifically complained about junk report.
            if b.get("subtasks") and len(b["subtasks"]) <= 20: 
                for st in b["subtasks"]:
                    task_name = st["task"]
                    score = st.get("score", "N/A")
                    f.write(
                        f"{stage},{phase},{model_name},{timestamp},{bench_name},{task_name},primary,{score}\n"
                    )

        f.write("```\n")


    return report_path


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Check for base configuration inheritance
    base_file = config.get("base_config")
    if base_file:
        config_dir = os.path.dirname(config_path)
        base_path = os.path.join(config_dir, base_file)
        
        if os.path.exists(base_path):
            with open(base_path, "r") as f:
                base_config = yaml.safe_load(f)
            
            # Merge logic: Base benchmarks are the foundation
            # Stage benchmarks override or add by name
            merged_benchmarks_dict = {b["name"]: b for b in base_config.get("benchmarks", [])}
            stage_benchmarks = config.get("benchmarks", [])
            
            # Update with stage-specific overrides
            for b in stage_benchmarks:
                merged_benchmarks_dict[b["name"]] = b
            
            config["benchmarks"] = list(merged_benchmarks_dict.values())

            # Merge logic for baselines (Base + Overrides)
            base_baselines = base_config.get("baselines", {})
            stage_baselines = config.get("baselines", {})
            base_baselines.update(stage_baselines)
            config["baselines"] = base_baselines
            # Copy other fields from stage config if they exist
            # Note: 'stage', 'description' etc remain from the child config
        else:
            print(f"Warning: Base config {base_path} not found.")
            
    return config


def run_harness_benchmark(
    benchmark_info, model_args, logger, run_dir, limit=None, device=None, batch_size="1"
):
    """
    Executes a benchmark using the lm-evaluation-harness.
    Captures aggregate and granular (subject/subset) results.
    Honors 'subjects', 'subset', and 'tasks' refinement from YAML.
    """
    base_task = benchmark_info.get("harness_task")
    shots = benchmark_info.get("shots", 0)

    # Refine task selection based on YAML fields
    task_list = []
    if benchmark_info.get("subjects"):
        task_list = [f"{base_task}_{s}" for s in benchmark_info["subjects"]]
    elif benchmark_info.get("tasks"):
        task_list = [f"{base_task}_{t}" for t in benchmark_info["tasks"]]
    elif benchmark_info.get("subset"):
        task_list = [f"{base_task}_{benchmark_info['subset']}"]
    else:
        task_list = [base_task]

    task_str = ",".join(task_list)

    # Robustness: Strip 'device' from model_args if it exists there to prevent lm-eval crash
    if "device=" in model_args:
        parts = model_args.split(",")
        new_parts = []
        for p in parts:
            if p.startswith("device="):
                if not device:
                    device = p.split("=")[1]
            else:
                new_parts.append(p)
        model_args = ",".join(new_parts)

    # Setup harness raw status folder
    harness_raw_dir = os.path.join(run_dir, "harness_raw")
    if not os.path.exists(harness_raw_dir):
        os.makedirs(harness_raw_dir)

    # Use the first task name for the filename to avoid extremely long paths
    task_filename = task_list[0] if task_list else base_task
    task_output_path = os.path.join(harness_raw_dir, f"{task_filename}.json")

    # Determine python executable (prefer .venv in root or current dir)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_venv = os.path.join(root_dir, ".venv", "bin", "python3")
    cwd_venv = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    
    py_exec = sys.executable
    if os.path.exists(local_venv):
        py_exec = local_venv
    elif os.path.exists(cwd_venv):
        py_exec = cwd_venv

    # Construct lm-eval command
    cmd = [
        py_exec,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        task_str,
        "--num_fewshot",
        str(shots),
        "--batch_size",
        str(batch_size),
        "--output_path",
        task_output_path,
    ]

    if limit:
        cmd += ["--limit", str(limit)]
    if device:
        cmd += ["--device", device]

    logger.info(f"  [Harness] Running: {task_str} (BS={batch_size}, limit={limit})...")
    try:
        # Run process but capture error if needed. Output will go to logger through subprocess.run defaults if not redirected
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logger.info(result.stdout)

        # Determine the actual result file (lm-eval might add a timestamp or create a directory)
        actual_path = task_output_path
        if not os.path.isfile(actual_path):
            # lm-eval 0.4+ often creates a directory structure: output_path/model_name/results_timestamp.json
            # or it might not have created the file at the exact path if it added a timestamp.
            search_base = harness_raw_dir
            if os.path.isdir(task_output_path):
                search_base = task_output_path
            
            found_path = None
            for root, dirs, files in os.walk(search_base):
                # Look for files that look like results (results_*.json or starting with task name)
                matches = [f for f in files if f.endswith(".json") and (f.startswith("results_") or f.startswith(task_filename))]
                if matches:
                    # Sort by modification time to get the latest
                    matches.sort(key=lambda x: os.path.getmtime(os.path.join(root, x)))
                    found_path = os.path.join(root, matches[-1])
                    break # Found the most relevant file in this branch
            
            if found_path:
                actual_path = found_path
            else:
                logger.error(f"  [Error] Could not find results file for {task_str} in {search_base}")
                return {"name": benchmark_info["name"], "status": "failed", "reason": "Results file not found"}

        with open(actual_path, "r") as f:
            full_res = json.load(f)
            results_dict = full_res.get("results", {})

            subtasks = []
            aggregate_score = None

            for t_name, t_res in results_dict.items():
                possible_metrics = [
                    "acc",
                    "exact_match",
                    "acc_norm",
                    "word_perplexity",
                    "byte_perplexity",
                    "bits_per_byte",
                ]
                score = None

                # Try preferred suffixes first
                for m in possible_metrics:
                    for suffix in [
                        ",none",
                        ",remove_whitespace",
                        ",flexible-extract",
                        ",strict-match",
                    ]:
                        key = f"{m}{suffix}"
                        if key in t_res:
                            score = t_res[key]
                            break
                    if score is not None:
                        break

                # Fallback: find any key that contains a possible metric name
                if score is None:
                    for k, v in t_res.items():
                        if (
                            any(m in k for m in possible_metrics)
                            and "_stderr" not in k
                            and isinstance(v, (int, float))
                        ):
                            score = v
                            break

                # Use first available numeric value if still nothing
                if score is None:
                    for k, v in t_res.items():
                        if isinstance(v, (int, float)) and "_stderr" not in k:
                            score = v
                            break

                # If we ran a group/base task, that name will be the key for aggregate
                if t_name == base_task or (
                    len(task_list) == 1 and t_name == task_list[0]
                ):
                    aggregate_score = score
                else:
                    subtasks.append({"task": t_name, "score": score})

            # If no explicit aggregate score found but we have subtasks, use the average or first
            if aggregate_score is None and subtasks:
                aggregate_score = sum(
                    s["score"] for s in subtasks if isinstance(s["score"], (int, float))
                ) / len(subtasks)

            return {
                "name": benchmark_info["name"],
                "type": "harness",
                "task": task_str,
                "score": aggregate_score,
                "subtasks": subtasks,
                "status": "success",
            }
    except subprocess.CalledProcessError as e:
        error_msg = f"Command {e.cmd} failed with exit status {e.returncode}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        logger.error(f"  [Error] {error_msg}")
        return {
            "name": benchmark_info["name"],
            "type": "harness",
            "status": "failed",
            "error": error_msg,
        }
    except Exception as e:
        logger.error(f"  [Error] {str(e)}")
        return {
            "name": benchmark_info["name"],
            "type": "harness",
            "status": "failed",
            "error": str(e),
        }


def run_custom_benchmark(
    benchmark_info, model_args, logger, config_dir=None, limit=None, device=None
):
    """
    Executes a custom benchmark not covered by lm-evaluation-harness.
    """
    name = benchmark_info["name"]
    script = benchmark_info.get("custom_script")

    logger.info(f"  [Custom] Running {name}...")

    # Resolve script path
    resolved_script = script
    if script and not os.path.isabs(script):
        # Try relative to CWD first, then relative to config_dir, then relative to experiment root
        # Structure: <experiment_root>/01-EleutherAI-v1/src/eval_runner.py
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        possible_paths = [
            script,
            os.path.join(config_dir, script) if config_dir else None,
            os.path.join(root_dir, script)
        ]
        
        for p in possible_paths:
            if p and os.path.exists(p):
                resolved_script = p
                break

    if not resolved_script or not os.path.exists(resolved_script):
        logger.warning(
            f"  [Warning] Script {script} not found (tried {resolved_script})"
        )
        return {
            "name": name,
            "type": "custom",
            "status": "failed",
            "error": f"Script {script} not found",
        }

    try:
        # Expect custom script to output JSON to stdout or a specific file
        # Use same python as harness if possible
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_venv = os.path.join(root_dir, ".venv", "bin", "python3")
        cwd_venv = os.path.join(os.getcwd(), ".venv", "bin", "python3")
        
        py_exec = sys.executable
        if os.path.exists(local_venv):
            py_exec = local_venv
        elif os.path.exists(cwd_venv):
            py_exec = cwd_venv

        cmd = [py_exec, resolved_script, "--model_args", model_args]
        
        # Inject context length for NIAH tasks
        if "niah_" in name:
            try:
                # Handle group:task format (e.g. context_window:niah_8k) or standalone
                task_part = name.split(":")[-1] if ":" in name else name
                if "niah_" in task_part:
                    length_str = task_part.split("niah_")[1] # "8k" or "16k"
                    length = int(length_str.replace("k", "")) * 1024
                    cmd += ["--context_length", str(length)]
            except Exception as e:
                logger.warning(f"  [Warning] Failed to parse context length for {name}: {e}")

        if limit:
            cmd += ["--limit", str(limit)]
        if device:
            cmd += ["--device", device]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"  [Error] {str(e)}")
        return {"name": name, "type": "custom", "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Multi-stage LLM Evaluator (Harness + Custom)"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to stage config"
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pretraining", "sft"],
        required=True,
        help="Training phase",
    )
    parser.add_argument(
        "--model_args",
        type=str,
        default="",
        help="Model args for harness (e.g. 'pretrained=eleutherai/polyglot-ko-1.3b')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for evaluation (e.g. 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--trial", action="store_true", help="Rapid trial run (sets --limit 5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing ONLY)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="1",
        help="Batch size for harness (default: 1, use 'auto' for auto-detection)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark-results",
        help="Top-level output directory",
    )

    args = parser.parse_args()

    # Honor --trial flag
    if args.trial and args.limit is None:
        args.limit = 5

    if not os.path.exists(args.config):
        print(f"Error: Config {args.config} not found.")
        return

    config = load_config(args.config)
    stage = config.get("stage", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create structured directory hierarchy: output_dir/stage/phase/timestamp/
    run_dir = os.path.join(args.output_dir, stage, args.phase, timestamp)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    logger, log_path = setup_logging(run_dir, timestamp)

    logger.info(
        f"--- Evaluation Stage: {stage.upper()} | Phase: {args.phase.upper()} ---"
    )
    logger.info(f"Run Directory: {run_dir}")

    results = {
        "metadata": {
            "stage": stage,
            "phase": args.phase,
            "timestamp": datetime.now().isoformat(),
            "model_args": args.model_args,
            "device": args.device,
            "limit": args.limit,
            "batch_size": args.batch_size,
        },
        "benchmarks": [],
    }

    active_benchmarks = [
        b
        for b in config["benchmarks"]
        if b.get("enabled") and args.phase in b.get("phases", [])
    ]

    # Explicitly naming the JSON file as 'incremental_results.json' to emphasize its behavior
    output_json_path = os.path.join(run_dir, "incremental_results.json")

    # Colab specific pre-check
    if IS_COLAB:
        sync_colab_dependencies(logger)
    
    # Proactive Auto-Heal Check
    if not check_torchvision_nms(logger):
        logger.info("  [Auto-Heal] Triggering torchvision recovery...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "torchvision"],
                check=True,
                capture_output=True,
                text=True
            )
            if check_torchvision_nms(logger):
                logger.info("  [Auto-Heal] recovery successful.")
            else:
                logger.warning("  [Auto-Heal] recovery failed. You might need to restart the Colab runtime.")
        except Exception as e:
            logger.warning(f"  [Auto-Heal] recovery script failed: {str(e)}")

    for b in active_benchmarks:
        # Limit handling: CLI/Trial override YAML
        current_limit = args.limit if args.limit is not None else b.get("limit")

        if b.get("harness_task"):
            res = run_harness_benchmark(
                b,
                args.model_args,
                logger,
                run_dir,
                limit=current_limit,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif b.get("custom_script"):
            res = run_custom_benchmark(
                b,
                args.model_args,
                logger,
                config_dir=os.path.dirname(args.config),
                limit=current_limit,
                device=args.device,
            )
        else:
            res = {
                "name": b["name"],
                "status": "skipped",
                "reason": "No harness task or custom script provided",
            }

        results["benchmarks"].append(res)

        # Incremental save: Write current results to file after each task completing
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)

    # Collect baselines from benchmarks for reporting
    baselines = {}
    for b in config.get("benchmarks", []):
        if b.get("baseline"):
            bl = b["baseline"]
            if isinstance(bl, str):
                bl = {"label": bl}
            else:
                bl = bl.copy()

            if not bl.get("tasks"):
                # Derive from subjects, tasks, or subset
                source_tasks = b.get("subjects") or b.get("tasks")
                if source_tasks:
                    if isinstance(source_tasks, str):
                        source_tasks = [source_tasks]
                    prefix = f"{b['harness_task']}_" if b.get("harness_task") else ""
                    bl["tasks"] = [f"{prefix}{t}" for t in source_tasks]
                elif b.get("subset"):
                    prefix = f"{b['harness_task']}_" if b.get("harness_task") else ""
                    bl["tasks"] = [f"{prefix}{b['subset']}"]
                else:
                    # Default: use the harness task name itself
                    bl["tasks"] = [b.get("harness_task", b["name"])]
            
            baselines[b["name"]] = bl

    # Generate human-readable report inside the run directory
    report_path = generate_summary_report(results, run_dir, baselines=baselines)

    logger.info("\nEvaluation Complete.")
    logger.info(f"Primary Output: {output_json_path}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"Summary Report: {report_path}")


if __name__ == "__main__":
    main()
