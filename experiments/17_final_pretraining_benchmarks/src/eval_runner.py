import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

import yaml


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


def generate_summary_report(results, run_dir):
    report_dir = os.path.join(run_dir, "reports")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, "summary_report.md")

    with open(report_path, "w") as f:
        f.write("# Evaluation Summary Report\n\n")
        f.write(f"- **Stage**: {results['metadata']['stage'].upper()}\n")
        f.write(f"- **Phase**: {results['metadata']['phase'].upper()}\n")
        f.write(f"- **Timestamp**: {results['metadata']['timestamp']}\n")
        f.write(f"- **Model Args**: `{results['metadata']['model_args']}`\n")
        f.write(f"- **Device**: `{results['metadata']['device']}`\n")
        f.write(f"- **Batch Size**: `{results['metadata']['batch_size']}`\n")
        f.write(f"- **Limit**: {results['metadata']['limit']}\n\n")

        f.write("## Benchmark Overview\n\n")
        f.write("| Benchmark | Type | Status | Agg. Score | Sub-tasks |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")

        for b in results["benchmarks"]:
            score = b.get("score", "N/A")
            subtask_count = len(b.get("subtasks", []))
            f.write(
                f"| **{b['name']}** | {b.get('type', 'N/A')} | {b['status']} | **{score}** | {subtask_count} |\n"
            )

        f.write("\n\n## Detailed Results (per subject/subset)\n\n")
        f.write("| Task/Subject | Status | Score | Error |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")

        for b in results["benchmarks"]:
            if b["status"] == "failed":
                f.write(
                    f"| **{b['name']} (FAILED)** | error | - | {b.get('error')} |\n"
                )
                continue

            # Header for high-level benchmark
            score = b.get("score", "N/A")
            f.write(
                f"| **{b['name']} (Aggregate)** | {b['status']} | **{score}** | - |\n"
            )

            # Granular sub-tasks
            if b.get("subtasks"):
                # Sort subtasks by name for better readability
                sorted_subtasks = sorted(b["subtasks"], key=lambda x: x["task"])
                for st in sorted_subtasks:
                    s = st.get("score", "N/A")
                    f.write(
                        f"| &nbsp;&nbsp;&nbsp;&nbsp;↳ {st['task']} | {b['status']} | {s} | - |\n"
                    )

    return report_path


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_harness_benchmark(
    benchmark_info,
    model_args,
    logger,
    run_dir,
    limit=None,
    device=None,
    batch_size="1",
    is_test=False,
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

    if is_test:
        logger.info(
            f"  [Simulation] Would run tasks: {task_str}, shots: {shots}, limit: {limit}, device: {device}, batch_size: {batch_size}"
        )
        sim_subtasks = (
            [{"task": t, "score": 0.5} for t in task_list] if len(task_list) > 1 else []
        )
        return {
            "name": benchmark_info["name"],
            "type": "harness",
            "task": task_str,
            "status": "simulation",
            "score": 0.5,
            "subtasks": sim_subtasks,
        }

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

    # Determine python executable (prefer .venv)
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    py_exec = venv_python if os.path.exists(venv_python) else sys.executable

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
        if not os.path.exists(actual_path) and os.path.isdir(harness_raw_dir):
            # Check for files starting with task name in the directory
            files = [
                f
                for f in os.listdir(harness_raw_dir)
                if f.startswith(task_filename) and f.endswith(".json")
            ]
            if files:
                # Get the most recent one if multiple exist
                files.sort()
                actual_path = os.path.join(harness_raw_dir, files[-1])

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


def run_custom_benchmark(benchmark_info, model_args, logger, is_test=False):
    """
    Executes a custom benchmark not covered by lm-evaluation-harness.
    """
    name = benchmark_info["name"]
    script = benchmark_info.get("custom_script")

    if is_test:
        logger.info(f"  [Simulation] Would run custom script for {name} via {script}")
        return {"name": name, "type": "custom", "status": "simulation"}

    logger.info(f"  [Custom] Running {name}...")
    if not script or not os.path.exists(script):
        logger.warning(f"  [Warning] Script {script} not found")
        return {
            "name": name,
            "type": "custom",
            "status": "failed",
            "error": f"Script {script} not found",
        }

    try:
        # Expect custom script to output JSON to stdout or a specific file
        result = subprocess.run(
            ["python3", script, "--model_args", model_args],
            capture_output=True,
            text=True,
            check=True,
        )
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
        "--test", action="store_true", help="Simulation mode (no model loading)"
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
        "--output_dir", type=str, default="results", help="Top-level output directory"
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
            "is_test": args.test,
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

    for b in active_benchmarks:
        if b.get("harness_task"):
            res = run_harness_benchmark(
                b,
                args.model_args,
                logger,
                run_dir,
                limit=args.limit,
                device=args.device,
                batch_size=args.batch_size,
                is_test=args.test,
            )
        elif b.get("custom_script"):
            res = run_custom_benchmark(b, args.model_args, logger, is_test=args.test)
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

    # Generate human-readable report inside the run directory
    report_path = generate_summary_report(results, run_dir)

    logger.info("\nEvaluation Complete.")
    logger.info(f"Primary Output: {output_json_path}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"Summary Report: {report_path}")


if __name__ == "__main__":
    main()
