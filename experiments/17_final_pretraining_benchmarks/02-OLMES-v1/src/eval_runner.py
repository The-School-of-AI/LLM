import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml


def ensure_olmes_vendor(logger):
    """
    Ensures that the OLMES vendor library is present and installed.
    Clones it from GitHub if missing.
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vendor_dir = os.path.join(root_dir, "vendor")
    olmes_dir = os.path.join(vendor_dir, "olmes")

    if os.path.exists(olmes_dir):
        return

    logger.info("  [OLMES] Vendor directory missing. Attempting to clone...")
    if not os.path.exists(vendor_dir):
        os.makedirs(vendor_dir)

    try:
        # Clone official OLMES repository
        subprocess.run(
            ["git", "clone", "https://github.com/allenai/olmes.git", olmes_dir],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"  [OLMES] Successfully cloned to {olmes_dir}")

        # Install in editable mode
        # Determine python executable (prefer .venv)
        venv_python = os.path.join(root_dir, ".venv", "bin", "python3")
        py_exec = venv_python if os.path.exists(venv_python) else sys.executable

        logger.info(f"  [OLMES] Installing vendor package...")
        
        # Try uv pip install first if uv is available
        uv_path = subprocess.run(["which", "uv"], capture_output=True, text=True).stdout.strip()
        
        install_success = False
        if uv_path:
            try:
                logger.info(f"  [OLMES] Using uv to install: {uv_path} pip install -e {olmes_dir}")
                subprocess.run(
                    [uv_path, "pip", "install", "-e", olmes_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
                install_success = True
            except subprocess.CalledProcessError:
                logger.warning("  [OLMES] uv pip install failed, falling back to standard pip...")
        
        if not install_success:
            try:
                logger.info(f"  [OLMES] Using pip to install: {py_exec} -m pip install -e {olmes_dir}")
                subprocess.run(
                    [py_exec, "-m", "pip", "install", "-e", olmes_dir],
                    check=True,
                    capture_output=True,
                    text=True
                )
                install_success = True
            except subprocess.CalledProcessError as e:
                logger.error(f"  [Error] Failed to setup OLMES vendor with pip: {e.stderr}")
                sys.exit(1)

        if install_success:
            logger.info("  [OLMES] Vendor package installed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"  [Error] Failed to setup OLMES vendor: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"  [Error] Unexpected error during OLMES setup: {str(e)}")
        sys.exit(1)


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

    metadata = results["metadata"]
    benchmarks = results["benchmarks"]

    # Extract model name
    model_name = "unknown"
    if "pretrained=" in metadata["model_args"]:
        try:
            model_name = metadata["model_args"].split("pretrained=")[1].split(",")[0]
        except Exception: pass
    elif metadata["model_args"]:
        model_name = metadata["model_args"]

    # Capability Mapping
    capability_map = {
        "Knowledge": ["MMLU", "TriviaQA"],
        "Reasoning": ["ARC-Challenge", "GSM8K", "BBH (Big Bench Hard)", "MATH", "HellaSwag", "Winogrande", "PIQA"],
        "Language Modeling": ["LAMBADA", "BLiMP", "IFEval", "HumanEval"],
        "Safety & Bias": ["TruthfulQA", "Indic-Bias", "HELM Safety"],
        "Experimental/Custom": ["AIME 2025", "IndicGLUE", "IndicQA", "L-Eval", "RULER", "SimpleQA_Verified", "SWE-bench Verified"]
    }

    with open(report_path, "w") as f:
        f.write(f"# Evaluation Report: {model_name}\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n\n")
        success_count = sum(1 for b in benchmarks if b["status"] == "success")
        total_count = len(benchmarks)
        pass_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        f.write(f"| **Metric** | **Value** |\n")
        f.write(f"| :--- | :--- |\n")
        f.write(f"| **Status** | {'✅ PASS' if pass_rate > 80 else '⚠️ PARTIAL' if pass_rate > 50 else '❌ FAIL'} |\n")
        f.write(f"| **Completion** | {success_count}/{total_count} ({pass_rate:.1f}%) |\n")
        f.write(f"| **Timestamp** | {metadata['timestamp']} |\n")
        f.write(f"| **Run Directory** | `{run_dir}` |\n\n")

        # 2. Results by Capability
        f.write("## 2. Capability Benchmarks\n\n")
        
        for capability, names in capability_map.items():
            cab_benchmarks = [b for b in benchmarks if b["name"] in names]
            if not cab_benchmarks: continue

            f.write(f"### {capability}\n\n")
            f.write("| Benchmark | Engine | Status | Score | Details |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for b in cab_benchmarks:
                score_str = f"**{b.get('score', 'N/A')}**" if b["status"] == "success" else "N/A"
                engine = b.get("type", "N/A")
                status_icon = "✅" if b["status"] == "success" else "❌"
                
                # Details (Subtasks)
                subtasks = b.get("subtasks", [])
                details = f"{len(subtasks)} subtasks" if subtasks else "-"
                if subtasks:
                    sub_items = "".join([f"<li>{st['task']}: {st.get('score', 'N/A')}</li>" for st in sorted(subtasks, key=lambda x: x['task'])[:10]])
                    if len(subtasks) > 10: sub_items += "<li>...</li>"
                    details = f"<details><summary>{len(subtasks)} subtasks</summary><ul>{sub_items}</ul></details>"

                f.write(f"| {b['name']} | {engine} | {status_icon} {b['status']} | {score_str} | {details} |\n")
            f.write("\n")

        # 3. Uncategorized Benchmarks (Fallback)
        uncategorized = [b for b in benchmarks if not any(b["name"] in names for names in capability_map.values())]
        if uncategorized:
            f.write("### Other / Uncategorized\n\n")
            f.write("| Benchmark | Engine | Status | Score |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for b in uncategorized:
                score_str = f"**{b.get('score', 'N/A')}**" if b["status"] == "success" else "N/A"
                f.write(f"| {b['name']} | {b.get('type', 'N/A')} | {b['status']} | {score_str} |\n")
            f.write("\n")

        # 4. Failure Analysis
        failures = [b for b in benchmarks if b["status"] == "failed"]
        if failures:
            f.write("## 3. Failure Analysis\n\n")
            for b in failures:
                f.write(f"#### ❌ {b['name']}\n")
                error_snippet = b.get('error', 'Unknown Error')
                f.write(f"```text\n{error_snippet[:500]}\n```\n\n")

        # 5. Metadata & Traceability
        f.write("## 4. Environment & Traceability\n\n")
        f.write(f"- **Model Args**: `{metadata['model_args']}`\n")
        f.write(f"- **Device**: `{metadata['device']}`\n")
        f.write(f"- **Batch Size**: `{metadata['batch_size']}`\n")
        f.write(f"- **Limit**: {metadata['limit']}\n")
        f.write(f"- **Log File**: `execution.log`\n")

        # 6. CSV Data
        f.write("\n## 5. Raw Data (CSV)\n\n")
        f.write("```csv\n")
        f.write("Benchmark,Status,Score,Engine\n")
        for b in benchmarks:
            f.write(f"{b['name']},{b['status']},{b.get('score', 'N/A')},{b.get('type', 'N/A')}\n")
        f.write("```\n")

    return report_path

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    # Also check parent dir (if running from a subdirectory like 01-EleutherAI-v1)
    parent_venv = os.path.join(os.path.dirname(os.getcwd()), ".venv", "bin", "python3")

    py_exec = sys.executable
    if os.path.exists(venv_python):
        py_exec = venv_python
    elif os.path.exists(parent_venv):
        py_exec = parent_venv

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
        
        # If the expected path is a directory or doesn't exist, search recursively
        if not os.path.exists(actual_path) or os.path.isdir(actual_path):
            found_files = []
            search_base = harness_raw_dir
            if os.path.isdir(actual_path):
                search_base = actual_path
            
            for root, dirs, f_list in os.walk(search_base):
                for f in f_list:
                    if f.endswith(".json"):
                        # If we have a target filename, prioritize it
                        if task_filename in f:
                            found_files.append(os.path.join(root, f))
                        else:
                            found_files.append(os.path.join(root, f))
            
            if found_files:
                # Get the most recent one if multiple exist (by name/path)
                found_files.sort()
                actual_path = found_files[-1]

        if not os.path.exists(actual_path) or os.path.isdir(actual_path):
            raise FileNotFoundError(f"Could not find a valid result JSON at {actual_path}")

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
        # Try relative to CWD first, then relative to config_dir
        if not os.path.exists(resolved_script) and config_dir:
            resolved_script = os.path.join(config_dir, script)

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
        venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
        parent_venv = os.path.join(
            os.path.dirname(os.getcwd()), ".venv", "bin", "python3"
        )
        py_exec = sys.executable
        if os.path.exists(venv_python):
            py_exec = venv_python
        elif os.path.exists(parent_venv):
            py_exec = parent_venv

        cmd = [py_exec, resolved_script, "--model_args", model_args]
        if limit:
            cmd += ["--limit", str(limit)]
        if device:
            cmd += ["--device", device]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        logger.error(f"  [Error] {str(e)}")
        return {"name": name, "type": "custom", "status": "failed", "error": str(e)}


def run_olmes_benchmark(
    benchmark_info, model_args, logger, run_dir, limit=None, device=None, batch_size="1"
):
    """
    Executes a benchmark using the OLMES (oe-eval) engine.
    """
    task = benchmark_info.get("olmes_task")
    
    # Setup OLMES raw status folder - Isolate by task to avoid collision/overwriting metrics.json
    task_safe = task.replace("/", "_").replace(":", "_")
    task_raw_dir = os.path.join(run_dir, "olmes_raw", task_safe)
    if not os.path.exists(task_raw_dir):
        os.makedirs(task_raw_dir)

    task_output_path = os.path.join(task_raw_dir, "metrics.json")

    # Determine python executable
    venv_python = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    parent_venv = os.path.join(os.path.dirname(os.getcwd()), ".venv", "bin", "python3")

    py_exec = sys.executable
    if os.path.exists(venv_python):
        py_exec = venv_python
    elif os.path.exists(parent_venv):
        py_exec = parent_venv

    # Extract model path from model_args
    model_path = ""
    if "pretrained=" in model_args:
        model_path = model_args.split("pretrained=")[1].split(",")[0]
    else:
        model_path = model_args # Fallback

    # Construct olmes command
    # Usage: olmes --model <model> --task <task> --output-dir <dir>
    # Determine olmes executable path robustly
    import shutil
    olmes_executable = shutil.which("olmes")
    
    # If not in PATH, check relative to py_exec
    if not olmes_executable:
        olmes_relative = os.path.join(os.path.dirname(py_exec), "olmes")
        if os.path.exists(olmes_relative):
            olmes_executable = olmes_relative

    if olmes_executable:
        cmd = [olmes_executable]
    else:
        # Fallback to module if script not found
        logger.warning("  [OLMES] 'olmes' executable not found in PATH or near python. Falling back to 'python -m oe_eval'.")
        cmd = [py_exec, "-m", "oe_eval"]

    cmd += [
        "--model",
        model_path,
        "--task",
        task,
        "--output-dir",
        task_raw_dir,
    ]

    if limit:
        cmd += ["--limit", str(limit)]
    # Note: OLMES might have different flags for device/batch_size, 
    # but we'll stick to the core ones for now.

    logger.info(f"  [OLMES] Running: {task} (BS={batch_size}, limit={limit})...")
    
    try:
        # PATH Shim: OLMES internal launcher expects a 'python' command.
        # We create a temporary directory with a 'python' symlink pointing to the current interpreter.
        with tempfile.TemporaryDirectory() as tmp_dir:
            python_shim = os.path.join(tmp_dir, "python")
            try:
                # Create a shell script wrapper for the 'python' shim
                # This is more robust than a symlink for virtual environments
                with open(python_shim, "w") as f:
                    f.write(f"#!/bin/sh\nexec '{py_exec}' \"$@\"\n")
                os.chmod(python_shim, 0o755)
            except Exception as e:
                # Fallback for systems where this might fail
                import shutil
                shutil.copy2(py_exec, python_shim)

            # Update environment PATH
            env = os.environ.copy()
            env["PATH"] = f"{tmp_dir}{os.pathsep}{env.get('PATH', '')}"

            # Run OLMES with the PATH shim
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env
            )
            if result.stdout:
                logger.info(result.stdout)

        # OLMES saves aggregate results in metrics.json in the output-dir
        actual_path = os.path.join(task_raw_dir, "metrics.json")

        if not os.path.exists(actual_path):
             # Fallback: search for any json containing the task metrics
             for root, dirs, files in os.walk(task_raw_dir):
                 for file in files:
                     if file.endswith("metrics.json"):
                         actual_path = os.path.join(root, file)
                         break
                 if actual_path: break

        if not actual_path or not os.path.exists(actual_path):
             return {
                "name": benchmark_info["name"],
                "type": "olmes",
                "status": "failed",
                "error": f"Result JSON not found after OLMES run at {task_raw_dir}",
            }

        with open(actual_path, "r") as f:
            res_data = json.load(f)
            
            # OLMES metrics.json can have a 'tasks' list or a structure.
            # Usually it's: {"tasks": [{"alias": "task_name", "metrics": {"primary_score": X}, ...}]}
            score = 0.0
            found_task = False
            
            if "tasks" in res_data and isinstance(res_data["tasks"], list):
                for t_entry in res_data["tasks"]:
                    # Match by alias or task_name
                    if t_entry.get("alias") == task or task in t_entry.get("alias", ""):
                        score = t_entry.get("metrics", {}).get("primary_score", 0.0)
                        found_task = True
                        break
                
                # If still not found, take the first one if there's only one
                if not found_task and len(res_data["tasks"]) == 1:
                    score = res_data["tasks"][0].get("metrics", {}).get("primary_score", 0.0)
                    found_task = True
            
            # Fallback for different OLMES versions/shards
            if not found_task:
                score = res_data.get("results", {}).get(task, {}).get("primary_metric", 0.0)

            return {
                "name": benchmark_info["name"],
                "type": "olmes",
                "task": task,
                "score": score,
                "status": "success",
            }

    except subprocess.CalledProcessError as e:
        error_msg = f"Command {e.cmd} failed with exit status {e.returncode}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        logger.error(f"  [Error] {error_msg}")
        return {
            "name": benchmark_info["name"],
            "type": "olmes",
            "status": "failed",
            "error": error_msg,
        }
    except Exception as e:
        logger.error(f"  [Error] {str(e)}")
        return {
            "name": benchmark_info["name"],
            "type": "olmes",
            "status": "failed",
            "error": str(e),
        }


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

    # Ensure OLMES vendor is present
    ensure_olmes_vendor(logger)

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

    for b in active_benchmarks:
        if b.get("olmes_task"):
            res = run_olmes_benchmark(
                b,
                args.model_args,
                logger,
                run_dir,
                limit=args.limit,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif b.get("harness_task"):
            res = run_harness_benchmark(
                b,
                args.model_args,
                logger,
                run_dir,
                limit=args.limit,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif b.get("custom_script"):
            res = run_custom_benchmark(
                b,
                args.model_args,
                logger,
                config_dir=os.path.dirname(args.config),
                limit=args.limit,
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

    # Generate human-readable report inside the run directory
    report_path = generate_summary_report(results, run_dir)

    logger.info("\nEvaluation Complete.")
    logger.info(f"Primary Output: {output_json_path}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"Summary Report: {report_path}")


if __name__ == "__main__":
    main()
