import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

from infra_utils import (
    IS_COLAB,
    setup_logging,
    ensure_olmes_vendor,
    check_torchvision_nms
)


def get_intermediate_group(subtask_name):
    """
    Heuristic to group subtasks into intermediate task families.
    """
    # Clean up olmes suffix
    name = subtask_name.replace("::olmes", "").replace(":olmes", "")
    
    # OLMES/Harness prefixes
    prefixes = [
        "arc_challenge", "arc_easy", "mmlu_", "bbh_", 
        "codex_humanevalfim", "minerva_math", "gsm8k",
        "boolq", "csqa", "hellaswag", "piqa", "socialiqa", "winogrande",
        "openbookqa", "triviaqa"
    ]
    
    # Check for explicit prefixes first
    for p in prefixes:
        if name.startswith(p):
            # Special case for MMLU subjects: group by subject name
            if p == "mmlu_" or p == "bbh_":
                return name.split(":")[0]
            return p
            
    # Handle OLMES segments (task:segment)
    if ":" in name:
        return name.split(":")[0]
        
    # Handle Harness segments (task_segment)
    if "_" in name:
        # Avoid splitting common task names that use underscores but aren't subject-based
        if not any(name.startswith(p) for p in prefixes):
             # If it's a known task followed by an underscore, it's likely a harness subject
             # But we need to be careful not to over-split.
             # For now, if it's not in our explicit prefixes, we split by first understore.
             return name.split("_")[0]

    return name

def generate_summary_report(results, run_dir, capability_map=None, baselines=None):
    report_dir = os.path.join(run_dir, "reports")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    report_path = os.path.join(report_dir, "summary_report.md")
    if baselines is None:
        baselines = {}

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
    if capability_map is None:
        capability_map = {
            "Knowledge": ["MMLU", "TriviaQA"],
            "Reasoning": ["ARC-Challenge", "GSM8K", "BBH (Big Bench Hard)", "MATH", "HellaSwag", "Winogrande", "PIQA"],
            "Language Modeling": ["LAMBADA", "BLiMP", "IFEval", "HumanEval"],
            "Safety & Bias": ["TruthfulQA", "Indic-Bias", "HELM Safety"],
            "Experimental/Custom": ["AIME 2025", "IndicGLUE", "IndicQA", "L-Eval", "RULER", "SimpleQA_Verified", "SWE-bench Verified"]
        }

    with open(report_path, "w") as f:
        f.write(f"# Evaluation Report: {model_name}\n\n")

        # Longitudinal Tracking (Comparative Baselines)
        if baselines:
            f.write(f"## Comparative Baselines\n")
            f.write(f"Metrics tracked across all stages for longitudinal analysis.\n\n")
            
            for bench_match, baseline_info in baselines.items():
                label = baseline_info.get("label", f"{bench_match} Baseline")
                target_tasks = baseline_info.get("tasks", [])
                
                baseline_score = None
                for b in benchmarks:
                    if b["name"] == bench_match:
                        if b.get("subtasks"):
                            scores = [st["score"] for st in b["subtasks"] if st["task"] in target_tasks]
                            if scores:
                                baseline_score = sum(scores) / len(scores)
                        elif b.get("task") in target_tasks:
                             baseline_score = b.get("score")
                
                if baseline_score is not None:
                    display_val = f"{baseline_score:.4f}" if isinstance(baseline_score, (int, float)) else str(baseline_score)
                    f.write(f"- **{label}**: {display_val}\n")
            f.write("\n")
        
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
            # Match if any target name is a substring of the benchmark name
            cab_benchmarks = [
                b for b in benchmarks 
                if any(target.lower() in b["name"].lower() for target in names)
            ]
            if not cab_benchmarks: continue

            f.write(f"### {capability}\n\n")
            f.write("| Benchmark | Engine | Status | Score | Details |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            
            for b in cab_benchmarks:
                score_str = f"**{b.get('score', 'N/A')}**" if b["status"] == "success" else "N/A"
                engine = b.get("type", "N/A")
                status_icon = "✅" if b["status"] == "success" else "❌"
                
                # Details (Subtasks with Intermediate Grouping)
                subtasks = b.get("subtasks", [])
                details = f"{len(subtasks)} subtasks" if subtasks else "-"
                if subtasks:
                    # Group subtasks
                    groups = {}
                    for st in subtasks:
                        g_name = get_intermediate_group(st["task"])
                        if g_name not in groups:
                            groups[g_name] = []
                        groups[g_name].append(st)
                    
                    sub_items = []
                    for g_name in sorted(groups.keys()):
                        g_subtasks = groups[g_name]
                        if len(g_subtasks) > 1:
                            # Aggregate score for the group
                            g_score = sum(st["score"] for st in g_subtasks if isinstance(st["score"], (int, float))) / len(g_subtasks)
                            g_details = "".join([f"<li>{st['task']}: {st.get('score', 'N/A')}</li>" for st in sorted(g_subtasks, key=lambda x: x['task'])])
                            sub_items.append(f"<li><b>{g_name}</b>: {g_score:.4f} <details><summary>{len(g_subtasks)} variants</summary><ul>{g_details}</ul></details></li>")
                        else:
                            st = g_subtasks[0]
                            sub_items.append(f"<li>{st['task']}: {st.get('score', 'N/A')}</li>")
                    
                    details = f"<details><summary>{len(subtasks)} subtasks in {len(groups)} groups</summary><ul>{''.join(sub_items)}</ul></details>"

                f.write(f"| {b['name']} | {engine} | {status_icon} {b['status']} | {score_str} | {details} |\n")
            f.write("\n")

        # 3. Uncategorized Benchmarks (Fallback)
        uncategorized = [b for b in benchmarks if not any(b["name"] in (names or []) for names in (capability_map or {}).values())]
        if uncategorized:
            f.write("### Other / Uncategorized\n\n")
            f.write("| Benchmark | Engine | Status | Score | Details |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for b in uncategorized:
                score_str = f"**{b.get('score', 'N/A')}**" if b["status"] == "success" else "N/A"
                subtasks = b.get("subtasks", [])
                details = f"{len(subtasks)} subtasks" if subtasks else "-"
                if subtasks:
                    # Group subtasks
                    groups = {}
                    for st in subtasks:
                        g_name = get_intermediate_group(st["task"])
                        if g_name not in groups:
                            groups[g_name] = []
                        groups[g_name].append(st)
                    
                    sub_items = []
                    for g_name in sorted(groups.keys()):
                        g_subtasks = groups[g_name]
                        if len(g_subtasks) > 1:
                            g_score = sum(st["score"] for st in g_subtasks if isinstance(st["score"], (int, float))) / len(g_subtasks)
                            g_details = "".join([f"<li>{st['task']}: {st.get('score', 'N/A')}</li>" for st in sorted(g_subtasks, key=lambda x: x['task'])])
                            sub_items.append(f"<li><b>{g_name}</b>: {g_score:.4f} <details><summary>{len(g_subtasks)} variants</summary><ul>{g_details}</ul></details></li>")
                        else:
                            st = g_subtasks[0]
                            sub_items.append(f"<li>{st['task']}: {st.get('score', 'N/A')}</li>")
                            
                    details = f"<details><summary>{len(subtasks)} subtasks in {len(groups)} groups</summary><ul>{''.join(sub_items)}</ul></details>"
                f.write(f"| {b['name']} | {b.get('type', 'N/A')} | {b['status']} | {score_str} | {details} |\n")
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

    # Save Structured Aggregated and Granular results in same way for comparison
    flat_results = {
        "metadata": metadata,
        "aggregates": {b["name"]: b.get("score") for b in benchmarks if b.get("status") == "success"},
        "granular": {}
    }
    for b in benchmarks:
        if b.get("subtasks"):
            for st in b["subtasks"]:
                flat_results["granular"][st["task"]] = st.get("score")
    
    json_report_path = os.path.join(report_dir, "final_results.json")
    with open(json_report_path, "w") as f:
        json.dump(flat_results, f, indent=4)

    return report_path

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Handle base configuration inheritance
    if base_file := config.get("base_config"):
        base_path = os.path.join(os.path.dirname(config_path), base_file)
        if os.path.exists(base_path):
            with open(base_path, "r") as f:
                base_cfg = yaml.safe_load(f)
            # Merge: stage benchmarks override base benchmarks by 'name'
            merged = {b["name"]: b for b in base_cfg.get("benchmarks", [])}
            merged.update({b["name"]: b for b in config.get("benchmarks", [])})
            config["benchmarks"] = list(merged.values())
        else:
            print(f"Warning: Base config {base_path} not found.")
            
    return config

def get_baselines(benchmarks):
    """
    Helper to extract baseline task lists from benchmark definitions.
    Shared between eval_runner and pipeline_runner.
    """
    baselines = {}
    for b in benchmarks:
        if not (bl_raw := b.get("baseline")): continue
        
        bl = {"label": bl_raw} if isinstance(bl_raw, str) else bl_raw.copy()
        if not bl.get("tasks"):
            prefix = f"{b.get('harness_task', b.get('olmes_task', ''))}_"
            if source := (b.get("subjects") or b.get("tasks")):
                bl["tasks"] = [f"{prefix}{t}" for t in ([source] if isinstance(source, str) else source)]
            elif subset := b.get("subset"):
                bl["tasks"] = [f"{prefix}{subset}"]
            else:
                bl["tasks"] = [b.get("harness_task", b.get("olmes_task", b["name"]))]
        baselines[b["name"]] = bl
    return baselines


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
    
    # If base_task is complex (JSON), use it as is
    is_json_task = (isinstance(base_task, str) and base_task.strip().startswith("{")) or isinstance(base_task, dict)
    
    if is_json_task:
        task_list = [base_task]
    elif benchmark_info.get("subjects"):
        # For subjects, we usually WANT the prefix, e.g. mmlu -> mmlu_anatomy
        task_list = [f"{base_task}_{s}" if not str(s).startswith(base_task) else s for s in benchmark_info["subjects"]]
    elif benchmark_info.get("tasks"):
        # For bundled tasks in a group, they might already be full names
        task_list = []
        for t in benchmark_info["tasks"]:
            t_str = str(t)
            # Heuristic: if it looks like a full task name, don't prefix
            if ":" in t_str or t_str.startswith("mmlu_") or t_str.startswith("bbh_") or t_str == base_task:
                task_list.append(t_str)
            else:
                task_list.append(f"{base_task}_{t_str}")
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

    # Determine python executable (prefer local .venv in the experiment dir)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_venv = os.path.join(root_dir, ".venv", "bin", "python3")
    cwd_venv = os.path.join(os.getcwd(), ".venv", "bin", "python3")
    
    py_exec = sys.executable
    if os.path.exists(local_venv):
        py_exec = local_venv
    elif os.path.exists(cwd_venv):
        py_exec = cwd_venv

    # Construct lm-eval command flags
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
        # Run process
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
    benchmark_info, model_args, logger, run_dir, config_dir=None, limit=None, device=None, batch_size=None
):
    """
    Executes a custom benchmark not covered by lm-evaluation-harness.
    """
    name = benchmark_info["name"]
    script = benchmark_info.get("custom_script")

    logger.info(f"  [Custom] Running {name}...")

    # Setup Custom raw status folder
    task_safe = name.replace("/", "_").replace(":", "_")
    custom_raw_dir = os.path.join(run_dir, "custom_raw", task_safe)
    if not os.path.exists(custom_raw_dir):
        os.makedirs(custom_raw_dir)

    # Resolve script path
    resolved_script = script
    if script and not os.path.isabs(script):
        # Try relative to CWD first, then relative to config_dir, then relative to experiment root
        # Structure: <experiment_root>/02-OLMES-v1/src/eval_runner.py
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
        logger.warning(f"  [Warning] Script {script} not found (tried {resolved_script})")
        return {
            "name": name,
            "type": "custom",
            "status": "failed",
            "error": f"Script {script} not found",
        }

    try:
        # Determine python executable (prefer local .venv in the experiment dir)
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
                    if "k" in length_str:
                        length = int(length_str.replace("k", "")) * 1024
                        cmd += ["--context_length", str(length)]
                    elif length_str.isdigit():
                        cmd += ["--context_length", length_str]
                    else:
                        logger.warning(f"  [Warning] Task '{task_part}' does not specify a simple numeric context length. Skipping --context_length arg.")
            except Exception:
                logger.warning(f"  [Warning] Task '{task_part}' does not specify a simple numeric context length. Skipping --context_length arg.")

        if limit:
            cmd += ["--limit", str(limit)]
        if device:
            cmd += ["--device", device]
        if batch_size:
            cmd += ["--batch_size", str(batch_size)]

        env = os.environ.copy()
        python_dir = os.path.dirname(py_exec)
        env["PATH"] = f"{python_dir}{os.pathsep}{env.get('PATH', '')}"

        stdout_path = os.path.join(custom_raw_dir, "stdout.log")
        stderr_path = os.path.join(custom_raw_dir, "stderr.log")

        with open(stdout_path, "w") as f_out, open(stderr_path, "w") as f_err:
            process = subprocess.Popen(
                cmd, 
                stdout=f_out, 
                stderr=f_err, 
                text=True, 
                env=env,
                cwd=root_dir
            )
            process.wait()

        # Check return code
        if process.returncode != 0:
            logger.error(f"  [Custom] Script failed with exit code {process.returncode}")
            return {
                "name": name,
                "type": "custom",
                "status": "failed",
                "error": f"Script failed (exit {process.returncode}). Check logs in {custom_raw_dir}"
            }

        # Read stdout to parse JSON result
        with open(stdout_path, "r") as f:
            stdout_content = f.read()

        # Attempt to parse JSON from stdout
        res_data = {}
        for line in stdout_content.splitlines():
            if line.strip().startswith("{") and line.strip().endswith("}"):
                try:
                    res_data = json.loads(line)
                except:
                    continue
        
        if not res_data:
            return {"name": name, "type": "custom", "status": "failed", "error": "No JSON output found in custom script stdout"}

        return res_data

    except Exception as e:
        logger.error(f"  [Error] {str(e)}")
        return {"name": name, "type": "custom", "status": "failed", "error": str(e)}


def run_olmes_benchmark(
    benchmark_info, model_args, logger, run_dir, limit=None, device=None, batch_size="1", num_workers=1
):
    """
    Executes a benchmark using the OLMES (oe-eval) engine.
    Supports single task or batch of tasks.
    """
    is_batch = isinstance(benchmark_info, list)
    if is_batch:
        tasks = [b.get("olmes_task") for b in benchmark_info]
        bench_names = [b["name"] for b in benchmark_info]
        primary_name = f"Batch:{len(tasks)}_tasks"
    else:
        tasks = [benchmark_info.get("olmes_task")]
        bench_names = [benchmark_info["name"]]
        primary_name = benchmark_info["name"]

    # Setup OLMES raw status folder
    # For batches, we use a generic name or the first task's name
    # Setup OLMES raw status folder
    # For batches, we use a generic name or the first task's name
    if is_batch:
        folder_name = "batch_" + datetime.now().strftime("%H%M%S")
    else:
        task_name = tasks[0]
        if task_name.startswith("{"): # JSON task
            folder_name = "json_task_" + datetime.now().strftime("%H%M%S")
        else:
            folder_name = task_name.replace("/", "_").replace(":", "_")
    task_raw_dir = os.path.join(run_dir, "olmes_raw", folder_name)
    if not os.path.exists(task_raw_dir):
        os.makedirs(task_raw_dir)

    # Determine python executable
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_venv = os.path.join(root_dir, ".venv", "bin", "python3")
    py_exec = local_venv if os.path.exists(local_venv) else sys.executable

    # Extract model path
    model_path = ""
    if "pretrained=" in model_args:
        model_path = model_args.split("pretrained=")[1].split(",")[0]
    else:
        model_path = model_args

    import shutil
    olmes_executable = shutil.which("olmes")
    if not olmes_executable:
        olmes_relative = os.path.join(os.path.dirname(py_exec), "olmes")
        if os.path.exists(olmes_relative):
            olmes_executable = olmes_relative

    cmd = [olmes_executable] if olmes_executable else [py_exec, "-m", "oe_eval"]
    
    cmd += ["--model", model_path]
    
    # Add all tasks
    cmd += ["--task"] + tasks
        
    cmd += ["--output-dir", task_raw_dir]

    if limit:
        cmd += ["--limit", str(limit)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    
    # Pass GPU count when running on CUDA
    if device and "cuda" in str(device):
        cmd += ["--gpus", "1"]

    cmd += ["--num-workers", str(num_workers)]

    logger.info(f"  [OLMES] Running {len(tasks)} tasks (limit={limit}, batch_size={batch_size})...")
    
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            python_shim = os.path.join(tmp_dir, "python")
            with open(python_shim, "w") as f:
                f.write(f"#!/bin/sh\nexec '{py_exec}' \"$@\"\n")
            os.chmod(python_shim, 0o755)

            env = os.environ.copy()
            env["PATH"] = f"{tmp_dir}{os.pathsep}{env.get('PATH', '')}"

            subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

        metrics_path = os.path.join(task_raw_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return {"name": primary_name, "type": "olmes", "status": "failed", "error": "metrics.json missing"}

        with open(metrics_path, "r") as f:
            res_data = json.load(f)
            
        # Parse results for each task
        task_results = []
        parsed_tasks = res_data.get("tasks", [])
        
        # OLMES might return more tasks than requested due to subtasks. 
        # We try to map them back.
        for i, original_task in enumerate(tasks):
            # Find matching result
            matching_result = None
            for pt in parsed_tasks:
                # check alias or task_name
                alias = pt.get("alias", "")
                if original_task in alias or alias in original_task:
                    matching_result = pt
                    break
            
            score = 0.0
            subtasks = []
            if matching_result:
                score = matching_result.get("metrics", {}).get("primary_score", 0.0)
                # If there are sub-metrics, collect them
                for m_name, m_val in matching_result.get("metrics", {}).items():
                    if m_name != "primary_score":
                        subtasks.append({"task": m_name, "score": m_val})

            task_results.append({
                "name": bench_names[i],
                "type": "olmes",
                "status": "success",
                "score": score,
                "subtasks": subtasks,
                "raw_dir": task_raw_dir
            })
            
        return task_results if is_batch else task_results[0]

    except subprocess.CalledProcessError as e:
        logger.error(f"  [OLMES] Execution failed: {e.stderr}")
        err_msg = e.stderr or str(e)
        if is_batch:
            return [{"name": n, "type": "olmes", "status": "failed", "error": err_msg} for n in bench_names]
        return {"name": primary_name, "type": "olmes", "status": "failed", "error": err_msg}
    except Exception as e:
        logger.error(f"  [OLMES] Unexpected error: {str(e)}")
        if is_batch:
            return [{"name": n, "type": "olmes", "status": "failed", "error": str(e)} for n in bench_names]
        return {"name": primary_name, "type": "olmes", "status": "failed", "error": str(e)}


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
        # Limit handling: CLI/Trial override YAML
        current_limit = args.limit if args.limit is not None else b.get("limit")

        if b.get("olmes_task"):
            res = run_olmes_benchmark(
                b,
                args.model_args,
                logger,
                run_dir,
                limit=current_limit,
                device=args.device,
                batch_size=args.batch_size,
            )
        elif b.get("harness_task"):
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
                batch_size=args.batch_size,
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
    report_path = generate_summary_report(results, run_dir, baselines=get_baselines(config.get("benchmarks", [])))

    logger.info("\nEvaluation Complete.")
    logger.info(f"Primary Output: {output_json_path}")
    logger.info(f"Logs: {log_path}")
    logger.info(f"Summary Report: {report_path}")


if __name__ == "__main__":
    main()
