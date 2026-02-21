import argparse
import yaml
import os
import sys
import logging
import torch
import json
import time
from datetime import datetime

# Add src to path to import eval_runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import eval_runner

# Result of consolidation: Reporting is now handled by eval_runner.py

def setup_orchestrator_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    return logging.getLogger("Orchestrator")

def resolve_task_type(task_name):
    """
    Heuristic to determine if a task is OLMES, Harness, or Custom.
    """
    # Detect JSON-formatted tasks (Usually OLMES/oe-eval overrides)
    if (isinstance(task_name, str) and task_name.strip().startswith("{")) or isinstance(task_name, dict):
        return "olmes"

    # Standardize hyphen vs underscore for consistency
    normalized_name = task_name.replace("-", "_")
    
    # OLMES tasks
    if "olmo3:" in task_name or "::olmes" in task_name or ":rc" in task_name or ":mc" in task_name:
        return "olmes"
    
    # Custom Indic/Special scripts
    custom_scripts = {
        "indic_glue": "01-EleutherAI-v1/src/custom-scripts/indic_glue.py",
        "indic_qa": "01-EleutherAI-v1/src/custom-scripts/indic_qa.py",
        "indic_bias": "01-EleutherAI-v1/src/custom-scripts/indic_bias.py",
        "simpleqa": "01-EleutherAI-v1/src/custom-scripts/simpleqa.py",
        "humaneval": "01-EleutherAI-v1/src/custom-scripts/humaneval.py",
        "ruler": "01-EleutherAI-v1/src/custom-scripts/ruler.py",
        "leval": "01-EleutherAI-v1/src/custom-scripts/leval.py"
    }
    if normalized_name in custom_scripts:
        return ("custom", custom_scripts[normalized_name])
    
    # Standard Harness tasks
    harness_tasks = [
        "mmlu", "gsm8k", "bbh", "arc_challenge", "blimp", 
        "truthfulqa", "hellaswag", "winogrande", "piqa", "gpqa"
    ]
    
    # RULER & LongBench (Redirect to Custom)
    if "ruler_" in normalized_name or "longbench_" in normalized_name or "niah_multikey" in normalized_name:
        # These are handled by specialized scripts (ruler.py or needle_in_haystack.py)
        if "ruler_" in normalized_name or "longbench_" in normalized_name:
            return ("custom", "01-EleutherAI-v1/src/custom-scripts/ruler.py")
        else:
            return ("custom", "01-EleutherAI-v1/src/custom-scripts/needle_in_haystack.py")

    # Custom Context Length Benchmarks (niah_4k, niah_8k, etc.)
    if "niah_" in normalized_name:
         return ("custom", "01-EleutherAI-v1/src/custom-scripts/needle_in_haystack.py")
         
    if normalized_name in harness_tasks or any(h in normalized_name for h in harness_tasks):
        return "harness"
    
    # Fallback to OLMES as it's the primary engine for this config
    return "olmes"

def main():
    parser = argparse.ArgumentParser(description="OLMES Pipeline Orchestrator")
    parser.add_argument("--config", type=str, required=True, help="Path to stage config (e.g. pretrain_1b.yaml)")
    parser.add_argument("--stage", type=str, default="pretrain", help="Stage name for tracking")
    parser.add_argument("--model_args", type=str, required=True, help="HF model args (pretrained=...)")
    parser.add_argument("--limit", type=int, default=None, help="Global limit override")
    parser.add_argument("--sample", action="store_true", help="Run in sample mode: limit all tasks to 2 examples")
    # Detect best available device
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument("--device", type=str, default=default_device, help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--batch_size", type=int, default=1, help="Global batch size")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers (for OLMES only)")
    
    args = parser.parse_args()
    
    # Handle sample mode
    if args.sample and args.limit is None:
        args.limit = 2
        
    setup_orchestrator_logging()
    logger = logging.getLogger("Orchestrator")
    eval_logger = logging.getLogger("eval_runner")

    # 1. Load config
    if not os.path.exists(args.config):
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    stages_config = config.get("stages", {})
    if args.stage not in stages_config:
        logger.error(f"Stage '{args.stage}' not found in {args.config}")
        sys.exit(1)

    # 2. Setup run directory
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.get("defaults", {}).get("output_root", "./benchmark-results"), args.stage, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subfolders for raw logs
    for engine in ["olmes_raw", "harness_raw", "custom_raw", "reports"]:
        os.makedirs(os.path.join(run_dir, engine), exist_ok=True)
    
    pipeline_start = time.time()
    logger.info(f"Starting pipeline stage: {args.stage}")
    logger.info(f"Run directory: {run_dir}")
    
    # Check for HuggingFace token (needed for gated datasets like Indic-Bias)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        logger.info("🔑 HF_TOKEN detected — gated datasets (e.g., Indic-Bias) will be accessible.")
    else:
        logger.warning("⚠️  HF_TOKEN not set — benchmarks requiring gated datasets will fail. "
                       "Set it with: export HF_TOKEN=\"hf_your_token_here\"")
    
    # Initialize eval_runner logging (redirects execution logs to run_dir)
    eval_runner.setup_logging(run_dir, run_timestamp)
    
    # 3. Ensure Vendor
    eval_runner.ensure_olmes_vendor(logger)
    
    # 4. Flatten tasks and identify their types
    benchmarks_to_run = []
    
    for group in stages_config.get(args.stage, {}).get("benchmarks", []):
        group_name = group.get("name", "unnamed_group")
        
        # We handle groups by splitting them by engine
        # This prevents redundant lm-eval calls when multiple harness tasks are in one group
        tasks_by_engine = {}
        for t_name in group.get("tasks", []):
            t_type = resolve_task_type(t_name)
            if t_type not in tasks_by_engine:
                tasks_by_engine[t_type] = []
            tasks_by_engine[t_type].append(t_name)
            
        for engine_type, engine_tasks in tasks_by_engine.items():
            for t_name in engine_tasks:
                # For OLMES and Custom, we still treat them as individual benchmarks for now 
                # (OLMES will be batched later in the execution loop)
                if engine_type == "olmes" or (isinstance(engine_type, tuple) and engine_type[0] == "custom"):
                    b_def = {
                        "name": f"{group_name}:{t_name}",
                        "enabled": True,
                        "limit": group.get("limit"),
                        "subjects": group.get("subjects"),
                        "subset": group.get("subset"),
                    }
                    if engine_type == "olmes":
                        b_def["olmes_task"] = t_name
                    else:
                        b_def["custom_script"] = engine_type[1]
                    benchmarks_to_run.append(b_def)
                
                # For Harness, we combine all tasks for the same engine within a group into ONE call
                # But only do this if it's the first time we see a harness task for this group
                elif engine_type == "harness":
                    # Check if we already created a harness bundle for this group
                    existing = next((b for b in benchmarks_to_run if b.get("name") == group_name and b.get("harness_task")), None)
                    if not existing:
                        b_def = {
                            "name": group_name,
                            "enabled": True,
                            "harness_task": engine_tasks[0], # Primary task/base
                            "tasks": engine_tasks,          # Full list
                            "limit": group.get("limit"),
                            "subjects": group.get("subjects"),
                        }
                        benchmarks_to_run.append(b_def)
                    break # Skip the rest of engine_tasks as they are now bundled

    # 5. Execute via eval_runner logic
    results = {
        "metadata": {
            "stage": args.stage,
            "timestamp": datetime.now().isoformat(),
            "model_args": args.model_args,
            "device": args.device,
            "limit": args.limit,
            "batch_size": args.batch_size,
        },
        "benchmarks": [],
    }
    
    output_json_path = os.path.join(run_dir, "incremental_results.json")
    
    benchmark_timings = []  # List of (name, duration_seconds)
    benchmarks_start = time.time()
    
    # NEW: Grouping logic for OLMES tasks
    processed_indices = set()
    for i, b in enumerate(benchmarks_to_run):
        if i in processed_indices:
            continue
            
        current_limit = args.limit if args.limit is not None else b.get("limit")
        
        # Determine if this is a candidate for OLMES batching
        if b.get("olmes_task"):
            batch = [b]
            processed_indices.add(i)
            
            # Look ahead for more OLMES tasks with same limit
            for j in range(i + 1, len(benchmarks_to_run)):
                next_b = benchmarks_to_run[j]
                next_limit = args.limit if args.limit is not None else next_b.get("limit")
                
                if next_b.get("olmes_task") and next_limit == current_limit:
                    batch.append(next_b)
                    processed_indices.add(j)
                else:
                    # Break run on first non-matching task
                    break
            
            batch_names = ", ".join([batch_item.get("olmes_task") for batch_item in batch])
            logger.info(f"\n⏱  Starting OLMES Batch ({len(batch)} tasks): {batch_names}")
            bench_start = time.time()
            
            # Execute batch
            batch_results = eval_runner.run_olmes_benchmark(
                batch, args.model_args, eval_logger, run_dir, 
                limit=current_limit, device=args.device, batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            bench_elapsed = time.time() - bench_start
            
            # Handle results (could be a list if run_olmes_benchmark returned a list)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]
                
            for res in batch_results:
                res["duration_seconds"] = round(bench_elapsed / len(batch), 2)
                results["benchmarks"].append(res)
                benchmark_timings.append((res["name"], res["duration_seconds"]))
            
            logger.info(f"⏱  Finished OLMES Batch in {bench_elapsed:.1f}s")
            
        else:
            # Standard single-task run (Harness or Custom)
            bench_name = b.get("harness_task") or b.get("custom_script") or b["name"]
            logger.info(f"\n⏱  Starting benchmark: {bench_name}")
            bench_start = time.time()
            
            if b.get("harness_task"):
                res = eval_runner.run_harness_benchmark(
                    b, args.model_args, eval_logger, run_dir, 
                    limit=current_limit, device=args.device, batch_size=args.batch_size
                )
            elif b.get("custom_script"):
                res = eval_runner.run_custom_benchmark(
                    b, args.model_args, eval_logger, run_dir, 
                    limit=current_limit, device=args.device, batch_size=args.batch_size
                )
            else:
                # Fallback
                res = {"name": bench_name, "status": "skipped", "reason": "Unknown task type"}
                
            bench_elapsed = time.time() - bench_start
            benchmark_timings.append((bench_name, bench_elapsed))
            logger.info(f"⏱  Finished benchmark: {bench_name} in {bench_elapsed:.1f}s")
            
            res["duration_seconds"] = round(bench_elapsed, 2)
            results["benchmarks"].append(res)
            processed_indices.add(i)

        # Periodic save
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
    
    benchmarks_elapsed = time.time() - benchmarks_start

    report_start = time.time()
    report_path = eval_runner.generate_summary_report(
        results, run_dir, capability_map=config.get("buckets", {}), 
        baselines=eval_runner.get_baselines(benchmarks_to_run)
    )
    report_elapsed = time.time() - report_start
    pipeline_elapsed = time.time() - pipeline_start

    # Store timing in results metadata
    results["metadata"]["timing"] = {
        "benchmark_timings": [{"name": n, "duration_s": round(d, 2)} for n, d in benchmark_timings],
        "total_benchmarks_s": round(benchmarks_elapsed, 2),
        "report_generation_s": round(report_elapsed, 2),
        "total_pipeline_s": round(pipeline_elapsed, 2),
    }
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Print timing summary
    def fmt_time(s):
        if s < 60:
            return f"{s:.1f}s"
        elif s < 3600:
            return f"{s/60:.1f}m ({s:.0f}s)"
        else:
            return f"{s/3600:.1f}h ({s/60:.0f}m)"

    logger.info(f"\n{'='*60}")
    logger.info(f"⏱  TIMING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Benchmark':<45} {'Duration':>12}")
    logger.info(f"{'-'*45} {'-'*12}")
    for name, dur in benchmark_timings:
        logger.info(f"{name:<45} {fmt_time(dur):>12}")
    logger.info(f"{'-'*45} {'-'*12}")
    logger.info(f"{'Total benchmarks':<45} {fmt_time(benchmarks_elapsed):>12}")
    logger.info(f"{'Report generation':<45} {fmt_time(report_elapsed):>12}")
    logger.info(f"{'Total pipeline':<45} {fmt_time(pipeline_elapsed):>12}")
    logger.info(f"{'='*60}")

    logger.info(f"\n--- Stage Complete: {args.stage} ---")
    logger.info(f"Report: {report_path}")
    logger.info(f"Final results (JSON): {os.path.join(run_dir, 'reports', 'final_results.json')}")

if __name__ == "__main__":
    main()
