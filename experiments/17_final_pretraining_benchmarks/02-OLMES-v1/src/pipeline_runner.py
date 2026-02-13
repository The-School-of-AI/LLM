import argparse
import yaml
import os
import sys
import logging
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
    # OLMES tasks
    if "olmo3:" in task_name or "::olmes" in task_name or ":rc" in task_name or ":mc" in task_name:
        return "olmes"
    
    # Custom Indic/Special scripts
    custom_scripts = {
        "indic_glue": "01-EleutherAI-v1/src/custom-scripts/indic_glue.py",
        "indic_qa": "01-EleutherAI-v1/src/custom-scripts/indic_qa.py",
        "indic_bias": "01-EleutherAI-v1/src/custom-scripts/indic_bias.py",
        "simpleqa": "01-EleutherAI-v1/src/custom-scripts/simpleqa.py",
        "humaneval": "01-EleutherAI-v1/src/custom-scripts/humaneval.py"
    }
    if task_name in custom_scripts:
        return ("custom", custom_scripts[task_name])
    
    # Standard Harness tasks
    harness_tasks = [
        "mmlu", "gsm8k", "bbh", "arc_challenge", "blimp", 
        "truthfulqa", "hellaswag", "winogrande", "piqa", 
        "lambada", "realtoxicityprompts"
    ]
    if task_name in harness_tasks or "mmlu" in task_name:
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
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--batch_size", type=str, default="1", help="Execution batch size")
    
    args = parser.parse_args()
    
    # Handle sample mode
    if args.sample and args.limit is None:
        args.limit = 2
        
    logger = setup_orchestrator_logging()
    eval_logger = logging.getLogger("eval_runner")

    # 1. Load configuration
    with open(args.config, "r") as f:
        orchestrator_config = yaml.safe_load(f)
    
    if args.stage not in orchestrator_config.get("stages", {}):
        logger.error(f"Stage '{args.stage}' not found in {args.config}")
        sys.exit(1)
        
    stage_data = orchestrator_config["stages"][args.stage]
    
    # 2. Setup run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("benchmark-results", args.stage, timestamp)
    if not os.path.exists(run_dir): os.makedirs(run_dir)
    
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
    eval_runner.setup_logging(run_dir, timestamp)
    
    # 3. Ensure Vendor
    eval_runner.ensure_olmes_vendor(eval_logger)
    
    # 4. Resolve and Run Benchmarks
    benchmarks_to_run = []
    groups = stage_data.get("benchmarks", [])
    
    for group in groups:
        if not group.get("enabled", True):
            continue
            
        group_name = group["name"]
        tasks = group.get("tasks", [])
        
        for t_name in tasks:
            t_type = resolve_task_type(t_name)
            
            b_def = {
                "name": f"{group_name}:{t_name}",
                "enabled": True,
                "phases": [args.stage], 
                "baseline": group.get("baseline"),
                "subjects": group.get("subjects"),
                "subset": group.get("subset"),
                "tasks": group.get("tasks_refined") or group.get("tasks") 
            }
            
            if t_type == "olmes":
                b_def["olmes_task"] = t_name
            elif t_type == "harness":
                b_def["harness_task"] = t_name
            elif t_type == "custom" or (isinstance(t_type, tuple) and t_type[0] == "custom"):
                # Handle both string and tuple formats
                b_def["custom_script"] = t_type[1] if isinstance(t_type, tuple) else t_name
            
            benchmarks_to_run.append(b_def)

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
    
    for b in benchmarks_to_run:
        current_limit = args.limit if args.limit is not None else b.get("limit")
        bench_name = b.get("olmes_task") or b.get("harness_task") or b.get("custom_script") or b["name"]
        
        logger.info(f"\n⏱  Starting benchmark: {bench_name}")
        bench_start = time.time()
        
        if b.get("olmes_task"):
            res = eval_runner.run_olmes_benchmark(
                b, args.model_args, eval_logger, run_dir, 
                limit=current_limit, device=args.device, batch_size=args.batch_size
            )
        elif b.get("harness_task"):
            res = eval_runner.run_harness_benchmark(
                b, args.model_args, eval_logger, run_dir, 
                limit=current_limit, device=args.device, batch_size=args.batch_size
            )
        elif b.get("custom_script"):
            res = eval_runner.run_custom_benchmark(
                b, args.model_args, eval_logger, run_dir, 
                limit=current_limit, device=args.device
            )
        
        bench_elapsed = time.time() - bench_start
        benchmark_timings.append((bench_name, bench_elapsed))
        logger.info(f"⏱  Finished benchmark: {bench_name} in {bench_elapsed:.1f}s ({bench_elapsed/60:.1f}m)")
        
        if res.get("status") == "success" and not res.get("subtasks"):
            # Enrichment: OLMES tasks might not return subtasks if eval_runner is kept vanilla
            if b.get("olmes_task"):
                task_safe = b["olmes_task"].replace("/", "_").replace(":", "_")
                metrics_path = os.path.join(run_dir, "olmes_raw", task_safe, "metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as mf:
                            res_data = json.load(mf)
                            if "tasks" in res_data and isinstance(res_data["tasks"], list):
                                res["subtasks"] = sorted([
                                    {"task": t.get("alias", "unknown"), "score": t.get("metrics", {}).get("primary_score", 0.0)}
                                    for t in res_data["tasks"]
                                ], key=lambda x: x["task"])
                    except Exception: pass

        # Store timing in the result too
        res["duration_seconds"] = round(bench_elapsed, 2)
        results["benchmarks"].append(res)
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
    
    benchmarks_elapsed = time.time() - benchmarks_start

    report_start = time.time()
    report_path = eval_runner.generate_summary_report(
        results, run_dir, capability_map=orchestrator_config.get("buckets", {}), 
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
