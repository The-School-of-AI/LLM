import yaml
import json
import argparse
import os
import subprocess
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_harness_benchmark(benchmark_info, model_args, is_test=False):
    """
    Executes a benchmark using the lm-evaluation-harness.
    """
    task = benchmark_info.get('harness_task')
    shots = benchmark_info.get('shots', 0)
    
    if is_test:
        print(f"  [Harness Test] Simulating task: {task}, shots: {shots}")
        return {
            "name": benchmark_info['name'],
            "type": "harness",
            "task": task,
            "score": 0.55,
            "metric": "acc",
            "status": "success"
        }
    
    # Construct lm-eval command
    # Basic usage: lm_eval --model hf --model_args pretrained=... --tasks mmlu --num_fewshot 5
    cmd = [
        "python3", "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", task,
        "--num_fewshot", str(shots),
        "--batch_size", "auto",
        "--output_path", "temp_harness_results.json"
    ]
    
    print(f"  [Harness] Running task: {task}...")
    try:
        subprocess.run(cmd, check=True)
        with open("temp_harness_results.json", 'r') as f:
            full_res = json.load(f)
            # Extract relevant score. This simplified mapping may need adjustment per task.
            task_res = full_res.get('results', {}).get(task, {})
            score = task_res.get('acc,none') or task_res.get('acc') or task_res.get('exact_match,none')
            return {
                "name": benchmark_info['name'],
                "type": "harness",
                "task": task,
                "score": score,
                "status": "success"
            }
    except Exception as e:
        return {"name": benchmark_info['name'], "type": "harness", "status": "failed", "error": str(e)}

def run_custom_benchmark(benchmark_info, model_args, is_test=False):
    """
    Executes a custom benchmark not covered by lm-evaluation-harness.
    """
    name = benchmark_info['name']
    script = benchmark_info.get('custom_script')
    
    if is_test:
        print(f"  [Custom Test] Simulating {name} via {script}")
        return {
            "name": name,
            "type": "custom",
            "score": 0.38,
            "metric": "custom_metric",
            "status": "success"
        }

    print(f"  [Custom] Running {name}...")
    if not script or not os.path.exists(script):
        return {"name": name, "type": "custom", "status": "failed", "error": f"Script {script} not found"}

    try:
        # Expect custom script to output JSON to stdout or a specific file
        result = subprocess.run(["python3", script, "--model_args", model_args], 
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except Exception as e:
        return {"name": name, "type": "custom", "status": "failed", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Multi-stage LLM Evaluator (Harness + Custom)")
    parser.add_argument("--config", type=str, required=True, help="Path to stage config")
    parser.add_argument("--phase", type=str, choices=["pretraining", "sft"], required=True, help="Training phase")
    parser.add_argument("--model_args", type=str, default="", help="Model args for harness (e.g. 'pretrained=eleutherai/polyglot-ko-1.3b')")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config {args.config} not found.")
        return

    config = load_config(args.config)
    stage = config.get('stage', 'unknown')
    
    print(f"--- Evaluation Stage: {stage.upper()} | Phase: {args.phase.upper()} ---")
    
    results = {
        "metadata": {
            "stage": stage,
            "phase": args.phase,
            "timestamp": datetime.now().isoformat(),
            "model_args": args.model_args,
            "is_test": args.test
        },
        "benchmarks": []
    }

    active_benchmarks = [b for b in config['benchmarks'] if b.get('enabled') and args.phase in b.get('phases', [])]

    for b in active_benchmarks:
        if b.get('harness_task'):
            res = run_harness_benchmark(b, args.model_args, is_test=args.test)
        elif b.get('custom_script'):
            res = run_custom_benchmark(b, args.model_args, is_test=args.test)
        else:
            res = {"name": b['name'], "status": "skipped", "reason": "No harness task or custom script provided"}
        
        results['benchmarks'].append(res)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    output_filename = f"eval_{stage}_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation Complete. Total: {len(results['benchmarks'])} benchmarks. Results: {output_path}")

if __name__ == "__main__":
    main()
