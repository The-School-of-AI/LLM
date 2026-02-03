"""
Analyze All Benchmarks

Main orchestrator script that analyzes all benchmarks from the benchmarks-list.txt file.
For each benchmark, it:
1. Counts tokens in the test dataset
2. Identifies the evaluation metric used
3. Outputs results to CSV for easy reference
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from token_counter import TokenCounter
from metric_finder import MetricFinder


# Dataset mapping: benchmark name -> HuggingFace dataset info
DATASET_MAPPING = {
    # "MMLU": {"name": "cais/mmlu", "split": "test", "config": "all"},
    # "TriviaQA": {"name": "trivia_qa", "split": "validation", "config": "unfiltered"},
    # "GPQA Diamond": {"name": "Idavidrein/gpqa", "split": "test", "config": "gpqa_diamond"},
    # "GSM8K": {"name": "gsm8k", "split": "test", "config": "main"},
    # "BBH (Big Bench Hard)": {"name": "lukaemon/bbh", "split": "test"},
    # "ARC-Challenge": {"name": "ai2_arc", "split": "test", "config": "ARC-Challenge"},
    # "MATH": {"name": "hendrycks/math", "split": "test", "config": "all"},
    # "IFEval": {"name": "google/IFEval", "split": "train"},  # No public test set
    # "SimpleQA_Verified": {"name": "openai/simple-qa", "split": "test"},
    # "HumanEval": {"name": "openai_humaneval", "split": "test"},
    # "APPS": {"name": "codeparrot/apps", "split": "test", "config": "all"},
    # "AIME 2025": {"name": None, "note": "Manual dataset - not on HuggingFace"},
    # "MSGS": {"name": None, "note": "Manual dataset - not on HuggingFace"},
    # "BLiMP": {"name": "blimp", "split": "test"},
    "IndicGLUE": {"name": "ai4bharat/indic_glue", "split": "test", "config": "actsa-sc.te"},
    # "IndicQA": {"name": "ai4bharat/IndicQA", "split": "test"},
    # "L-Eval (Long Context Evaluation Suite)": {"name": "L4NLP/LEval", "split": "test"},
    # "RULER": {"name": None, "note": "Synthetic dataset - generate on demand"},
    # "TruthfulQA": {"name": "truthful_qa", "split": "validation", "config": "multiple_choice"},
    # "Indic-Bias (FairITales)": {"name": None, "note": "Manual dataset - not on HuggingFace"},
    # "HELM Safety": {"name": None, "note": "Composite benchmark - multiple datasets"},
    # "SWE-bench Verified": {"name": "princeton-nlp/SWE-bench_Verified", "split": "test"},
    # "HellaSwag": {"name": "hellaswag", "split": "validation"},
    # "Winogrande": {"name": "winogrande", "split": "validation", "config": "winogrande_xl"}
}


class BenchmarkAnalyzer:
    """Main analyzer that coordinates token counting and metric finding."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        """
        Initialize the analyzer.
        
        Args:
            tokenizer_name: Tokenizer to use for counting tokens
        """
        self.token_counter = TokenCounter(tokenizer_name)
        self.metric_finder = MetricFinder()
        self.results = []
    
    def analyze_benchmark(self, benchmark_name: str, 
                         dataset_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a single benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            dataset_info: Dataset information (name, split, config)
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\n{'='*80}")
        print(f"Analyzing: {benchmark_name}")
        print(f"{'='*80}")
        
        result = {
            "benchmark": benchmark_name,
            "dataset_name": None,
            "split": None,
            "test_set_size": None,
            "total_tokens": None,
            "avg_tokens_per_sample": None,
            "min_tokens": None,
            "max_tokens": None,
            "metric": None,
            "metric_description": None,
            "paper": None,
            "status": "unknown",
            "notes": None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Get metric information
        print("\n[1/2] Finding evaluation metric...")
        metric_info = self.metric_finder.find_metric(benchmark_name)
        result["metric"] = metric_info.get("metric", "Unknown")
        result["metric_description"] = metric_info.get("description", "N/A")
        result["paper"] = metric_info.get("paper", "N/A")
        
        if metric_info.get("found"):
            print(f"✓ Metric: {result['metric']}")
        else:
            print(f"✗ Metric not found - manual research needed")
        
        # Get token counts
        print("\n[2/2] Counting tokens in test dataset...")
        
        if dataset_info is None:
            dataset_info = DATASET_MAPPING.get(benchmark_name, {})
        
        if dataset_info.get("name"):
            result["dataset_name"] = dataset_info["name"]
            result["split"] = dataset_info.get("split", "test")
            
            try:
                load_kwargs = {}
                if "config" in dataset_info and dataset_info["config"]:
                    load_kwargs["name"] = dataset_info["config"]
                
                token_stats = self.token_counter.count_dataset(
                    dataset_info["name"],
                    split=result["split"],
                    **load_kwargs
                )
                
                if "error" not in token_stats:
                    result["test_set_size"] = token_stats["num_samples"]
                    result["total_tokens"] = token_stats["total_tokens"]
                    result["avg_tokens_per_sample"] = round(token_stats["mean_tokens"], 2)
                    result["min_tokens"] = token_stats["min_tokens"]
                    result["max_tokens"] = token_stats["max_tokens"]
                    result["status"] = "success"
                    
                    print(f"✓ Analyzed {result['test_set_size']:,} samples")
                    print(f"  Total tokens: {result['total_tokens']:,}")
                    print(f"  Avg tokens/sample: {result['avg_tokens_per_sample']:.2f}")
                else:
                    result["status"] = "error"
                    result["notes"] = token_stats["error"]
                    print(f"✗ Error: {token_stats['error']}")
                    
            except Exception as e:
                result["status"] = "error"
                result["notes"] = str(e)
                print(f"✗ Error loading dataset: {e}")
        else:
            result["status"] = "manual"
            result["notes"] = dataset_info.get("note", "Dataset not available on HuggingFace")
            print(f"⚠ Manual analysis required: {result['notes']}")
        
        return result
    
    def analyze_all(self, benchmark_names: List[str], 
                   output_dir: Path = None) -> List[Dict[str, Any]]:
        """
        Analyze all benchmarks in the list.
        
        Args:
            benchmark_names: List of benchmark names
            output_dir: Directory to save results
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i, benchmark_name in enumerate(benchmark_names, 1):
            print(f"\n\nProgress: [{i}/{len(benchmark_names)}]")
            
            result = self.analyze_benchmark(benchmark_name)
            results.append(result)
            
            # Save intermediate results
            if output_dir:
                self._save_results(results, output_dir)
        
        self.results = results
        return results
    
    def _save_results(self, results: List[Dict[str, Any]], output_dir: Path):
        """Save results to CSV and JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / "benchmark_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        
        # Save as JSON (more detailed)
        json_path = output_dir / "benchmark_summary.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to:")
        print(f"  - {csv_path}")
        print(f"  - {json_path}")
    
    def print_summary(self):
        """Print a summary of the analysis."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        error = sum(1 for r in self.results if r["status"] == "error")
        manual = sum(1 for r in self.results if r["status"] == "manual")
        
        print(f"\nTotal benchmarks: {total}")
        print(f"  ✓ Successfully analyzed: {success}")
        print(f"  ✗ Errors: {error}")
        print(f"  ⚠ Manual analysis needed: {manual}")
        
        # Calculate total tokens across all benchmarks
        total_tokens = sum(r["total_tokens"] for r in self.results 
                          if r["total_tokens"] is not None)
        total_samples = sum(r["test_set_size"] for r in self.results 
                           if r["test_set_size"] is not None)
        
        if total_tokens > 0:
            print(f"\nAggregate statistics:")
            print(f"  Total test samples: {total_samples:,}")
            print(f"  Total tokens (all benchmarks): {total_tokens:,}")
            print(f"  Average tokens per benchmark: {total_tokens // success:,}")
        
        print("\n" + "="*80 + "\n")


def load_benchmarks_list(file_path: Path) -> List[str]:
    """Load benchmark names from file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header and empty lines
    benchmarks = []
    for line in lines[1:]:  # Skip first line (header)
        line = line.strip()
        if line and not line.startswith('#'):
            benchmarks.append(line)
    
    return benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Analyze all benchmarks for token counts and evaluation metrics"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Analyze a single benchmark by name"
    )
    parser.add_argument(
        "--benchmarks-file",
        type=str,
        default="../benchmarks-list.txt",
        help="Path to benchmarks list file (default: ../benchmarks-list.txt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Output directory for results (default: ../results)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use (default: gpt2)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(tokenizer_name=args.tokenizer)
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    benchmarks_file = script_dir / args.benchmarks_file
    output_dir = script_dir / args.output_dir
    
    if args.benchmark:
        # Analyze single benchmark
        result = analyzer.analyze_benchmark(args.benchmark)
        analyzer.results = [result]
        analyzer._save_results([result], output_dir)
    else:
        # Analyze all benchmarks
        if not benchmarks_file.exists():
            print(f"Error: Benchmarks file not found: {benchmarks_file}")
            print(f"Please ensure {benchmarks_file} exists")
            return
        
        print(f"Loading benchmarks from: {benchmarks_file}")
        benchmarks = load_benchmarks_list(benchmarks_file)
        print(f"Found {len(benchmarks)} benchmarks to analyze\n")
        
        # Run analysis
        analyzer.analyze_all(benchmarks, output_dir)
    
    # Print summary
    analyzer.print_summary()


if __name__ == "__main__":
    main()
