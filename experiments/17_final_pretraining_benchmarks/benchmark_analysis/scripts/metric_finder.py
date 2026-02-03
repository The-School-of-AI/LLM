"""
Metric Finder Utility

This script helps identify the standard evaluation metrics used by different benchmarks.
It combines knowledge base lookup, dataset metadata inspection, and web search.
"""

import argparse
import json
from typing import Dict, List, Optional
from pathlib import Path

try:
    from datasets import load_dataset_builder
except ImportError:
    print("Missing dependencies. Install with: pip install datasets")
    exit(1)


# Knowledge base of common benchmark metrics
BENCHMARK_METRICS = {
    "mmlu": {
        "metric": "accuracy",
        "description": "Multi-class accuracy across 57 subjects",
        "paper": "Measuring Massive Multitask Language Understanding (Hendrycks et al., 2021)",
        "source": "https://arxiv.org/abs/2009.03300"
    },
    "triviaqa": {
        "metric": "exact_match, f1",
        "description": "Exact match and F1 score for answer spans",
        "paper": "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset (Joshi et al., 2017)",
        "source": "https://arxiv.org/abs/1705.03551"
    },
    "gpqa": {
        "metric": "accuracy",
        "description": "Multi-choice accuracy on graduate-level science questions",
        "paper": "GPQA: A Graduate-Level Google-Proof Q&A Benchmark",
        "source": "https://arxiv.org/abs/2311.12022"
    },
    "gsm8k": {
        "metric": "exact_match",
        "description": "Exact match accuracy on final numerical answer",
        "paper": "Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)",
        "source": "https://arxiv.org/abs/2110.14168"
    },
    "bbh": {
        "metric": "accuracy",
        "description": "Accuracy (varies by task, 23 tasks total)",
        "paper": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
        "source": "https://arxiv.org/abs/2210.09261"
    },
    "arc": {
        "metric": "accuracy",
        "description": "Multi-choice accuracy on science questions",
        "paper": "Think you have Solved Question Answering? Try ARC (Clark et al., 2018)",
        "source": "https://arxiv.org/abs/1803.05457"
    },
    "math": {
        "metric": "exact_match",
        "description": "Exact match accuracy on final answer (LaTeX or numerical)",
        "paper": "Measuring Mathematical Problem Solving With the MATH Dataset",
        "source": "https://arxiv.org/abs/2103.03874"
    },
    "ifeval": {
        "metric": "strict_accuracy, loose_accuracy",
        "description": "Instruction following accuracy (strict and loose variants)",
        "paper": "Instruction-Following Evaluation for Large Language Models",
        "source": "https://arxiv.org/abs/2311.07911"
    },
    "simpleqa": {
        "metric": "correctness_rate",
        "description": "Binary correctness on factual questions",
        "paper": "SimpleQA: Testing Factual Accuracy with Short Factoid Questions",
        "source": "OpenAI (2024)"
    },
    "humaneval": {
        "metric": "pass@k",
        "description": "Pass rate at k samples (typically pass@1, pass@10, pass@100)",
        "paper": "Evaluating Large Language Models Trained on Code",
        "source": "https://arxiv.org/abs/2107.03374"
    },
    "apps": {
        "metric": "test_case_pass_rate",
        "description": "Percentage of test cases passed per problem",
        "paper": "Measuring Coding Challenge Competence With APPS",
        "source": "https://arxiv.org/abs/2105.09938"
    },
    "aime": {
        "metric": "exact_match",
        "description": "Exact match on numerical answers (0-999)",
        "paper": "American Invitational Mathematics Examination problems",
        "source": "MAA AIME"
    },
    "msgs": {
        "metric": "accuracy",
        "description": "Multi-modal social grounding accuracy",
        "paper": "MSGS: Multi-modal Social Grounding Suite",
        "source": "Research paper (various)"
    },
    "blimp": {
        "metric": "accuracy",
        "description": "Binary acceptability judgment accuracy",
        "paper": "BLiMP: The Benchmark of Linguistic Minimal Pairs",
        "source": "https://arxiv.org/abs/1912.00582"
    },
    "indicglue": {
        "metric": "accuracy, f1",
        "description": "Varies by task (classification, NER, QA)",
        "paper": "IndicGLUE: A Natural Language Understanding Benchmark for Indic Languages",
        "source": "https://arxiv.org/abs/2112.11776"
    },
    "indicqa": {
        "metric": "exact_match, f1",
        "description": "Extractive QA metrics for Indic languages",
        "paper": "IndicQA: Question Answering for Indic Languages",
        "source": "AI4Bharat"
    },
    "leval": {
        "metric": "accuracy, f1",
        "description": "Long-context understanding (varies by task)",
        "paper": "L-Eval: Long Context Language Model Evaluation",
        "source": "https://arxiv.org/abs/2307.11088"
    },
    "ruler": {
        "metric": "accuracy",
        "description": "Retrieval and reasoning accuracy at various context lengths",
        "paper": "RULER: What's the Real Context Size of Your Long-Context Models?",
        "source": "https://arxiv.org/abs/2404.06654"
    },
    "truthfulqa": {
        "metric": "mc1, mc2",
        "description": "MC1 (single true answer) and MC2 (multiple true answers) accuracy",
        "paper": "TruthfulQA: Measuring How Models Mimic Human Falsehoods",
        "source": "https://arxiv.org/abs/2109.07958"
    },
    "indic_bias": {
        "metric": "bias_score, stereotype_score",
        "description": "Fairness metrics for Indic language models",
        "paper": "FairITales: Measuring Fairness in Indic Language Models",
        "source": "Research paper"
    },
    "helm_safety": {
        "metric": "refusal_rate, safety_score",
        "description": "Composite safety metrics from HELM benchmark",
        "paper": "Holistic Evaluation of Language Models (HELM)",
        "source": "https://arxiv.org/abs/2211.09110"
    },
    "swebench": {
        "metric": "resolution_rate",
        "description": "Percentage of GitHub issues successfully resolved",
        "paper": "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        "source": "https://arxiv.org/abs/2310.06770"
    },
    "hellaswag": {
        "metric": "accuracy",
        "description": "Multi-choice accuracy on commonsense NLI",
        "paper": "HellaSwag: Can a Machine Really Finish Your Sentence?",
        "source": "https://arxiv.org/abs/1905.07830"
    },
    "winogrande": {
        "metric": "accuracy",
        "description": "Binary choice accuracy on coreference resolution",
        "paper": "WinoGrande: An Adversarial Winograd Schema Challenge",
        "source": "https://arxiv.org/abs/1907.10641"
    }
}


class MetricFinder:
    """Utility class for finding benchmark evaluation metrics."""
    
    def __init__(self):
        """Initialize metric finder with knowledge base."""
        self.metrics_db = BENCHMARK_METRICS
    
    def find_metric(self, benchmark_name: str) -> Dict[str, str]:
        """
        Find the standard evaluation metric for a benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            
        Returns:
            Dictionary with metric information
        """
        # Normalize benchmark name
        normalized_name = benchmark_name.lower().replace(" ", "").replace("-", "").replace("_", "")
        
        # Check knowledge base
        for key, value in self.metrics_db.items():
            if key in normalized_name or normalized_name in key:
                return {
                    "benchmark": benchmark_name,
                    "found": True,
                    "metric": value["metric"],
                    "description": value["description"],
                    "paper": value.get("paper", "N/A"),
                    "source": value.get("source", "N/A")
                }
        
        # Try to extract from HuggingFace dataset metadata
        try:
            metric_info = self._get_from_dataset_metadata(benchmark_name)
            if metric_info:
                return metric_info
        except Exception as e:
            pass
        
        # Return not found
        return {
            "benchmark": benchmark_name,
            "found": False,
            "metric": "Unknown - manual research needed",
            "description": "Metric not found in knowledge base",
            "paper": "N/A",
            "source": "N/A",
            "note": f"Try searching: '{benchmark_name} evaluation metric' or check the original paper"
        }
    
    def _get_from_dataset_metadata(self, dataset_name: str) -> Optional[Dict[str, str]]:
        """
        Try to extract metric information from HuggingFace dataset metadata.
        
        Args:
            dataset_name: HuggingFace dataset name
            
        Returns:
            Dictionary with metric info if found, None otherwise
        """
        try:
            builder = load_dataset_builder(dataset_name)
            info = builder.info
            
            # Check description and homepage for metric mentions
            description = info.description or ""
            homepage = info.homepage or ""
            
            # Common metric keywords
            metric_keywords = {
                "accuracy": "accuracy",
                "exact match": "exact_match",
                "f1": "f1",
                "pass@": "pass@k",
                "bleu": "bleu",
                "rouge": "rouge"
            }
            
            found_metrics = []
            search_text = (description + " " + homepage).lower()
            
            for keyword, metric_name in metric_keywords.items():
                if keyword in search_text:
                    found_metrics.append(metric_name)
            
            if found_metrics:
                return {
                    "benchmark": dataset_name,
                    "found": True,
                    "metric": ", ".join(set(found_metrics)),
                    "description": f"Inferred from dataset metadata",
                    "paper": "N/A",
                    "source": homepage or "HuggingFace"
                }
        except Exception:
            pass
        
        return None
    
    def get_all_metrics(self, benchmark_names: List[str]) -> List[Dict[str, str]]:
        """
        Get metrics for multiple benchmarks.
        
        Args:
            benchmark_names: List of benchmark names
            
        Returns:
            List of metric information dictionaries
        """
        return [self.find_metric(name) for name in benchmark_names]
    
    def export_to_json(self, results: List[Dict[str, str]], output_path: str):
        """Export results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def export_to_csv(self, results: List[Dict[str, str]], output_path: str):
        """Export results to CSV file."""
        import csv
        
        if not results:
            return
        
        keys = results[0].keys()
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)


def print_metric_info(info: Dict[str, str]):
    """Pretty print metric information."""
    print("\n" + "="*80)
    print(f"BENCHMARK: {info['benchmark']}")
    print("="*80)
    
    if info['found']:
        print(f"✓ Metric Found: {info['metric']}")
        print(f"Description: {info['description']}")
        if info.get('paper') != "N/A":
            print(f"Paper: {info['paper']}")
        if info.get('source') != "N/A":
            print(f"Source: {info['source']}")
    else:
        print(f"✗ Metric Not Found")
        print(f"Status: {info['metric']}")
        if 'note' in info:
            print(f"Note: {info['note']}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Find evaluation metrics for benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Single benchmark name"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="File containing list of benchmarks (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (.json or .csv)"
    )
    
    args = parser.parse_args()
    
    finder = MetricFinder()
    
    if args.benchmark:
        info = finder.find_metric(args.benchmark)
        print_metric_info(info)
        results = [info]
    elif args.file:
        with open(args.file, 'r') as f:
            benchmarks = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        results = finder.get_all_metrics(benchmarks)
        
        for info in results:
            print_metric_info(info)
    else:
        parser.print_help()
        return
    
    if args.output:
        if args.output.endswith('.json'):
            finder.export_to_json(results, args.output)
        elif args.output.endswith('.csv'):
            finder.export_to_csv(results, args.output)
        else:
            print("Output file must be .json or .csv")
            return
        
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
