"""
IndicGLUE Benchmark Analysis

Analyzes all configs/subtasks in the IndicGLUE benchmark.
Computes token counts and identifies metrics for this multi-task Indic language benchmark.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from token_counter import TokenCounter
from metric_finder import MetricFinder


# All IndicGLUE configs (60+ subtasks across different languages)
INDICGLUE_CONFIGS = [
    # Sentiment Analysis
    'actsa-sc.te',
    
    # BBC News Classification
    'bbca.hi',
    
    # COPA (Choice of Plausible Alternatives)
    'copa.en', 'copa.gu', 'copa.hi', 'copa.mr',
    
    # CommonsenseQA
    'csqa.as', 'csqa.bn', 'csqa.gu', 'csqa.hi', 'csqa.kn', 
    'csqa.ml', 'csqa.mr', 'csqa.or', 'csqa.pa', 'csqa.ta', 'csqa.te',
    
    # Cross-lingual Sentence Retrieval
    'cvit-mkb-clsr.en-bn', 'cvit-mkb-clsr.en-gu', 'cvit-mkb-clsr.en-hi',
    'cvit-mkb-clsr.en-ml', 'cvit-mkb-clsr.en-mr', 'cvit-mkb-clsr.en-or',
    'cvit-mkb-clsr.en-ta', 'cvit-mkb-clsr.en-te', 'cvit-mkb-clsr.en-ur',
    
    # IITP Movie/Product Reviews
    'iitp-mr.hi', 'iitp-pr.hi',
    
    # Headline Classification
    'inltkh.gu', 'inltkh.ml', 'inltkh.mr', 'inltkh.ta', 'inltkh.te',
    
    # Discourse Analysis
    'md.hi',
    
    # News Article Classification
    'sna.bn',
    
    # Named Entity Recognition
    'wiki-ner.as', 'wiki-ner.bn', 'wiki-ner.gu', 'wiki-ner.hi', 
    'wiki-ner.kn', 'wiki-ner.ml', 'wiki-ner.mr', 'wiki-ner.or', 
    'wiki-ner.pa', 'wiki-ner.ta', 'wiki-ner.te',
    
    # Winograd NLI
    'wnli.en', 'wnli.gu', 'wnli.hi', 'wnli.mr',
    
    # Wikipedia Section Title Prediction
    'wstp.as', 'wstp.bn', 'wstp.gu', 'wstp.hi', 'wstp.kn', 
    'wstp.ml', 'wstp.mr', 'wstp.or', 'wstp.pa', 'wstp.ta', 'wstp.te'
]


def analyze_indicglue(tokenizer_name: str = "Xenova/gpt-4", output_dir: Path = None, configs_to_analyze: list = None):
    """
    Analyze all IndicGLUE configs.
    
    Args:
        tokenizer_name: Tokenizer to use for counting (default: Xenova/gpt-4 using cl100k_base)
        output_dir: Directory to save results
        configs_to_analyze: List of configs to analyze (defaults to all)
    """
    if configs_to_analyze is None:
        configs_to_analyze = INDICGLUE_CONFIGS
    
    print("\n" + "="*80)
    print("IndicGLUE Benchmark Analysis")
    print("="*80)
    print(f"\nTotal configs to analyze: {len(configs_to_analyze)}")
    print(f"Tokenizer: {tokenizer_name}\n")
    
    # Initialize utilities
    counter = TokenCounter(tokenizer_name=tokenizer_name)
    metric_finder = MetricFinder()
    
    # Get metric information
    metric_info = metric_finder.find_metric("IndicGLUE")
    print(f"Benchmark Metric: {metric_info.get('metric', 'Unknown')}")
    print(f"Description: {metric_info.get('description', 'N/A')}\n")
    
    results = []
    successful = 0
    failed = 0
    
    # Analyze each config
    for i, config in enumerate(configs_to_analyze, 1):
        print(f"[{i}/{len(INDICGLUE_CONFIGS)}] Analyzing: {config}")
        
        try:
            stats = counter.count_dataset(
                "ai4bharat/indic_glue",
                split="test",
                name=config
            )
            
            results.append({
                "config": config,
                "status": "success",
                "dataset_name": "ai4bharat/indic_glue",
                "split": "test",
                "num_samples": stats["num_samples"],
                "total_tokens": stats["total_tokens"],
                "mean_tokens": stats["mean_tokens"],
                "min_tokens": stats["min_tokens"],
                "max_tokens": stats["max_tokens"],
                "tokenizer": tokenizer_name,
                "timestamp": datetime.now().isoformat()
            })
            
            successful += 1
            print(f"  ✓ {stats['num_samples']:,} samples | {stats['total_tokens']:,} tokens | "
                  f"avg: {stats['mean_tokens']:.1f}\n")
            
        except Exception as e:
            results.append({
                "config": config,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
            failed += 1
            print(f"  ✗ Error: {e}\n")
    
    # Calculate aggregates
    total_samples = sum(r.get('num_samples', 0) for r in results if r['status'] == 'success')
    total_tokens = sum(r.get('total_tokens', 0) for r in results if r['status'] == 'success')
    avg_tokens = total_tokens / total_samples if total_samples > 0 else 0
    
    # Create summary
    summary = {
        "benchmark": "IndicGLUE",
        "dataset_name": "ai4bharat/indic_glue",
        "total_configs": len(configs_to_analyze),
        "successful_configs": successful,
        "failed_configs": failed,
        "aggregate_stats": {
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "avg_tokens_per_sample": round(avg_tokens, 2)
        },
        "metric": metric_info.get('metric', 'Unknown'),
        "metric_description": metric_info.get('description', 'N/A'),
        "paper": metric_info.get('paper', 'N/A'),
        "tokenizer": tokenizer_name,
        "timestamp": datetime.now().isoformat(),
        "per_config_results": results
    }
    
    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "indicglue_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Benchmark: IndicGLUE")
    print(f"Total configs: {len(configs_to_analyze)}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"\nAggregate Statistics:")
    print(f"  Total test samples: {total_samples:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Average tokens/sample: {avg_tokens:.2f}")
    print(f"\nEvaluation Metric: {metric_info.get('metric', 'Unknown')}")
    print(f"\nResults saved to: {output_file}")
    print("="*80 + "\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze IndicGLUE benchmark - all 60+ configs"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Xenova/gpt-4",
        help="Tokenizer to use (default: Xenova/gpt-4 using cl100k_base for FLOPS)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Analyze only a specific config (e.g., 'copa.hi')"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    if args.config:
        # Analyze single config
        if args.config not in INDICGLUE_CONFIGS:
            print(f"Error: Config '{args.config}' not found.")
            print(f"Available configs: {', '.join(INDICGLUE_CONFIGS[:10])}...")
            return
        
        # Pass only the selected config
        analyze_indicglue(
            tokenizer_name=args.tokenizer, 
            output_dir=output_dir,
            configs_to_analyze=[args.config]
        )
    else:
        # Analyze all configs
        analyze_indicglue(
            tokenizer_name=args.tokenizer, 
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
