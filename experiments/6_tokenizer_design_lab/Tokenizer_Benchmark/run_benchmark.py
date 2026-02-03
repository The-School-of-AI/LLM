#!/usr/bin/env python3
"""
Tokenizer Benchmark Runner

Main entry point for running tokenizer benchmarks.
Supports benchmark-neutral evaluation with format-only probes.

Usage:
    python run_benchmark.py --config config.yaml
    python run_benchmark.py --config config.yaml --dry-run
    python run_benchmark.py --tokenizers custom,gpt4o
"""

import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tokenizer_loader import load_tokenizer, load_tokenizers_from_config, TokenizerInterface
from probes import (
    MathProbeGenerator,
    MCQProbeGenerator,
    CodeProbeGenerator,
    IndicProbeGenerator,
    SyntheticInstructionGenerator,
)
from metrics import (
    compute_compression_metrics,
    compute_fertility_metrics,
    benchmark_speed,
    compute_code_quality_metrics,
)
from validation import (
    NeutralityChecker,
    CurriculumAnalyzer,
    RoutingSkewAnalyzer,
    check_format_only_compliance,
)
from reporting import generate_report
from reporting.charts import save_all_charts


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_probes(config: Dict[str, Any], seed: int = 42) -> Dict[str, List[str]]:
    """Generate all format-only probes based on config."""
    probe_config = config.get('probes', {})
    count = probe_config.get('count_per_category', 500)
    
    probes = {
        'all': [],
        'by_type': {},
        'by_difficulty': {'easy': [], 'medium': [], 'hard': []},
    }
    
    # Math probes
    if probe_config.get('math', {}).get('enabled', True):
        gen = MathProbeGenerator(seed=seed)
        math_probes = gen.generate_batch(count)
        probes['by_type']['math'] = [p.content for p in math_probes]
        probes['all'].extend(probes['by_type']['math'])
        for p in math_probes:
            probes['by_difficulty'][p.difficulty].append(p.content)
    
    # MCQ probes
    if probe_config.get('mcq', {}).get('enabled', True):
        gen = MCQProbeGenerator(seed=seed)
        mcq_probes = gen.generate_batch(count)
        probes['by_type']['mcq'] = [p.content for p in mcq_probes]
        probes['all'].extend(probes['by_type']['mcq'])
        for p in mcq_probes:
            probes['by_difficulty'][p.difficulty].append(p.content)
    
    # Code probes
    if probe_config.get('code', {}).get('enabled', True):
        gen = CodeProbeGenerator(seed=seed)
        code_probes = gen.generate_batch(count)
        probes['by_type']['code'] = [p.content for p in code_probes]
        probes['all'].extend(probes['by_type']['code'])
        for p in code_probes:
            probes['by_difficulty'][p.difficulty].append(p.content)
    
    # Indic probes
    if probe_config.get('indic', {}).get('enabled', True):
        scripts = probe_config.get('indic', {}).get('scripts', ['devanagari', 'tamil'])
        gen = IndicProbeGenerator(seed=seed)
        indic_probes = gen.generate_batch(count, scripts=scripts)
        probes['by_type']['indic'] = [p.content for p in indic_probes]
        probes['all'].extend(probes['by_type']['indic'])
        for p in indic_probes:
            probes['by_difficulty'][p.difficulty].append(p.content)
    
    # Synthetic instructions
    gen = SyntheticInstructionGenerator(seed=seed)
    instr_probes = gen.generate_batch(count // 2)
    probes['by_type']['instructions'] = [p.content for p in instr_probes]
    probes['all'].extend(probes['by_type']['instructions'])
    
    return probes


def run_metrics(
    tokenizers: Dict[str, TokenizerInterface],
    probes: Dict[str, List[str]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Run all metrics on tokenizers."""
    results = {}
    metrics_config = config.get('metrics', {})
    all_probes = probes['all']
    
    for tok_name, tokenizer in tokenizers.items():
        print(f"  Evaluating: {tok_name}...")
        tok_results = {}
        
        # Compression metrics
        if metrics_config.get('compression', {}).get('enabled', True):
            tok_results['compression'] = compute_compression_metrics(tokenizer, all_probes)
        
        # Fertility metrics
        if metrics_config.get('fertility', {}).get('enabled', True):
            tok_results['fertility'] = compute_fertility_metrics(tokenizer, all_probes)
        
        # Speed metrics
        if metrics_config.get('speed', {}).get('enabled', True):
            iterations = metrics_config.get('speed', {}).get('iterations', 100)
            warmup = metrics_config.get('speed', {}).get('warmup_iterations', 10)
            # Use subset for speed testing
            speed_probes = all_probes[:100] if len(all_probes) > 100 else all_probes
            tok_results['speed'] = benchmark_speed(tokenizer, speed_probes, iterations, warmup)
        
        # Code quality metrics
        if metrics_config.get('code_quality', {}).get('enabled', True):
            tok_results['code_quality'] = compute_code_quality_metrics(tokenizer)
        
        results[tok_name] = tok_results
    
    return results


def run_validation(
    tokenizers: Dict[str, TokenizerInterface],
    probes: Dict[str, List[str]],
    config: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """Run validation checks."""
    validation_config = config.get('validation', {})
    results = {}
    
    all_probes = probes['all']
    
    # Neutrality check on probes
    if validation_config.get('mirroring_detection', {}).get('enabled', True):
        sensitivity = validation_config.get('mirroring_detection', {}).get('sensitivity', 'high')
        checker = NeutralityChecker(sensitivity=sensitivity)
        probe_result = check_format_only_compliance(all_probes)
        results['probe_neutrality'] = {
            'passed': probe_result.passed,
            'score': probe_result.score,
            'issues': probe_result.issues,
        }
    
    # Curriculum analysis
    if validation_config.get('curriculum_analysis', {}).get('enabled', True):
        analyzer = CurriculumAnalyzer()
        for tok_name, tokenizer in tokenizers.items():
            curriculum_result = analyzer.analyze(tokenizer, probes['by_difficulty'])
            results[f'{tok_name}_curriculum'] = {
                'passed': curriculum_result.passed,
                'distortion_score': curriculum_result.distortion_score,
                'issues': curriculum_result.issues,
            }
    
    # Routing skew
    if validation_config.get('routing_skew', {}).get('enabled', True):
        threshold = validation_config.get('routing_skew', {}).get('entropy_threshold', 0.8)
        analyzer = RoutingSkewAnalyzer(entropy_threshold=threshold)
        for tok_name, tokenizer in tokenizers.items():
            skew_result = analyzer.analyze(tokenizer, all_probes)
            results[f'{tok_name}_routing'] = {
                'passed': not skew_result.skew_detected,
                'entropy': skew_result.normalized_entropy,
                'issues': skew_result.issues,
            }
    
    return results


def save_report(
    results: Dict[str, Dict[str, Any]],
    validation_results: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """Save benchmark report to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    formats = config.get('output', {}).get('format', ['markdown'])
    
    # Generate reports
    for fmt in formats:
        report = generate_report(results, validation_results, fmt)
        ext = 'md' if fmt == 'markdown' else 'html'
        filename = output_path / f"benchmark_report.{ext}"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        saved_files.append(str(filename))
        print(f"  Saved: {filename}")
    
    # Generate charts
    if config.get('output', {}).get('include_charts', True):
        chart_files = save_all_charts(results, str(output_path))
        saved_files.extend(chart_files)
        for cf in chart_files:
            print(f"  Saved: {cf}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description="Tokenizer Benchmark Framework - Benchmark-neutral evaluation"
    )
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--tokenizers', '-t',
        default=None,
        help='Comma-separated list of tokenizers to evaluate'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without loading actual tokenizers (uses mock data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for probe generation'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\n{'='*60}")
    print("TOKENIZER BENCHMARK FRAMEWORK")
    print(f"{'='*60}\n")
    
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    if args.dry_run:
        print("  [DRY RUN] Using mock tokenizer")
        # Create a simple mock tokenizer for testing
        class MockTokenizer:
            name = "mock"
            def encode(self, text: str) -> List[int]:
                # Simple mock: one token per 4 characters
                return list(range(len(text) // 4 + 1))
            def decode(self, ids: List[int]) -> str:
                return "mock" * len(ids)
            def vocab_size(self) -> int:
                return 50000
        
        tokenizers = {'mock': MockTokenizer()}
    else:
        tokenizers = load_tokenizers_from_config(config)
        
        # Filter if specific tokenizers requested
        if args.tokenizers:
            requested = set(args.tokenizers.split(','))
            tokenizers = {k: v for k, v in tokenizers.items() if k in requested}
    
    if not tokenizers:
        print("ERROR: No tokenizers loaded. Check your config or install required packages.")
        sys.exit(1)
    
    print(f"  Loaded {len(tokenizers)} tokenizer(s): {', '.join(tokenizers.keys())}")
    
    # Generate probes
    print("\nGenerating format-only probes...")
    probes = generate_probes(config, seed=args.seed)
    print(f"  Generated {len(probes['all'])} total probes")
    for probe_type, probe_list in probes['by_type'].items():
        print(f"    - {probe_type}: {len(probe_list)}")
    
    # Run metrics
    print("\nRunning metrics...")
    results = run_metrics(tokenizers, probes, config)
    
    # Run validation
    print("\nRunning validation checks...")
    validation_results = run_validation(tokenizers, probes, config)
    
    # Generate report
    output_dir = args.output or config.get('output', {}).get('report_dir', 'reports')
    print(f"\nGenerating report...")
    saved_files = save_report(results, validation_results, config, output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}/")
    
    # Print quick summary
    print("\nQuick Summary:")
    for tok_name, tok_results in results.items():
        comp = tok_results.get('compression', {})
        if comp:
            tpb = comp.get('tokens_per_byte', {})
            if isinstance(tpb, dict):
                print(f"  {tok_name}: {tpb.get('mean', 0):.4f} tokens/byte")
    
    # Validation summary
    passed = sum(1 for v in validation_results.values() if v.get('passed', False))
    total = len(validation_results)
    print(f"\nValidation: {passed}/{total} checks passed")
    
    print()


if __name__ == "__main__":
    main()
